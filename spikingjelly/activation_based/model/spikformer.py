from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from .. import functional, layer, memopt, neuron
from ..distributed.tensor_parallel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)
from ..distributed.tensor_parallel.channel import _ColwiseBackwardAllReduce
from ..distributed.vision.config import ModelBuilder, ModelConfig
from ..layer.attention import SpikingSelfAttention

__all__ = [
    "Spikformer",
    "SpikformerCIFAR10Config",
    "SpikformerConfig",
    "spikformer_cifar10",
    "spikformer_s",
    "spikformer_ti",
]


class SpikformerConv2dBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <SpikformerConv2dBN-cn>` | :ref:`English <SpikformerConv2dBN-en>`

        ----

        .. _SpikformerConv2dBN-cn:

        * **中文**

        ``Conv2d`` + ``BatchNorm2d`` 的组合模块。可选是否在最后添加 ``MaxPool2d`` (kernel_size=3, stride=2, padding=1)。

        :param in_channels: 输入图像的通道数
        :type in_channels: int

        :param out_channels: 输出通道数
        :type out_channels: int

        :param kernel_size: 卷积核大小
        :type kernel_size: int

        :param stride: 卷积步长。默认为 1
        :type stride: int

        :param padding: 卷积填充。默认为 0
        :type padding: int

        :param pool: 若为 ``True``，则在最后添加 ``MaxPool2d(kernel_size=3, stride=2, padding=1)``。默认为 ``False``
        :type pool: bool

        ----

        .. _SpikformerConv2dBN-en:

        * **English**

        A sequential block of ``Conv2d`` + ``BatchNorm2d``. When ``pool`` is ``True``, a ``MaxPool2d(kernel_size=3, stride=2, padding=1)`` is appended after batch norm.

        :param in_channels: Number of channels in the input image
        :type in_channels: int

        :param out_channels: Number of output channels
        :type out_channels: int

        :param kernel_size: Size of the convolution kernel
        :type kernel_size: int

        :param stride: Stride of the convolution. Default: 1
        :type stride: int

        :param padding: Padding added to both sides of the input. Default: 0
        :type padding: int

        :param pool: If ``True``, appends ``MaxPool2d(kernel_size=3, stride=2, padding=1)``. Default: ``False``
        :type pool: bool
        """
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block = layer.SeqToANNContainer(*layers)

    def forward(self, x_seq: torch.Tensor):
        return self.block(x_seq)


class SpikformerConv2dBNLIF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        r"""
        **API Language** - :ref:`中文 <SpikformerConv2dBNLIF-cn>` | :ref:`English <SpikformerConv2dBNLIF-en>`

        ----

        .. _SpikformerConv2dBNLIF-cn:

        * **中文**

        ``Conv2d`` + ``BatchNorm2d`` + ``LIFNode`` 的组合模块，支持多步模式。内部使用 ``SpikformerConv2dBN`` 进行卷积和批归一化，后接一个 ``LIFNode`` 脉冲神经元。

        :param in_channels: 输入图像的通道数
        :type in_channels: int

        :param out_channels: 输出通道数
        :type out_channels: int

        :param kernel_size: 卷积核大小
        :type kernel_size: int

        :param stride: 卷积步长。默认为 1
        :type stride: int

        :param padding: 卷积填充。默认为 0
        :type padding: int

        :param pool: 若为 ``True``，则在 ``SpikformerConv2dBN`` 中添加最大池化层。默认为 ``False``
        :type pool: bool

        :param backend: 神经元后端。默认为 ``"torch"``
        :type backend: str

        :param tau: ``LIFNode`` 的膜电位时间常数。默认为 2.0
        :type tau: float

        :param detach_reset: 是否在重置时断开计算图。默认为 ``True``
        :type detach_reset: bool

        ----

        .. _SpikformerConv2dBNLIF-en:

        * **English**

        A sequential module combining ``Conv2d`` + ``BatchNorm2d`` + ``LIFNode`` with multi-step support. Uses ``SpikformerConv2dBN`` internally for convolution and batch normalization, followed by a ``LIFNode`` spiking neuron.

        :param in_channels: Number of channels in the input image
        :type in_channels: int

        :param out_channels: Number of output channels
        :type out_channels: int

        :param kernel_size: Size of the convolution kernel
        :type kernel_size: int

        :param stride: Stride of the convolution. Default: 1
        :type stride: int

        :param padding: Padding added to both sides of the input. Default: 0
        :type padding: int

        :param pool: If ``True``, adds max-pooling inside ``SpikformerConv2dBN``. Default: ``False``
        :type pool: bool

        :param backend: Backend for the LIF neuron. Default: ``"torch"``
        :type backend: str

        :param tau: Membrane time constant of the ``LIFNode``. Default: 2.0
        :type tau: float

        :param detach_reset: Whether to detach the computational graph on reset. Default: ``True``
        :type detach_reset: bool
        """
        super().__init__()
        self.conv_bn = SpikformerConv2dBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool=pool,
        )
        self.neuron = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    def forward(self, x_seq: torch.Tensor):
        return self.neuron(self.conv_bn(x_seq))


class SpikformerPatchStem(nn.Module):
    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 256,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        r"""
        **API Language** - :ref:`中文 <SpikformerPatchStem-cn>` | :ref:`English <SpikformerPatchStem-en>`

        ----

        .. _SpikformerPatchStem-cn:

        * **中文**

        图像分块嵌入 (patch embedding) 模块，由 4 个卷积阶段和 1 个位置编码卷积组成。
        ``patch_size=4`` 时仅后两个阶段池化；``patch_size=16`` 时全部阶段池化。

        :param img_size_h: 输入图像高度。默认为 224
        :type img_size_h: int

        :param img_size_w: 输入图像宽度。默认为 224
        :type img_size_w: int

        :param patch_size: 分块大小，支持 4 或 16。4 仅在后两个卷积阶段下采样，
            16 在全部四个阶段下采样。
        :type patch_size: int

        :param in_channels: 输入图像的通道数。默认为 3
        :type in_channels: int

        :param embed_dims: 最终的嵌入维度。默认为 256
        :type embed_dims: int

        :param backend: 神经元后端。默认为 ``"torch"``
        :type backend: str

        :param tau: ``LIFNode`` 的膜电位时间常数。默认为 2.0
        :type tau: float

        :param detach_reset: 是否在重置时断开计算图。默认为 ``True``
        :type detach_reset: bool

        :raises ValueError: 当 ``patch_size`` 不是 4 或 16 时抛出

        ----

        .. _SpikformerPatchStem-en:

        * **English**

        Image patch embedding stem with four convolution stages and one positional
        encoding convolution. Only the last two stages pool for ``patch_size=4``;
        all four stages pool for ``patch_size=16``.

        :param img_size_h: Input image height. Default: 224
        :type img_size_h: int

        :param img_size_w: Input image width. Default: 224
        :type img_size_w: int

        :param patch_size: Patch size, either 4 or 16. Size 4 downsamples only in
            the last two convolution stages; size 16 downsamples in all four stages.
        :type patch_size: int

        :param in_channels: Number of channels in the input image. Default: 3
        :type in_channels: int

        :param embed_dims: Final embedding dimension. Default: 256
        :type embed_dims: int

        :param backend: Backend for the LIF neuron. Default: ``"torch"``
        :type backend: str

        :param tau: Membrane time constant of the ``LIFNode``. Default: 2.0
        :type tau: float

        :param detach_reset: Whether to detach the computational graph on reset. Default: ``True``
        :type detach_reset: bool

        :raises ValueError: If ``patch_size`` is not 4 or 16
        """
        super().__init__()
        if patch_size not in {4, 16}:
            raise ValueError(
                "SpikformerPatchStem supports patch_size=4 or 16, "
                f"but got {patch_size}."
            )
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.embed_dims = embed_dims

        stage_dims = [embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims]
        layers = []
        in_c = in_channels
        pool_from = 2 if patch_size == 4 else 0
        for index, out_c in enumerate(stage_dims):
            layers.append(
                SpikformerConv2dBNLIF(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool=index >= pool_from,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
            )
            in_c = out_c
        self.stages = nn.Sequential(*layers)
        self.positional_encoding = SpikformerConv2dBNLIF(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=False,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )
        self.grid_size = (img_size_h // patch_size, img_size_w // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x_seq: torch.Tensor):
        x_seq = self.stages(x_seq)
        residual = x_seq
        x_seq = self.positional_encoding(x_seq)
        return x_seq + residual


class SpikformerMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        r"""
        **API Language** - :ref:`中文 <SpikformerMLP-cn>` | :ref:`English <SpikformerMLP-en>`

        ----

        .. _SpikformerMLP-cn:

        * **中文**

        脉冲 MLP 模块，包含两个 ``Conv1d`` 层（kernel_size=1）和两个 ``LIFNode`` 脉冲神经元。支持多步模式。

        :param in_features: 输入特征维度
        :type in_features: int

        :param hidden_features: 隐藏层特征维度
        :type hidden_features: int

        :param out_features: 输出特征维度
        :type out_features: int

        :param backend: 神经元后端。默认为 ``"torch"``
        :type backend: str

        :param tau: ``LIFNode`` 的膜电位时间常数。默认为 2.0
        :type tau: float

        :param detach_reset: 是否在重置时断开计算图。默认为 ``True``
        :type detach_reset: bool

        ----

        .. _SpikformerMLP-en:

        * **English**

        Spiking MLP block consisting of two ``Conv1d`` layers (kernel_size=1) and two ``LIFNode`` spiking neurons. Supports multi-step mode.

        :param in_features: Input feature dimension
        :type in_features: int

        :param hidden_features: Hidden feature dimension
        :type hidden_features: int

        :param out_features: Output feature dimension
        :type out_features: int

        :param backend: Backend for the LIF neuron. Default: ``"torch"``
        :type backend: str

        :param tau: Membrane time constant of the ``LIFNode``. Default: 2.0
        :type tau: float

        :param detach_reset: Whether to detach the computational graph on reset. Default: ``True``
        :type detach_reset: bool
        """
        super().__init__()
        self.fc1 = layer.SeqToANNContainer(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
        )
        self.neuron1 = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )
        self.fc2 = layer.SeqToANNContainer(
            nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_features),
        )
        self.neuron2 = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    def forward(self, x_seq: torch.Tensor):
        x_seq = self.neuron1(self.fc1(x_seq))
        x_seq = self.neuron2(self.fc2(x_seq))
        return x_seq


class SpikformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        r"""
        **API Language** - :ref:`中文 <SpikformerBlock-cn>` | :ref:`English <SpikformerBlock-en>`

        ----

        .. _SpikformerBlock-cn:

        * **中文**

        Spikformer 基础块，包含一个 ``SpikingSelfAttention`` 和一个 ``SpikformerMLP``，并使用残差连接。输入必须是 5D 张量 ``[T, N, C, H, W]``。

        :param dim: 特征维度
        :type dim: int

        :param num_heads: 自注意力头数
        :type num_heads: int

        :param mlp_ratio: MLP 隐藏层维度相对于 ``dim`` 的倍数。默认为 4.0
        :type mlp_ratio: float

        :param backend: 神经元后端。默认为 ``"torch"``
        :type backend: str

        :param tau: ``LIFNode`` 的膜电位时间常数。默认为 2.0
        :type tau: float

        :param detach_reset: 是否在重置时断开计算图。默认为 ``True``
        :type detach_reset: bool

        :raises ValueError: 如果输入不是 5D 张量 ``[T, N, C, H, W]``

        ----

        .. _SpikformerBlock-en:

        * **English**

        A Spikformer transformer block consisting of a ``SpikingSelfAttention`` layer and a ``SpikformerMLP`` with residual connections. The input must be a 5D tensor ``[T, N, C, H, W]``.

        :param dim: Feature dimension
        :type dim: int

        :param num_heads: Number of attention heads
        :type num_heads: int

        :param mlp_ratio: Ratio of MLP hidden dimension to ``dim``. Default: 4.0
        :type mlp_ratio: float

        :param backend: Backend for the LIF neuron. Default: ``"torch"``
        :type backend: str

        :param tau: Membrane time constant of the ``LIFNode``. Default: 2.0
        :type tau: float

        :param detach_reset: Whether to detach the computational graph on reset. Default: ``True``
        :type detach_reset: bool

        :raises ValueError: If the input is not a 5D tensor ``[T, N, C, H, W]``
        """
        super().__init__()
        self.attn = SpikingSelfAttention(dim=dim, num_heads=num_heads, backend=backend)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SpikformerMLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )

    def forward(self, x_seq: torch.Tensor):
        if x_seq.ndim != 5:
            raise ValueError(
                f"expected 5D input with shape [T, N, C, H, W], but got {x_seq.shape}"
            )
        T, N, C, H, W = x_seq.shape
        x_tokens = x_seq.flatten(3)
        x_tokens = x_tokens + self.attn(x_tokens)
        x_tokens = x_tokens + self.mlp(x_tokens)
        return x_tokens.reshape(T, N, C, H, W).contiguous()


class Spikformer(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dims: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depths: int = 4,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        r"""
        **API Language** - :ref:`中文 <Spikformer-cn>` | :ref:`English <Spikformer-en>`

        ----

        .. _Spikformer-cn:

        * **中文**

        Spikformer 脉冲视觉 Transformer 模型，用于图像分类。输入图像首先通过 ``SpikformerPatchStem`` 进行分块嵌入，
        然后经过多个 ``SpikformerBlock`` 处理，最后通过线性分类头输出类别预测。支持多步 (multi-step) 时序处理。

        :param T: 时间步数。默认为 4
        :type T: int

        :param in_channels: 输入图像的通道数。默认为 3
        :type in_channels: int

        :param img_size_h: 输入图像高度。默认为 224
        :type img_size_h: int

        :param img_size_w: 输入图像宽度。默认为 224
        :type img_size_w: int

        :param patch_size: 分块大小。默认为 16
        :type patch_size: int

        :param num_classes: 分类类别数。默认为 1000
        :type num_classes: int

        :param embed_dims: 嵌入维度。默认为 256
        :type embed_dims: int

        :param num_heads: 自注意力头数。默认为 8
        :type num_heads: int

        :param mlp_ratio: MLP 隐藏层维度相对于 ``embed_dims`` 的倍数。默认为 4.0
        :type mlp_ratio: float

        :param depths: Transformer 块的数量。默认为 4
        :type depths: int

        :param backend: 神经元后端。默认为 ``"torch"``
        :type backend: str

        :param tau: ``LIFNode`` 的膜电位时间常数。默认为 2.0
        :type tau: float

        :param detach_reset: 是否在重置时断开计算图。默认为 ``True``
        :type detach_reset: bool

        ----

        .. _Spikformer-en:

        * **English**

        Spikformer spiking vision Transformer for image classification. Input images are first patch-embedded by ``SpikformerPatchStem``,
        then processed by multiple ``SpikformerBlock`` modules, and finally classified by a linear head. Supports multi-step temporal processing.

        :param T: Number of time steps. Default: 4
        :type T: int

        :param in_channels: Number of channels in the input image. Default: 3
        :type in_channels: int

        :param img_size_h: Input image height. Default: 224
        :type img_size_h: int

        :param img_size_w: Input image width. Default: 224
        :type img_size_w: int

        :param patch_size: Patch size. Default: 16
        :type patch_size: int

        :param num_classes: Number of classes. Default: 1000
        :type num_classes: int

        :param embed_dims: Embedding dimension. Default: 256
        :type embed_dims: int

        :param num_heads: Number of attention heads. Default: 8
        :type num_heads: int

        :param mlp_ratio: Ratio of MLP hidden dimension to ``embed_dims``. Default: 4.0
        :type mlp_ratio: float

        :param depths: Number of Transformer blocks. Default: 4
        :type depths: int

        :param backend: Backend for the LIF neuron. Default: ``"torch"``
        :type backend: str

        :param tau: Membrane time constant of the ``LIFNode``. Default: 2.0
        :type tau: float

        :param detach_reset: Whether to detach the computational graph on reset. Default: ``True``
        :type detach_reset: bool
        """
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.depths = depths
        self.patch_embed = SpikformerPatchStem(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
                for _ in range(depths)
            ]
        )
        self.head = layer.Linear(embed_dims, num_classes, step_mode="m")
        self._init_weights()
        functional.set_step_mode(self, "m")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _to_sequence(self, x: torch.Tensor):
        if x.ndim == 4:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if x.ndim == 5:
            return x
        raise ValueError(
            f"expected 4D image input [N, C, H, W] or 5D sequence input [T, N, C, H, W], but got {x.shape}"
        )

    def forward_features(self, x_seq: torch.Tensor):
        x_seq = self.patch_embed(x_seq)
        for block in self.blocks:
            x_seq = block(x_seq)
        return x_seq.flatten(3).mean(dim=-1)

    def forward(self, x: torch.Tensor):
        x_seq = self._to_sequence(x)
        x_seq = self.forward_features(x_seq)
        return self.head(x_seq)


def spikformer_ti(
    T: int = 4,
    in_channels: int = 3,
    img_size_h: int = 224,
    img_size_w: int = 224,
    num_classes: int = 1000,
    backend: str = "torch",
) -> Spikformer:
    r"""
    **API Language** - :ref:`中文 <spikformer_ti-cn>` | :ref:`English <spikformer_ti-en>`

    ----

    .. _spikformer_ti-cn:

    * **中文**

    返回一个 Spikformer-Ti (tiny) 模型，其 ``embed_dims=256, num_heads=8, depths=4``。

    :param T: 时间步数。默认为 4
    :type T: int
    :param in_channels: 输入图像的通道数。默认为 3
    :type in_channels: int
    :param img_size_h: 输入图像高度。默认为 224
    :type img_size_h: int
    :param img_size_w: 输入图像宽度。默认为 224
    :type img_size_w: int
    :param num_classes: 分类类别数。默认为 1000
    :type num_classes: int
    :param backend: 神经元后端。默认为 ``\"torch\"``
    :type backend: str
    :return: 模型实例
    :rtype: Spikformer

    ----

    .. _spikformer_ti-en:

    * **English**

    Return a Spikformer-Ti (tiny) model with ``embed_dims=256, num_heads=8, depths=4``.

    :param T: Number of time steps. Default: 4
    :type T: int
    :param in_channels: Number of input channels. Default: 3
    :type in_channels: int
    :param img_size_h: Input image height. Default: 224
    :type img_size_h: int
    :param img_size_w: Input image width. Default: 224
    :type img_size_w: int
    :param num_classes: Number of classes. Default: 1000
    :type num_classes: int
    :param backend: Backend for neurons. Default: ``\"torch\"``
    :type backend: str
    :return: Model instance
    :rtype: Spikformer
    """
    return Spikformer(
        T=T,
        in_channels=in_channels,
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        num_classes=num_classes,
        embed_dims=256,
        num_heads=8,
        mlp_ratio=4.0,
        depths=4,
        backend=backend,
    )


def spikformer_s(
    T: int = 4,
    in_channels: int = 3,
    img_size_h: int = 224,
    img_size_w: int = 224,
    num_classes: int = 1000,
    backend: str = "torch",
) -> Spikformer:
    r"""
    **API Language** - :ref:`中文 <spikformer_s-cn>` | :ref:`English <spikformer_s-en>`

    ----

    .. _spikformer_s-cn:

    * **中文**

    返回一个 Spikformer-S (small) 模型，其 ``embed_dims=384, num_heads=12, depths=6``。

    :param T: 时间步数。默认为 4
    :type T: int
    :param in_channels: 输入图像的通道数。默认为 3
    :type in_channels: int
    :param img_size_h: 输入图像高度。默认为 224
    :type img_size_h: int
    :param img_size_w: 输入图像宽度。默认为 224
    :type img_size_w: int
    :param num_classes: 分类类别数。默认为 1000
    :type num_classes: int
    :param backend: 神经元后端。默认为 ``\"torch\"``
    :type backend: str
    :return: 模型实例
    :rtype: Spikformer

    ----

    .. _spikformer_s-en:

    * **English**

    Return a Spikformer-S (small) model with ``embed_dims=384, num_heads=12, depths=6``.

    :param T: Number of time steps. Default: 4
    :type T: int
    :param in_channels: Number of input channels. Default: 3
    :type in_channels: int
    :param img_size_h: Input image height. Default: 224
    :type img_size_h: int
    :param img_size_w: Input image width. Default: 224
    :type img_size_w: int
    :param num_classes: Number of classes. Default: 1000
    :type num_classes: int
    :param backend: Backend for neurons. Default: ``\"torch\"``
    :type backend: str
    :return: Model instance
    :rtype: Spikformer
    """
    return Spikformer(
        T=T,
        in_channels=in_channels,
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        num_classes=num_classes,
        embed_dims=384,
        num_heads=12,
        mlp_ratio=4.0,
        depths=6,
        backend=backend,
    )


def spikformer_cifar10(
    T: int = 4,
    num_classes: int = 10,
    backend: str = "torch",
) -> Spikformer:
    r"""Build the Spikformer configuration used for CIFAR-10.

    **API Language** - 中文 | English

    **中文：** 构建官方 CIFAR-10 结构：32×32 输入、4×4 patch、384 维、
    12 个 attention heads 和 4 个 Transformer blocks。

    **English:** Build the official CIFAR-10 architecture with 32×32 input,
    4×4 patches, 384 channels, 12 attention heads, and 4 Transformer blocks.

    :param T: SNN 时间步。 / SNN time steps.
    :type T: int
    :param num_classes: 分类类别数。 / Number of classes.
    :type num_classes: int
    :param backend: 神经元 backend。 / Neuron backend.
    :type backend: str
    :return: CIFAR-10 Spikformer。 / CIFAR-10 Spikformer.
    :rtype: Spikformer
    """
    return Spikformer(
        T=T,
        in_channels=3,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dims=384,
        num_heads=12,
        mlp_ratio=4.0,
        depths=4,
        backend=backend,
    )


class _SpikformerHead(nn.Module):
    def __init__(self, block: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x).flatten(3).mean(dim=-1))


def _pipeline_stage(model: nn.Module, rank: int, size: int) -> nn.Module:
    if size == 1:
        return model
    if size not in {2, 4}:
        raise ValueError("Spikformer PP size must be 1, 2, or 4.")
    if len(model.blocks) == 6:
        chunks = (
            model.patch_embed,
            nn.Sequential(model.blocks[0], model.blocks[1]),
            nn.Sequential(model.blocks[2], model.blocks[3]),
            _SpikformerHead(
                nn.Sequential(model.blocks[4], model.blocks[5]), model.head
            ),
        )
    elif len(model.blocks) == 4:
        if size == 4:
            raise ValueError("The 4-block Spikformer supports PP size 1 or 2.")
        chunks = (
            model.patch_embed,
            nn.Sequential(model.blocks[0], model.blocks[1]),
            nn.Sequential(model.blocks[2], model.blocks[3]),
            _SpikformerHead(nn.Identity(), model.head),
        )
    else:
        raise ValueError("Spikformer PP requires 4 or 6 transformer blocks.")
    chunks_per_stage = len(chunks) // size
    start = rank * chunks_per_stage
    return nn.Sequential(*chunks[start : start + chunks_per_stage])


def _copy_batch_norm(
    source: nn.modules.batchnorm._BatchNorm,
    indices: Optional[torch.Tensor] = None,
) -> nn.Module:
    if not isinstance(source, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise TypeError(f"Unsupported batch norm type {type(source).__name__}.")
    size = source.num_features if indices is None else indices.numel()
    batch_norm = (
        layer.BatchNorm1d if isinstance(source, nn.BatchNorm1d) else layer.BatchNorm2d
    )
    target = batch_norm(
        size,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    )

    def select(value: torch.Tensor) -> torch.Tensor:
        return value if indices is None else value[indices]

    with torch.no_grad():
        if source.affine:
            target.weight.copy_(select(source.weight))
            target.bias.copy_(select(source.bias))
            target.weight.requires_grad_(source.weight.requires_grad)
            target.bias.requires_grad_(source.bias.requires_grad)
        if source.track_running_stats:
            target.running_mean.copy_(select(source.running_mean))
            target.running_var.copy_(select(source.running_var))
            target.num_batches_tracked.copy_(source.num_batches_tracked)
    target.train(source.training)
    return target


def _convert_batch_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)) and not isinstance(
            child, (layer.BatchNorm1d, layer.BatchNorm2d)
        ):
            setattr(module, name, _copy_batch_norm(child))
        else:
            _convert_batch_norms(child)


class _HeadShardQKVConv1d(nn.Conv1d):
    def __init__(
        self,
        source: nn.Conv1d,
        process_group: ProcessGroup,
        num_heads: int,
    ) -> None:
        if source.kernel_size != (1,) or source.groups != 1:
            raise ValueError("Spikformer QKV TP requires an ungrouped 1x1 Conv1d.")
        if source.out_channels != 3 * source.in_channels:
            raise ValueError("Spikformer QKV projection must have 3 * dim outputs.")

        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if num_heads % world_size:
            raise ValueError("Spikformer num_heads must be divisible by TP size.")
        dim = source.in_channels
        if dim % num_heads:
            raise ValueError("Spikformer channels must be divisible by num_heads.")
        heads_per_rank = num_heads // world_size
        head_dim = dim // num_heads
        start = rank * heads_per_rank * head_dim
        end = start + heads_per_rank * head_dim
        self.output_indices = torch.cat(
            [torch.arange(offset + start, offset + end) for offset in (0, dim, 2 * dim)]
        )
        local_dim = end - start
        super().__init__(
            dim,
            3 * local_dim,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight[self.output_indices])
            if source.bias is not None:
                self.bias.copy_(source.bias[self.output_indices])
        self.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None:
            self.bias.requires_grad_(source.bias.requires_grad)
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ColwiseBackwardAllReduce.apply(x, self.process_group)
        return super().forward(x)


@dataclass(frozen=True)
class SpikformerConfig(ModelConfig):
    builder: ClassVar[str] = (
        "spikingjelly.activation_based.model.spikformer.SpikformerBuilder"
    )
    image_height: int = 224
    image_width: int = 224
    in_channels: int = 3
    neuron_backend: str = "torch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_height <= 0 or self.image_width <= 0 or self.in_channels <= 0:
            raise ValueError("image dimensions and in_channels must be positive.")
        if self.step_mode != "m":
            raise ValueError("The built-in Spikformer requires step_mode='m'.")


SpikformerConfig.__init__.__doc__ = r"""Configure Spikformer-S distributed execution.

**API Language** - 中文 | English

**中文：** 声明 Spikformer-S 的时间步、图像尺寸、输入通道、类别数和神经元后端。
模型 recipe 按 attention head 分片 TP，并重新拼接本地 Q、K、V heads。

**English:** Declare time steps, image dimensions, input channels, classes, and
the neuron backend for Spikformer-S. The model recipe shards TP by attention
head and reconstructs local Q, K, and V heads.

:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param num_classes: 分类类别数。 / Number of classes.
:type num_classes: int
:param step_mode: 固定为 ``"m"``。 / Must be ``"m"``.
:type step_mode: str
:param image_height: 输入图像高度。 / Input image height.
:type image_height: int
:param image_width: 输入图像宽度。 / Input image width.
:type image_width: int
:param in_channels: 输入通道数。 / Input channels.
:type in_channels: int
:param neuron_backend: 神经元 backend。 / Neuron backend.
:type neuron_backend: str
:raises ValueError: 图像尺寸、通道或 step mode 无效。 / If image dimensions,
    channels, or the step mode are invalid.
"""


@dataclass(frozen=True)
class SpikformerCIFAR10Config(ModelConfig):
    builder: ClassVar[str] = (
        "spikingjelly.activation_based.model.spikformer.SpikformerBuilder"
    )
    num_classes: int = 10
    neuron_backend: str = "torch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.step_mode != "m":
            raise ValueError("The built-in Spikformer requires step_mode='m'.")


SpikformerCIFAR10Config.__init__.__doc__ = r"""Configure CIFAR-10 Spikformer.

**API Language** - 中文 | English

**中文：** 声明固定 32×32 输入、4×4 patch、384 维、12 heads、4 blocks 的
CIFAR-10 Spikformer。TP 按 attention head 分片；PP 支持 1 或 2 stages。

**English:** Declare the CIFAR-10 Spikformer with fixed 32×32 input, 4×4 patches,
384 channels, 12 heads, and 4 blocks. TP shards attention heads; PP supports one
or two stages.

:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param num_classes: 分类类别数。 / Number of classes.
:type num_classes: int
:param step_mode: 固定为 ``"m"``。 / Must be ``"m"``.
:type step_mode: str
:param neuron_backend: 神经元 backend。 / Neuron backend.
:type neuron_backend: str
:raises ValueError: 时间步、类别数或 step mode 无效。 / If time steps, class count,
    or the step mode are invalid.
"""


class SpikformerBuilder(ModelBuilder):
    @staticmethod
    def _memopt_split(module: nn.Module) -> tuple[nn.Module, ...]:
        if isinstance(module, SpikformerBlock):
            return module.attn, module.mlp
        if isinstance(module, SpikingSelfAttention):
            return (
                module.qkv_conv_bn,
                module.qkv_lif,
                module.attn_lif,
                module.proj_conv_bn,
                module.proj_lif,
            )
        if isinstance(module, SpikformerMLP):
            return module.fc1, module.neuron1, module.fc2, module.neuron2
        if isinstance(module, layer.SeqToANNContainer):
            return tuple(module.children())
        return ()

    @staticmethod
    def _memopt_can_chunk(module: nn.Module) -> bool:
        return isinstance(module, (nn.Conv1d, nn.Conv2d, neuron.BaseNode))

    def _build_canonical_model(self) -> nn.Module:
        config = self.config
        if isinstance(config, SpikformerCIFAR10Config):
            model = spikformer_cifar10(
                T=config.time_steps,
                num_classes=config.num_classes,
                backend=config.neuron_backend,
            )
        elif isinstance(config, SpikformerConfig):
            model = spikformer_s(
                T=config.time_steps,
                in_channels=config.in_channels,
                img_size_h=config.image_height,
                img_size_w=config.image_width,
                num_classes=config.num_classes,
                backend=config.neuron_backend,
            )
        else:
            raise TypeError("SpikformerBuilder requires a Spikformer config.")
        _convert_batch_norms(model)
        return model

    def _pipeline_stage(self, model: nn.Module, rank: int, size: int) -> nn.Module:
        return _pipeline_stage(model, rank, size)

    @staticmethod
    def _qkv_indices(total: int, rank: int, size: int) -> torch.Tensor:
        dim = total // 3
        local_dim = dim // size
        start = rank * local_dim
        end = start + local_dim
        return torch.cat(
            [torch.arange(offset + start, offset + end) for offset in (0, dim, 2 * dim)]
        )

    def _merge_tensor_parallel_shards(
        self,
        name: str,
        shards: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            "attn.qkv_conv_bn." not in name
            or reference.ndim == 0
            or shards[0].shape == reference.shape
        ):
            return super()._merge_tensor_parallel_shards(name, shards, reference)
        result = torch.empty_like(reference)
        for rank, shard in enumerate(shards):
            result[self._qkv_indices(reference.shape[0], rank, len(shards))] = shard
        return result

    def _shard_tensor_parallel_tensor(
        self,
        name: str,
        value: torch.Tensor,
        target: torch.Tensor,
        tensor_rank: int,
        tensor_size: int,
    ) -> torch.Tensor:
        if (
            "attn.qkv_conv_bn." in name
            and value.ndim > 0
            and value.shape != target.shape
        ):
            return value[
                self._qkv_indices(value.shape[0], tensor_rank, tensor_size)
            ].contiguous()
        return super()._shard_tensor_parallel_tensor(
            name, value, target, tensor_rank, tensor_size
        )

    @staticmethod
    def _parallelize_stem(model: nn.Module, process_group: ProcessGroup) -> None:
        for index, stage in enumerate(model.patch_embed.stages):
            container = stage.conv_bn.block
            conv = container[0]
            batch_norm = container[1]
            if index % 2 == 0:
                container[0] = ChannelShardConv2d(conv, process_group, "colwise")
                container[1] = ChannelShardBatchNorm2d(batch_norm, process_group)
            else:
                container[0] = ChannelShardConv2d(conv, process_group, "rowwise")
                container[1] = _copy_batch_norm(batch_norm)
        positional = model.patch_embed.positional_encoding.conv_bn.block
        positional[1] = _copy_batch_norm(positional[1])

    @staticmethod
    def _parallelize_block(block: SpikformerBlock, process_group: ProcessGroup) -> None:
        attention = block.attn
        qkv_source = attention.qkv_conv_bn[0]
        qkv = _HeadShardQKVConv1d(
            qkv_source,
            process_group,
            attention.num_heads,
        )
        attention.qkv_conv_bn[0] = qkv
        attention.qkv_conv_bn[1] = _copy_batch_norm(
            attention.qkv_conv_bn[1], qkv.output_indices
        )
        attention.num_heads //= dist.get_world_size(process_group)
        attention.proj_conv_bn[0] = ChannelShardConv1d(
            attention.proj_conv_bn[0], process_group, "rowwise"
        )
        attention.proj_conv_bn[1] = _copy_batch_norm(attention.proj_conv_bn[1])

        mlp = block.mlp
        fc1 = mlp.fc1[0]
        mlp.fc1[0] = ChannelShardConv1d(fc1, process_group, "colwise")
        mlp.fc1[1] = ChannelShardBatchNorm1d(mlp.fc1[1], process_group)
        mlp.fc2[0] = ChannelShardConv1d(mlp.fc2[0], process_group, "rowwise")
        mlp.fc2[1] = _copy_batch_norm(mlp.fc2[1])

    def build(
        self,
        *,
        process_group: Optional[ProcessGroup],
        memopt_process_group: Optional[ProcessGroup],
        pipeline_rank: int,
        pipeline_size: int,
        pipeline_microbatches: int,
        device: torch.device,
        micro_batch_size: int,
        memopt_level: int,
        memopt_compress_inputs: bool,
        memopt_checkpoint_budget: str,
    ) -> tuple[
        nn.Module,
        tuple[str, ...],
        Optional[tuple[int, ...]],
        Optional[tuple[int, ...]],
    ]:
        config = self.config
        if isinstance(config, SpikformerCIFAR10Config):
            image_height = image_width = 32
            in_channels = 3
            patch_size = 4
        elif isinstance(config, SpikformerConfig):
            image_height = config.image_height
            image_width = config.image_width
            in_channels = config.in_channels
            patch_size = 16
        else:
            raise TypeError("SpikformerBuilder requires a Spikformer config.")
        model = self._build_canonical_model()
        if pipeline_size > 1 and (
            image_height % patch_size or image_width % patch_size
        ):
            raise ValueError(
                f"Spikformer image dimensions must be divisible by {patch_size} with PP."
            )
        world_size = dist.get_world_size(process_group) if process_group else 1
        if world_size > 1:
            self._parallelize_stem(model, process_group)
            for block in model.blocks:
                self._parallelize_block(block, process_group)

        model = self._pipeline_stage(model, pipeline_rank, pipeline_size)
        fsdp_roots = tuple(
            name
            for name, module in model.named_modules()
            if isinstance(module, (SpikformerPatchStem, SpikformerBlock))
        )
        model.to(device)
        if memopt_level:
            batch = (
                micro_batch_size
                if pipeline_size == 1
                else micro_batch_size // pipeline_microbatches
            )
            if pipeline_size == 1:
                shape = (batch, in_channels, image_height, image_width)
            elif pipeline_rank == 0:
                shape = (
                    config.time_steps,
                    batch,
                    in_channels,
                    image_height,
                    image_width,
                )
            else:
                shape = (
                    config.time_steps,
                    batch,
                    384,
                    image_height // patch_size,
                    image_width // patch_size,
                )
            dummy = torch.zeros(shape, device=device)
            memopt.optimize_memory(
                model,
                SpikformerBlock,
                lambda current: current(dummy),
                level=memopt_level,
                checkpoint_budget=memopt_checkpoint_budget,
                compress=False,
                split_fn=self._memopt_split,
                can_chunk=self._memopt_can_chunk,
                process_group=memopt_process_group,
            )
        if pipeline_size == 1:
            return model, fsdp_roots, None, None
        micro_batch = micro_batch_size // pipeline_microbatches
        activation_shape = (
            config.time_steps,
            micro_batch,
            384,
            image_height // patch_size,
            image_width // patch_size,
        )
        stage_shapes = (
            activation_shape,
            activation_shape,
            activation_shape,
            (config.time_steps, micro_batch, config.num_classes),
        )
        chunks_per_stage = len(stage_shapes) // pipeline_size
        start = pipeline_rank * chunks_per_stage
        input_shape = (
            (
                micro_batch,
                config.time_steps,
                in_channels,
                image_height,
                image_width,
            )
            if start == 0
            else stage_shapes[start - 1]
        )
        return (
            model,
            fsdp_roots,
            input_shape,
            stage_shapes[start + chunks_per_stage - 1],
        )
