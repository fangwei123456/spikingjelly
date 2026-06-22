import torch
import torch.nn as nn

from .. import base, functional, layer, neuron
from ..layer.attention import SpikingSelfAttention

__all__ = [
    "Spikformer",
    "SpikformerBlock",
    "SpikformerConv2dBN",
    "SpikformerConv2dBNLIF",
    "SpikformerMLP",
    "SpikformerPatchStem",
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
        **API Language:**
        :ref:`中文 <SpikformerConv2dBN-cn>` | :ref:`English <SpikformerConv2dBN-en>`

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

    def __spatial_split__(self):
        return tuple(self.block.children())


class SpikformerConv2dBNLIF(nn.Module, base.MultiStepModule):
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
        **API Language:**
        :ref:`中文 <SpikformerConv2dBNLIF-cn>` | :ref:`English <SpikformerConv2dBNLIF-en>`

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

    def __spatial_split__(self):
        return self.conv_bn, self.neuron


class SpikformerPatchStem(nn.Module, base.MultiStepModule):
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
        **API Language:**
        :ref:`中文 <SpikformerPatchStem-cn>` | :ref:`English <SpikformerPatchStem-en>`

        ----

        .. _SpikformerPatchStem-cn:

        * **中文**

        图像分块嵌入 (patch embedding) 模块，由 4 个卷积降采样阶段和 1 个位置编码卷积组成。每个阶段包含 ``SpikformerConv2dBNLIF`` (Conv2d + BN + MaxPool + LIF)。

        :param img_size_h: 输入图像高度。默认为 224
        :type img_size_h: int

        :param img_size_w: 输入图像宽度。默认为 224
        :type img_size_w: int

        :param patch_size: 分块大小，当前固定为 16。传入其他值将抛出 ``ValueError``
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

        :raises ValueError: 当 ``patch_size`` 不是 16 时抛出

        ----

        .. _SpikformerPatchStem-en:

        * **English**

        Image patch embedding stem consisting of 4 convolutional downsampling stages and a positional encoding convolution. Each stage uses ``SpikformerConv2dBNLIF`` (Conv2d + BN + MaxPool + LIF).

        :param img_size_h: Input image height. Default: 224
        :type img_size_h: int

        :param img_size_w: Input image width. Default: 224
        :type img_size_w: int

        :param patch_size: Patch size, currently fixed to 16. Other values will raise a ``ValueError``
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

        :raises ValueError: If ``patch_size`` is not 16
        """
        super().__init__()
        if patch_size != 16:
            raise ValueError(
                "The current SpikformerPatchStem uses a fixed 4-stage /16 patch pipeline; "
                f"expected patch_size=16, but got {patch_size}."
            )
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.embed_dims = embed_dims

        stage_dims = [embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims]
        layers = []
        in_c = in_channels
        for out_c in stage_dims:
            layers.append(
                SpikformerConv2dBNLIF(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool=True,
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


class SpikformerMLP(nn.Module, base.MultiStepModule):
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
        **API Language:**
        :ref:`中文 <SpikformerMLP-cn>` | :ref:`English <SpikformerMLP-en>`

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

    def __spatial_split__(self):
        return self.fc1, self.neuron1, self.fc2, self.neuron2


class SpikformerBlock(nn.Module, base.MultiStepModule):
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
        **API Language:**
        :ref:`中文 <SpikformerBlock-cn>` | :ref:`English <SpikformerBlock-en>`

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


class Spikformer(nn.Module, base.MultiStepModule):
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
        **API Language:**
        :ref:`中文 <Spikformer-cn>` | :ref:`English <Spikformer-en>`

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
    **API Language:**
    :ref:`中文 <spikformer_ti-cn>` | :ref:`English <spikformer_ti-en>`

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
    **API Language:**
    :ref:`中文 <spikformer_s-cn>` | :ref:`English <spikformer_s-en>`

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
