from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .. import functional, layer, neuron
from .._neuron import _make_multi_step_neuron

__all__ = ["MaxFormer", "maxformer_10_384", "maxformer_cifar"]


class _MaxCifarEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, spiking_neuron, neuron_kwargs):
        super().__init__()
        hidden = out_channels // 2
        self.conv = layer.Conv2d(
            in_channels, hidden, 3, padding=1, bias=False, step_mode="m"
        )
        self.bn = layer.BatchNorm2d(hidden, step_mode="m")
        self.branch = nn.Sequential(
            _make_multi_step_neuron(spiking_neuron, neuron_kwargs),
            layer.Conv2d(hidden, out_channels, 3, padding=1, bias=False, step_mode="m"),
            layer.BatchNorm2d(out_channels, step_mode="m"),
        )
        self.shortcut = nn.Sequential(
            _make_multi_step_neuron(spiking_neuron, neuron_kwargs),
            layer.Conv2d(hidden, out_channels, 1, bias=False, step_mode="m"),
            layer.BatchNorm2d(out_channels, step_mode="m"),
        )

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.branch(x) + self.shortcut(x)


class _MaxCifarSSA(nn.Module):
    def __init__(self, dim, num_heads, spiking_neuron, neuron_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.input_lif = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.qkv = layer.SeqToANNContainer(
            nn.Conv1d(dim, 3 * dim, 1, bias=False), nn.BatchNorm1d(3 * dim)
        )
        self.qkv_lif = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.attn_lif = _make_multi_step_neuron(
            spiking_neuron,
            neuron_kwargs,
            {"tau": 2.0, "detach_reset": True, "v_threshold": 0.5},
        )
        self.proj = layer.SeqToANNContainer(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))

    def forward(self, x):
        T, N, C, L = x.shape
        q, k, v = self.qkv_lif(self.qkv(self.input_lif(x))).chunk(3, dim=2)
        q, k, v = [
            z.transpose(-1, -2)
            .reshape(T, N, L, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            for z in (q, k, v)
        ]
        x = (q @ (k.transpose(-2, -1) @ v)) * 0.125
        x = x.transpose(3, 4).reshape(T, N, C, L)
        return self.proj(self.attn_lif(x))


class _MaxEmbedInit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
        neuron_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.conv1 = layer.Conv2d(
            in_channels, hidden, 3, stride=2, padding=1, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(hidden, step_mode="m")
        self.lif1 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.conv2 = layer.Conv2d(
            hidden, out_channels, 3, stride=2, padding=1, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.conv3 = layer.Conv2d(
            out_channels, out_channels, 3, padding=1, step_mode="m"
        )
        self.bn3 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.shortcut = nn.Sequential(
            layer.Conv2d(hidden, out_channels, 1, stride=2, step_mode="m"),
            layer.BatchNorm2d(out_channels, step_mode="m"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.lif1(x)
        identity = x
        x = self.bn2(self.conv2(x))
        x = self.lif2(x)
        x = self.bn3(self.conv3(x))
        return x + self.shortcut(identity)


class _MaxEmbedStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
        neuron_kwargs: Dict[str, Any],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = layer.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=bias, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif1 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.pool = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.conv2 = layer.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=bias, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.shortcut = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, 1, bias=bias, step_mode="m"),
            layer.BatchNorm2d(out_channels, step_mode="m"),
            layer.MaxPool2d(3, stride=2, padding=1, step_mode="m"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The paper's Embed_Max feeds the same spike tensor to both branches.
        x = self.lif1(x)
        identity = self.shortcut(x)
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.lif2(x)
        x = self.bn2(self.conv2(x))
        return x + identity


class _SpatialMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
        neuron_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.residual = dim == hidden_dim
        self.fc1 = layer.Conv2d(dim, hidden_dim, 1, step_mode="m")
        self.bn1 = layer.BatchNorm2d(hidden_dim, step_mode="m")
        self.lif1 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.fc2 = layer.Conv2d(hidden_dim, dim, 1, step_mode="m")
        self.bn2 = layer.BatchNorm2d(dim, step_mode="m")
        self.lif2 = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.bn1(self.fc1(self.lif1(x)))
        if self.residual:
            x = x + identity
            identity = x
        x = self.bn2(self.fc2(self.lif2(x)))
        return x + identity


class _DWCBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        mlp_ratio: float,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
        neuron_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.lif = _make_multi_step_neuron(spiking_neuron, neuron_kwargs)
        self.dwconv = layer.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            step_mode="m",
        )
        self.bn = layer.BatchNorm2d(dim, step_mode="m")
        self.mlp = _SpatialMLP(dim, int(dim * mlp_ratio), spiking_neuron, neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bn(self.dwconv(self.lif(x)))
        return self.mlp(x)


class _SSABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
        neuron_kwargs: Dict[str, Any],
        cifar: bool = False,
    ) -> None:
        super().__init__()
        self.attn = (
            _MaxCifarSSA(dim, num_heads, spiking_neuron, neuron_kwargs)
            if cifar
            else layer.SpikingSelfAttention(
                dim, num_heads, spiking_neuron=spiking_neuron, **neuron_kwargs
            )
        )
        self.mlp = _SpatialMLP(dim, int(dim * mlp_ratio), spiking_neuron, neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.flatten(3)
        x = x + self.attn(x)
        return self.mlp(x.reshape(shape))


class MaxFormer(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 384,
        depths: tuple[int, int, int] = (1, 2, 7),
        mlp_ratio: float = 4.0,
        spiking_neuron: Optional[Callable[..., neuron.BaseNode]] = None,
        cifar: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxFormer.__init__-cn>` | :ref:`English <MaxFormer.__init__-en>`

        ----

        .. _MaxFormer.__init__-cn:

        * **中文**

        Max-Former 层级脉冲 Transformer。模型使用额外的 max-pool 和前两级
        depth-wise convolution 保留高频特征，最后一级使用 Spiking Self-Attention。
        ``cifar=True`` 时改用不下采样的入口及 Identity/DWC3/SSA 三阶段。
        ``forward`` 接收与模型参数同设备、同浮点类型的 ``[N, C, H, W]`` 图像或
        ``[T, N, C, H, W]`` 序列，并返回 ``[N, num_classes]`` 分类结果。静态图像
        会沿时间维重复 ``T`` 次。
        处理相互独立的输入序列时，应调用 :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>` 重置网络状态。

        :param T: 静态图像输入的仿真时间步数
        :type T: int
        :param in_channels: 输入通道数
        :type in_channels: int
        :param num_classes: 分类类别数
        :type num_classes: int
        :param embed_dims: 最后一级的通道数；CIFAR 结构须能被 ``8`` 整除，其他结构能被 ``4`` 整除
        :type embed_dims: int
        :param depths: 三个 stage 的 block 数
        :type depths: tuple[int, int, int]
        :param mlp_ratio: MLP 隐藏通道数相对输入通道数的倍率
        :type mlp_ratio: float
        :param cifar: 是否使用官方 CIFAR 结构：不下采样的入口、Identity/DWC3/SSA
            三阶段及八头 SSA。应配合 ``depths=(1, 1, 2)``、``embed_dims=384``
        :type cifar: bool
        :param spiking_neuron: 自定义神经元类；``None`` 保留论文默认值
        :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
        :param kwargs: 传给所有内部神经元的参数；指定自定义类时统一决定其构造
        :type kwargs: Any
        :raises ValueError: ``depths`` 不含三个值，或 ``embed_dims`` 不满足整除条件

        ----

        .. _MaxFormer.__init__-en:

        * **English**

        Hierarchical Max-Former. Extra max-pooling and depth-wise convolutions in
        the first two stages preserve high-frequency features, followed by Spiking
        Self-Attention in the final stage. With ``cifar=True``, the stem keeps
        spatial resolution and stages use Identity/DWC3/SSA. ``forward`` accepts a floating-point
        image ``[N, C, H, W]`` or sequence ``[T, N, C, H, W]`` on the same device
        and with the same dtype as the model parameters, and returns logits shaped
        ``[N, num_classes]``. A static image is repeated for ``T`` time steps.
        Call :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>`
        between independent input sequences.

        :param T: number of simulation steps used for static images
        :type T: int
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of classes
        :type num_classes: int
        :param embed_dims: final-stage channels; divisible by ``8`` for CIFAR,
            or by ``4`` otherwise
        :type embed_dims: int
        :param depths: block counts in the three stages
        :type depths: tuple[int, int, int]
        :param mlp_ratio: ratio of MLP hidden channels to input channels
        :type mlp_ratio: float
        :param cifar: use the official CIFAR structure: a non-downsampling stem,
            Identity/DWC3/SSA stages and eight-head SSA. Use with
            ``depths=(1, 1, 2)`` and ``embed_dims=384``
        :type cifar: bool
        :param spiking_neuron: custom neuron class; ``None`` keeps the paper defaults
        :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
        :param kwargs: arguments passed to all internal neurons; with a custom
            class, these arguments define its construction uniformly
        :type kwargs: Any
        :raises ValueError: if ``depths`` does not contain three values or
            ``embed_dims`` does not meet the divisibility requirement

        **参考文献 | Reference**

        `Spiking Neural Networks Need High Frequency Information
        <https://arxiv.org/abs/2505.18608>`_
        """
        super().__init__()
        if len(depths) != 3 or embed_dims % (8 if cifar else 4):
            raise ValueError(
                "depths must contain three values; embed_dims must be divisible "
                "by 8 for CIFAR or 4 otherwise"
            )
        dims = (embed_dims // 4, embed_dims // 2, embed_dims)
        self.T = T
        if cifar and spiking_neuron is None:
            kwargs = {"tau": 2.0, "detach_reset": True, **kwargs}
        embed_init = _MaxCifarEmbed if cifar else _MaxEmbedInit
        self.patch_embed1 = embed_init(in_channels, dims[0], spiking_neuron, kwargs)
        self.patch_embed2 = _MaxEmbedStage(
            dims[0], dims[1], spiking_neuron, kwargs, bias=not cifar
        )
        self.patch_embed3 = _MaxEmbedStage(
            dims[1], dims[2], spiking_neuron, kwargs, bias=not cifar
        )
        self.stage1 = nn.ModuleList(
            [
                _SpatialMLP(dims[0], int(dims[0] * mlp_ratio), spiking_neuron, kwargs)
                if cifar
                else _DWCBlock(dims[0], 7, mlp_ratio, spiking_neuron, kwargs)
                for _ in range(depths[0])
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                _DWCBlock(dims[1], 3 if cifar else 5, mlp_ratio, spiking_neuron, kwargs)
                for _ in range(depths[1])
            ]
        )
        self.stage3 = nn.ModuleList(
            [
                _SSABlock(
                    dims[2],
                    8 if cifar else max(1, dims[2] // 64),
                    mlp_ratio,
                    spiking_neuron,
                    kwargs,
                    cifar=cifar,
                )
                for _ in range(depths[2])
            ]
        )
        self.head_lif = _make_multi_step_neuron(spiking_neuron, kwargs)
        self.head = nn.Linear(embed_dims, num_classes)
        functional.set_step_mode(self, "m")
        if cifar:
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if x.ndim == 5:
            return x
        raise ValueError(
            f"expected 4D image or 5D sequence input, but got shape {x.shape}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <MaxFormer.forward-cn>` | :ref:`English <MaxFormer.forward-en>`

        ----

        .. _MaxFormer.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量

        ----

        .. _MaxFormer.forward-en:

        * **English**

        :param x: floating-point image ``[N, C, H, W]`` or sequence
            ``[T, N, C, H, W]``
        :type x: torch.Tensor
        :return: classification logits shaped ``[N, num_classes]``
        :rtype: torch.Tensor
        :raises ValueError: if the input is neither four- nor five-dimensional
        """
        x = self._to_sequence(x)
        for embed, stage in (
            (self.patch_embed1, self.stage1),
            (self.patch_embed2, self.stage2),
            (self.patch_embed3, self.stage3),
        ):
            x = embed(x)
            for block in stage:
                x = block(x)
        x = self.head_lif(x.flatten(3).mean(-1))
        return self.head(x.mean(0))


def maxformer_10_384(
    T: int = 4,
    in_channels: int = 3,
    num_classes: int = 1000,
    spiking_neuron: Optional[Callable[..., neuron.BaseNode]] = None,
    **kwargs: Any,
) -> MaxFormer:
    r"""
    **API Language** - :ref:`中文 <maxformer_10_384-cn>` | :ref:`English <maxformer_10_384-en>`

    ----

    .. _maxformer_10_384-cn:

    * **中文**

    构建 Max-Former-10-384。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param spiking_neuron: 自定义神经元类；``None`` 保留论文默认值
    :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
    :param kwargs: 传给内部神经元的参数
    :type kwargs: Any
    :return: Max-Former-10-384 模型
    :rtype: MaxFormer

    ----

    .. _maxformer_10_384-en:

    * **English**

    Builds Max-Former-10-384.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param spiking_neuron: custom neuron class; ``None`` keeps the paper default
    :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
    :param kwargs: arguments passed to the internal neurons
    :type kwargs: Any
    :return: Max-Former-10-384 model
    :rtype: MaxFormer
    """
    return MaxFormer(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=384,
        depths=(1, 2, 7),
        spiking_neuron=spiking_neuron,
        **kwargs,
    )


def maxformer_cifar(
    T: int = 4,
    in_channels: int = 3,
    num_classes: int = 100,
    spiking_neuron: Optional[Callable[..., neuron.BaseNode]] = None,
    **kwargs: Any,
) -> MaxFormer:
    r"""
    **API Language** - :ref:`中文 <maxformer_cifar-cn>` | :ref:`English <maxformer_cifar-en>`

    ----

    .. _maxformer_cifar-cn:

    * **中文**

    构建官方 CIFAR Max-Former-4-384（八头 SSA），默认用于 CIFAR-100。
    静态输入为 ``[N, 3, 32, 32]``；每个独立 batch 后须调用 ``reset_net``。

    :param T: 静态图像输入的仿真时间步数，默认 ``4``
    :type T: int
    :param in_channels: 输入通道数，默认 ``3``
    :type in_channels: int
    :param num_classes: 分类类别数，默认 ``100``
    :type num_classes: int
    :param spiking_neuron: 自定义神经元类；``None`` 保留论文默认值
    :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
    :param kwargs: 传给内部神经元的参数；默认 LIF 使用 tau=2、硬复位到零、
        Sigmoid 替代梯度和 detach_reset=True，SSA 放电阈值为0.5，其他位置为1.0
    :type kwargs: Any
    :return: Max-Former-4-384 CIFAR 模型
    :rtype: MaxFormer

    ----

    .. _maxformer_cifar-en:

    * **English**

    Builds the official CIFAR Max-Former-4-384 with eight-head SSA, defaulting
    to CIFAR-100. Static input is ``[N, 3, 32, 32]``; call ``reset_net`` between
    independent batches.

    :param T: number of simulation steps used for static images; default ``4``
    :type T: int
    :param in_channels: number of input channels; default ``3``
    :type in_channels: int
    :param num_classes: number of classes; default ``100``
    :type num_classes: int
    :param spiking_neuron: custom neuron class; ``None`` keeps the paper default
    :type spiking_neuron: Optional[Callable[..., neuron.BaseNode]]
    :param kwargs: internal neuron arguments; default LIF uses tau=2, hard reset
        to zero, Sigmoid surrogate and detach_reset=True. SSA firing threshold is
        0.5; other sites use 1.0
    :type kwargs: Any
    :return: Max-Former-4-384 CIFAR model
    :rtype: MaxFormer

    **参考文献 | Reference**

    `Official CIFAR source
    <https://github.com/bic-L/MaxFormer/tree/6ddd4ccde2cc9c4b965059403868e57803b9b6b0/cifar10-100>`_
    """
    return MaxFormer(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=384,
        depths=(1, 1, 2),
        cifar=True,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
