import torch
import torch.nn as nn

from .. import functional, layer, neuron
from ..layer.attention import SpikeDrivenSelfAttention

__all__ = ["SpikeDrivenTransformer", "sdt_8_384"]


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        pooling_stat: str,
        backend: str,
    ):
        super().__init__()
        channels = (embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims)
        self.stages = nn.ModuleList()
        previous = in_channels
        for use_pool, channels_out in zip(pooling_stat[:3], channels[:3]):
            stage = [
                layer.Conv2d(previous, channels_out, 3, padding=1, step_mode="m"),
                layer.BatchNorm2d(channels_out, step_mode="m"),
                neuron.LIFNode(step_mode="m", backend=backend),
            ]
            if use_pool == "1":
                stage.append(layer.MaxPool2d(3, stride=2, padding=1, step_mode="m"))
            self.stages.append(nn.Sequential(*stage))
            previous = channels_out
        final_stage = [
            layer.Conv2d(previous, embed_dims, 3, padding=1, step_mode="m"),
            layer.BatchNorm2d(embed_dims, step_mode="m"),
        ]
        if pooling_stat[3] == "1":
            final_stage.append(layer.MaxPool2d(3, stride=2, padding=1, step_mode="m"))
        self.final_stage = nn.Sequential(*final_stage)
        self.final_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.rpe_conv = layer.Conv2d(
            embed_dims, embed_dims, 3, padding=1, bias=False, step_mode="m"
        )
        self.rpe_bn = layer.BatchNorm2d(embed_dims, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.final_stage(x)
        identity = x
        x = self.rpe_bn(self.rpe_conv(self.final_lif(x)))
        return x + identity


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, backend: str):
        super().__init__()
        self.residual = dim == hidden_dim
        self.fc1 = layer.Conv2d(dim, hidden_dim, 1, step_mode="m")
        self.bn1 = layer.BatchNorm2d(hidden_dim, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.fc2 = layer.Conv2d(hidden_dim, dim, 1, step_mode="m")
        self.bn2 = layer.BatchNorm2d(dim, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.bn1(self.fc1(self.lif1(x)))
        if self.residual:
            x = x + identity
            identity = x
        x = self.bn2(self.fc2(self.lif2(x)))
        return x + identity


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, backend: str):
        super().__init__()
        self.attn = SpikeDrivenSelfAttention(dim, num_heads, backend=backend)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x))


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 384,
        num_heads: int = 8,
        depths: int = 8,
        mlp_ratio: float = 4.0,
        pooling_stat: str = "1111",
        backend: str = "torch",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <SpikeDrivenTransformer.__init__-cn>` | :ref:`English <SpikeDrivenTransformer.__init__-en>`

        ----

        .. _SpikeDrivenTransformer.__init__-cn:

        * **中文**

        Spike-driven Transformer v1。模型使用四级脉冲 patch embedding 和
        Spike-driven Self-Attention，通过 ``K * V`` 的逐元素乘法、token 求和与
        ``Q * (K * V)`` 代替稠密 softmax attention。``forward`` 接收与模型参数
        同设备、同浮点类型的 ``[N, C, H, W]`` 图像或 ``[T, N, C, H, W]`` 序列，
        并返回 ``[N, num_classes]`` 分类结果。静态图像会沿时间维重复 ``T`` 次。
        处理相互独立的输入序列时，应调用 :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>` 重置网络状态。

        :param T: 静态图像输入的仿真时间步数
        :type T: int
        :param in_channels: 输入通道数
        :type in_channels: int
        :param num_classes: 分类类别数
        :type num_classes: int
        :param embed_dims: embedding 通道数，必须能被 ``num_heads`` 整除
        :type embed_dims: int
        :param num_heads: attention head 数
        :type num_heads: int
        :param depths: Transformer block 数
        :type depths: int
        :param mlp_ratio: MLP 隐藏通道数相对输入通道数的倍率
        :type mlp_ratio: float
        :param pooling_stat: 四级 patch embedding 的 pooling 开关，每位为
            ``"0"`` 或 ``"1"``
        :type pooling_stat: str
        :param backend: 内部脉冲神经元使用的后端
        :type backend: str
        :raises ValueError: ``pooling_stat`` 不是四位 0/1 字符串，或
            ``embed_dims`` 不能被 ``num_heads`` 整除

        ----

        .. _SpikeDrivenTransformer.__init__-en:

        * **English**

        Spike-driven Transformer v1. The model combines a four-stage spiking patch
        embedding with Spike-driven Self-Attention, replacing dense softmax
        attention with element-wise ``K * V``, token reduction, and
        ``Q * (K * V)``. ``forward`` accepts a floating-point image
        ``[N, C, H, W]`` or sequence ``[T, N, C, H, W]`` on the same device and
        with the same dtype as the model parameters, and returns logits shaped
        ``[N, num_classes]``. A static image is repeated for ``T`` time steps.
        Call :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>`
        between independent input sequences.

        :param T: number of simulation steps used for static images
        :type T: int
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of classes
        :type num_classes: int
        :param embed_dims: number of embedding channels; must be divisible by ``num_heads``
        :type embed_dims: int
        :param num_heads: number of attention heads
        :type num_heads: int
        :param depths: number of Transformer blocks
        :type depths: int
        :param mlp_ratio: ratio of MLP hidden channels to input channels
        :type mlp_ratio: float
        :param pooling_stat: pooling mask for the four patch-embedding stages;
            every character must be ``"0"`` or ``"1"``
        :type pooling_stat: str
        :param backend: backend used by the internal spiking neurons
        :type backend: str
        :raises ValueError: if ``pooling_stat`` is not a four-character binary
            mask or ``embed_dims`` is not divisible by ``num_heads``

        **参考文献 | Reference**

        `Spike-Driven Transformer
        <https://arxiv.org/abs/2304.11954>`_
        """
        super().__init__()
        if len(pooling_stat) != 4 or any(value not in "01" for value in pooling_stat):
            raise ValueError("pooling_stat must be a four-character 0/1 mask")
        self.T = T
        self.patch_embed = _PatchEmbed(in_channels, embed_dims, pooling_stat, backend)
        self.blocks = nn.ModuleList(
            [_Block(embed_dims, num_heads, mlp_ratio, backend) for _ in range(depths)]
        )
        self.head_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.head = nn.Linear(embed_dims, num_classes)
        functional.set_step_mode(self, "m")

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
        **API Language** - :ref:`中文 <SpikeDrivenTransformer.forward-cn>` | :ref:`English <SpikeDrivenTransformer.forward-en>`

        ----

        .. _SpikeDrivenTransformer.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量

        ----

        .. _SpikeDrivenTransformer.forward-en:

        * **English**

        :param x: floating-point image ``[N, C, H, W]`` or sequence
            ``[T, N, C, H, W]``
        :type x: torch.Tensor
        :return: classification logits shaped ``[N, num_classes]``
        :rtype: torch.Tensor
        :raises ValueError: if the input is neither four- nor five-dimensional
        """
        x = self.patch_embed(self._to_sequence(x))
        for block in self.blocks:
            x = block(x)
        x = self.head_lif(x.flatten(3).mean(-1))
        return self.head(x.mean(0))


def sdt_8_384(
    T: int = 4,
    in_channels: int = 3,
    num_classes: int = 1000,
    backend: str = "torch",
) -> SpikeDrivenTransformer:
    r"""
    **API Language** - :ref:`中文 <sdt_8_384-cn>` | :ref:`English <sdt_8_384-en>`

    ----

    .. _sdt_8_384-cn:

    * **中文**

    构建 Spike-driven Transformer v1 的 SDT-8-384 配置。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :return: SDT-8-384 模型
    :rtype: SpikeDrivenTransformer

    ----

    .. _sdt_8_384-en:

    * **English**

    Builds the SDT-8-384 configuration of Spike-driven Transformer v1.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: SDT-8-384 model
    :rtype: SpikeDrivenTransformer
    """
    return SpikeDrivenTransformer(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=384,
        num_heads=8,
        depths=8,
        backend=backend,
    )
