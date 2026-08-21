import torch
import torch.nn as nn

from .. import functional, layer, neuron
from ..layer.attention import QKAttention, SpikingSelfAttention

__all__ = ["QKFormer", "qkformer_10_384"]


class _PatchEmbedInit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, backend: str):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = layer.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(hidden_channels, step_mode="m")
        self.pool1 = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.conv2 = layer.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.pool2 = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)
        self.conv3 = layer.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, step_mode="m"
        )
        self.bn3 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif3 = neuron.LIFNode(step_mode="m", backend=backend)
        self.shortcut = nn.Sequential(
            layer.Conv2d(
                hidden_channels, out_channels, kernel_size=1, stride=2, step_mode="m"
            ),
            layer.BatchNorm2d(out_channels, step_mode="m"),
            neuron.LIFNode(step_mode="m", backend=backend),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.lif1(x)
        identity = x
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.lif2(x)
        x = self.lif3(self.bn3(self.conv3(x)))
        return x + self.shortcut(identity)


class _PatchEmbedStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, backend: str):
        super().__init__()
        self.conv1 = layer.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.pool = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.conv2 = layer.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)
        self.shortcut = nn.Sequential(
            layer.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2, step_mode="m"
            ),
            layer.BatchNorm2d(out_channels, step_mode="m"),
            neuron.LIFNode(step_mode="m", backend=backend),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.lif1(self.pool(self.bn1(self.conv1(x))))
        x = self.lif2(self.bn2(self.conv2(x)))
        return x + self.shortcut(identity)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, backend: str):
        super().__init__()
        self.fc1 = layer.Conv1d(dim, hidden_dim, kernel_size=1, step_mode="m")
        self.bn1 = layer.BatchNorm1d(hidden_dim, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.fc2 = layer.Conv1d(hidden_dim, dim, kernel_size=1, step_mode="m")
        self.bn2 = layer.BatchNorm1d(dim, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif1(self.bn1(self.fc1(x)))
        return self.lif2(self.bn2(self.fc2(x)))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qka_type: str,
        backend: str,
        self_attention: bool,
    ):
        super().__init__()
        self.attn = (
            SpikingSelfAttention(dim, num_heads, backend=backend)
            if self_attention
            else QKAttention(dim, num_heads, qka_type=qka_type, backend=backend)
        )
        self.mlp = _MLP(dim, int(dim * mlp_ratio), backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.flatten(3)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x.reshape(shape)


class QKFormer(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 384,
        num_heads: tuple[int, int, int] = (3, 6, 6),
        depths: tuple[int, int, int] = (1, 2, 7),
        mlp_ratio: float = 4.0,
        qka_type: str = "token",
        backend: str = "torch",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <QKFormer.__init__-cn>` | :ref:`English <QKFormer.__init__-en>`

        ----

        .. _QKFormer.__init__-cn:

        * **中文**

        QKFormer 层级脉冲 Transformer。前两级使用 Q-K Attention，最后一级使用
        Spiking Self-Attention。``forward`` 接收与模型参数同设备、同浮点类型的
        ``[N, C, H, W]`` 图像或 ``[T, N, C, H, W]`` 序列，并返回
        ``[N, num_classes]`` 分类结果。静态图像会沿时间维重复 ``T`` 次。

        :param T: 静态图像输入的仿真时间步数
        :type T: int
        :param in_channels: 输入通道数
        :type in_channels: int
        :param num_classes: 分类类别数
        :type num_classes: int
        :param embed_dims: 最后一级的通道数，必须能被 ``4`` 整除
        :type embed_dims: int
        :param num_heads: 三个 stage 的 attention head 数
        :type num_heads: tuple[int, int, int]
        :param depths: 三个 stage 的 block 数
        :type depths: tuple[int, int, int]
        :param mlp_ratio: MLP 隐藏通道数相对输入通道数的倍率
        :type mlp_ratio: float
        :param qka_type: Q-K Attention 类型，可选 ``"token"`` 或 ``"channel"``
        :type qka_type: str
        :param backend: 内部脉冲神经元使用的后端
        :type backend: str
        :raises ValueError: ``num_heads`` 或 ``depths`` 不含三个值、
            ``embed_dims`` 不能被 ``4`` 整除、stage 通道数不能被对应 head 数整除，
            或 ``qka_type`` 无效

        ----

        .. _QKFormer.__init__-en:

        * **English**

        Hierarchical QKFormer. The first two stages use Q-K Attention and the final
        stage uses Spiking Self-Attention. ``forward`` accepts a floating-point
        image ``[N, C, H, W]`` or sequence ``[T, N, C, H, W]`` on the same device
        and with the same dtype as the model parameters, and returns logits shaped
        ``[N, num_classes]``. A static image is repeated for ``T`` time steps.

        :param T: number of simulation steps used for static images
        :type T: int
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of classes
        :type num_classes: int
        :param embed_dims: number of channels in the final stage; must be divisible by ``4``
        :type embed_dims: int
        :param num_heads: attention heads in the three stages
        :type num_heads: tuple[int, int, int]
        :param depths: block counts in the three stages
        :type depths: tuple[int, int, int]
        :param mlp_ratio: ratio of MLP hidden channels to input channels
        :type mlp_ratio: float
        :param qka_type: Q-K Attention type, either ``"token"`` or ``"channel"``
        :type qka_type: str
        :param backend: backend used by the internal spiking neurons
        :type backend: str
        :raises ValueError: if ``num_heads`` or ``depths`` does not contain three
            values, ``embed_dims`` is not divisible by ``4``, a stage dimension is
            not divisible by its head count, or ``qka_type`` is invalid

        **参考文献 | Reference**

        `QKFormer: Hierarchical Spiking Transformer using Q-K Attention
        <https://arxiv.org/abs/2403.16552>`_
        """
        super().__init__()
        if len(num_heads) != 3 or len(depths) != 3:
            raise ValueError("num_heads and depths must contain three stage values")
        if embed_dims % 4:
            raise ValueError("embed_dims must be divisible by 4")

        dims = (embed_dims // 4, embed_dims // 2, embed_dims)
        self.T = T
        self.patch_embed1 = _PatchEmbedInit(in_channels, dims[0], backend)
        self.patch_embed2 = _PatchEmbedStage(dims[0], dims[1], backend)
        self.patch_embed3 = _PatchEmbedStage(dims[1], dims[2], backend)
        self.stage1 = nn.ModuleList(
            [
                _Block(dims[0], num_heads[0], mlp_ratio, qka_type, backend, False)
                for _ in range(depths[0])
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                _Block(dims[1], num_heads[1], mlp_ratio, qka_type, backend, False)
                for _ in range(depths[1])
            ]
        )
        self.stage3 = nn.ModuleList(
            [
                _Block(dims[2], num_heads[2], mlp_ratio, qka_type, backend, True)
                for _ in range(depths[2])
            ]
        )
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
        **API Language** - :ref:`中文 <QKFormer.forward-cn>` | :ref:`English <QKFormer.forward-en>`

        ----

        .. _QKFormer.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量

        ----

        .. _QKFormer.forward-en:

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
        return self.head(x.flatten(3).mean(dim=-1).mean(0))


def qkformer_10_384(
    T: int = 4,
    in_channels: int = 3,
    num_classes: int = 1000,
    backend: str = "torch",
) -> QKFormer:
    r"""
    **API Language** - :ref:`中文 <qkformer_10_384-cn>` | :ref:`English <qkformer_10_384-en>`

    ----

    .. _qkformer_10_384-cn:

    * **中文**

    构建 QKFormer-10-384。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :return: QKFormer-10-384 模型
    :rtype: QKFormer

    ----

    .. _qkformer_10_384-en:

    * **English**

    Builds QKFormer-10-384.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: QKFormer-10-384 model
    :rtype: QKFormer
    """
    return QKFormer(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=384,
        num_heads=(6, 6, 6),
        depths=(1, 2, 7),
        backend=backend,
    )
