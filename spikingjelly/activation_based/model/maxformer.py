import torch
import torch.nn as nn

from .. import functional, layer, neuron

__all__ = ["MaxFormer", "maxformer_10_384"]


class _MaxEmbedInit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, backend: str):
        super().__init__()
        hidden = out_channels // 2
        self.conv1 = layer.Conv2d(
            in_channels, hidden, 3, stride=2, padding=1, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(hidden, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.conv2 = layer.Conv2d(
            hidden, out_channels, 3, stride=2, padding=1, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)
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
    def __init__(self, in_channels: int, out_channels: int, backend: str):
        super().__init__()
        self.conv1 = layer.Conv2d(
            in_channels, out_channels, 3, padding=1, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif1 = neuron.LIFNode(step_mode="m", backend=backend)
        self.pool = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.conv2 = layer.Conv2d(
            out_channels, out_channels, 3, padding=1, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = neuron.LIFNode(step_mode="m", backend=backend)
        self.shortcut = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, 1, stride=1, step_mode="m"),
            layer.BatchNorm2d(out_channels, step_mode="m"),
            layer.MaxPool2d(3, stride=2, padding=1, step_mode="m"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.pool(self.lif1(self.bn1(self.conv1(x))))
        x = self.lif2(x)
        x = self.bn2(self.conv2(x))
        return x + identity


class _SpatialMLP(nn.Module):
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


class _DWCBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, mlp_ratio: float, backend: str):
        super().__init__()
        self.lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.dwconv = layer.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            step_mode="m",
        )
        self.bn = layer.BatchNorm2d(dim, step_mode="m")
        self.mlp = _SpatialMLP(dim, int(dim * mlp_ratio), backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bn(self.dwconv(self.lif(x)))
        return self.mlp(x)


class _SSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, backend: str):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.x_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.q_conv = layer.Conv1d(dim, dim, 1, bias=False, step_mode="m")
        self.q_bn = layer.BatchNorm1d(dim, step_mode="m")
        self.q_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.k_conv = layer.Conv1d(dim, dim, 1, bias=False, step_mode="m")
        self.k_bn = layer.BatchNorm1d(dim, step_mode="m")
        self.k_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.v_conv = layer.Conv1d(dim, dim, 1, bias=False, step_mode="m")
        self.v_bn = layer.BatchNorm1d(dim, step_mode="m")
        self.v_lif = neuron.LIFNode(step_mode="m", backend=backend)
        self.attn_lif = neuron.LIFNode(v_threshold=0.5, step_mode="m", backend=backend)
        self.proj_conv = layer.Conv1d(dim, dim, 1, step_mode="m")
        self.proj_bn = layer.BatchNorm1d(dim, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, N, C, H, W = x.shape
        identity = x
        x = self.x_lif(x).flatten(3)
        q = self.q_lif(self.q_bn(self.q_conv(x)))
        k = self.k_lif(self.k_bn(self.k_conv(x)))
        v = self.v_lif(self.v_bn(self.v_conv(x)))
        q = q.transpose(-1, -2).reshape(
            T, N, H * W, self.num_heads, C // self.num_heads
        )
        k = k.transpose(-1, -2).reshape(
            T, N, H * W, self.num_heads, C // self.num_heads
        )
        v = v.transpose(-1, -2).reshape(
            T, N, H * W, self.num_heads, C // self.num_heads
        )
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        x = (q @ (k.transpose(-2, -1) @ v)) * 0.125
        x = x.transpose(3, 4).reshape(T, N, C, H * W)
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, N, C, H, W)
        return x + identity


class _SSABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, backend: str):
        super().__init__()
        self.attn = _SSA(dim, num_heads, backend)
        self.mlp = _SpatialMLP(dim, int(dim * mlp_ratio), backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x))


class MaxFormer(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: int = 384,
        depths: tuple[int, int, int] = (1, 2, 7),
        mlp_ratio: float = 4.0,
        backend: str = "torch",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxFormer.__init__-cn>` | :ref:`English <MaxFormer.__init__-en>`

        ----

        .. _MaxFormer.__init__-cn:

        * **中文**

        Max-Former 层级脉冲 Transformer。模型使用额外的 max-pool 和前两级
        depth-wise convolution 保留高频特征，最后一级使用 Spiking Self-Attention。
        ``forward`` 接收与模型参数同设备、同浮点类型的 ``[N, C, H, W]`` 图像或
        ``[T, N, C, H, W]`` 序列，并返回 ``[N, num_classes]`` 分类结果。静态图像
        会沿时间维重复 ``T`` 次。

        :param T: 静态图像输入的仿真时间步数
        :type T: int
        :param in_channels: 输入通道数
        :type in_channels: int
        :param num_classes: 分类类别数
        :type num_classes: int
        :param embed_dims: 最后一级的通道数，必须能被 ``4`` 整除
        :type embed_dims: int
        :param depths: 三个 stage 的 block 数
        :type depths: tuple[int, int, int]
        :param mlp_ratio: MLP 隐藏通道数相对输入通道数的倍率
        :type mlp_ratio: float
        :param backend: 内部脉冲神经元使用的后端
        :type backend: str
        :raises ValueError: ``depths`` 不含三个值，或 ``embed_dims`` 不能被
            ``4`` 整除

        ----

        .. _MaxFormer.__init__-en:

        * **English**

        Hierarchical Max-Former. Extra max-pooling and depth-wise convolutions in
        the first two stages preserve high-frequency features, followed by Spiking
        Self-Attention in the final stage. ``forward`` accepts a floating-point
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
        :param depths: block counts in the three stages
        :type depths: tuple[int, int, int]
        :param mlp_ratio: ratio of MLP hidden channels to input channels
        :type mlp_ratio: float
        :param backend: backend used by the internal spiking neurons
        :type backend: str
        :raises ValueError: if ``depths`` does not contain three values or
            ``embed_dims`` is not divisible by ``4``

        **参考文献 | Reference**

        `Spiking Neural Networks Need High Frequency Information
        <https://arxiv.org/abs/2505.18608>`_
        """
        super().__init__()
        if len(depths) != 3 or embed_dims % 4:
            raise ValueError(
                "depths must contain three values and embed_dims be divisible by 4"
            )
        dims = (embed_dims // 4, embed_dims // 2, embed_dims)
        self.T = T
        self.patch_embed1 = _MaxEmbedInit(in_channels, dims[0], backend)
        self.patch_embed2 = _MaxEmbedStage(dims[0], dims[1], backend)
        self.patch_embed3 = _MaxEmbedStage(dims[1], dims[2], backend)
        self.stage1 = nn.ModuleList(
            [_DWCBlock(dims[0], 7, mlp_ratio, backend) for _ in range(depths[0])]
        )
        self.stage2 = nn.ModuleList(
            [_DWCBlock(dims[1], 5, mlp_ratio, backend) for _ in range(depths[1])]
        )
        self.stage3 = nn.ModuleList(
            [
                _SSABlock(dims[2], max(1, dims[2] // 64), mlp_ratio, backend)
                for _ in range(depths[2])
            ]
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
    backend: str = "torch",
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
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
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
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: Max-Former-10-384 model
    :rtype: MaxFormer
    """
    return MaxFormer(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=384,
        depths=(1, 2, 7),
        backend=backend,
    )
