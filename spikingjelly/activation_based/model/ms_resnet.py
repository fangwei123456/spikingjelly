import torch
import torch.nn as nn

from .. import functional, layer, neuron, surrogate

__all__ = [
    "MSResNet",
    "MaxResNet",
    "ms_resnet18",
    "ms_resnet34",
    "max_resnet18",
]


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> layer.Conv2d:
    return layer.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        step_mode="m",
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> layer.Conv2d:
    return layer.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
        step_mode="m",
    )


def _lif(backend: str, tau: float) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=tau,
        v_threshold=0.5,
        detach_reset=True,
        decay_input=False,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
        backend=backend,
    )


class _MSBlock(nn.Module):
    tau = 4.0 / 3.0
    zero_init = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        backend: str,
        downsample: nn.Module | None,
    ):
        super().__init__()
        self.spike1 = _lif(backend, self.tau)
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.spike2 = _lif(backend, self.tau)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        if self.zero_init:
            nn.init.zeros_(self.bn2.weight)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(self.conv1(self.spike1(x)))
        out = self.bn2(self.conv2(self.spike2(out)))
        return out + identity


class _MaxResNetBlock(_MSBlock):
    tau = 2.0
    zero_init = False


class _MaxResNetMaxBlock(_MaxResNetBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        backend: str,
        downsample: nn.Module | None,
    ):
        super().__init__(in_channels, out_channels, 1, backend, downsample)
        self.max_pool = layer.MaxPool2d(
            kernel_size=3, stride=stride, padding=1, step_mode="m"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(self.conv1(self.spike1(x)))
        out = self.max_pool(out)
        out = self.bn2(self.conv2(self.spike2(out)))
        return out + identity


class MSResNet(nn.Module):
    r"""
    **API Language** - :ref:`中文 <MSResNet-cn>` | :ref:`English <MSResNet-en>`

    ----

    .. _MSResNet-cn:

    * **中文**

    膜电位捷径残差网络（Membrane Shortcut ResNet）。残差分支在卷积前发放，
    shortcut 不经过跨 block 的脉冲神经元，使相加点保留膜电位信息。
    ``forward`` 接收与模型参数同设备、同浮点类型的 ``[N, C, H, W]`` 图像或
    ``[T, N, C, H, W]`` 序列，并返回 ``[N, num_classes]`` 分类结果。静态图像
    会沿时间维重复 ``T`` 次。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param layers: 三个或四个 stage 的 block 数
    :type layers: tuple[int, ...]
    :param base_channels: stem 的输出通道数
    :type base_channels: int
    :param stem_kernel_size: stem 卷积核大小
    :type stem_kernel_size: int
    :param stem_stride: stem 卷积步幅
    :type stem_stride: int
    :param stem_pool: 是否在 stem 后使用 max-pool
    :type stem_pool: bool
    :param stage_channels: 各 stage 的通道数；为 ``None`` 时从
        ``base_channels`` 逐级翻倍
    :type stage_channels: tuple[int, ...] | None
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :raises ValueError: ``layers`` 不含三个或四个值，或
        ``stage_channels`` 与 ``layers`` 长度不同

    ----

    .. _MSResNet-en:

    * **English**

    Membrane Shortcut ResNet. Residual branches spike before convolution,
    while the shortcut bypasses inter-block spiking neurons so that each merge
    remains in membrane space. ``forward`` accepts a floating-point image
    ``[N, C, H, W]`` or sequence ``[T, N, C, H, W]`` on the same device and
    with the same dtype as the model parameters, and returns logits shaped
    ``[N, num_classes]``. A static image is repeated for ``T`` time steps.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param layers: block counts in three or four stages
    :type layers: tuple[int, ...]
    :param base_channels: number of stem output channels
    :type base_channels: int
    :param stem_kernel_size: stem convolution kernel size
    :type stem_kernel_size: int
    :param stem_stride: stem convolution stride
    :type stem_stride: int
    :param stem_pool: whether to apply max-pooling after the stem
    :type stem_pool: bool
    :param stage_channels: channels in each stage; ``None`` doubles
        ``base_channels`` at every stage
    :type stage_channels: tuple[int, ...] | None
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :raises ValueError: if ``layers`` does not contain three or four values,
        or ``stage_channels`` and ``layers`` have different lengths

    **参考文献 | Reference**

    `Advancing Spiking Neural Networks towards Deep Residual Learning
    <https://arxiv.org/abs/2112.08954>`_
    """

    _block_type = _MSBlock
    _transition_block_type = _MSBlock
    _head_tau = 4.0 / 3.0

    def __init__(
        self,
        T: int = 6,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        base_channels: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_pool: bool = False,
        stage_channels: tuple[int, ...] | None = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if len(layers) not in (3, 4):
            raise ValueError("layers must contain three or four stage depths")
        if stage_channels is None:
            stage_channels = tuple(base_channels * 2**i for i in range(len(layers)))
        if len(stage_channels) != len(layers):
            raise ValueError("stage_channels must match layers")
        self.T = T
        self.in_channels = base_channels
        self.backend = backend
        self.stem = nn.Sequential(
            layer.Conv2d(
                in_channels,
                base_channels,
                stem_kernel_size,
                stride=stem_stride,
                padding=stem_kernel_size // 2,
                bias=False,
                step_mode="m",
            ),
            layer.BatchNorm2d(base_channels, step_mode="m"),
        )
        if stem_pool:
            self.stem.append(layer.MaxPool2d(3, stride=2, padding=1, step_mode="m"))

        self.layer1 = self._make_layer(stage_channels[0], layers[0], 1)
        self.layer2 = self._make_layer(stage_channels[1], layers[1], 2)
        self.layer3 = self._make_layer(stage_channels[2], layers[2], 2)
        self.layer4 = (
            self._make_layer(stage_channels[3], layers[3], 2)
            if len(layers) == 4
            else None
        )
        final_channels = stage_channels[-1]
        self.head_lif = _lif(backend, self._head_tau)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.head = layer.Linear(final_channels, num_classes, step_mode="m")
        functional.set_step_mode(self, "m")

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1x1(self.in_channels, out_channels, stride),
                layer.BatchNorm2d(out_channels, step_mode="m"),
            )
        block_type = self._transition_block_type if stride != 1 else self._block_type
        layers = [
            block_type(self.in_channels, out_channels, stride, self.backend, downsample)
        ]
        self.in_channels = out_channels
        layers.extend(
            self._block_type(out_channels, out_channels, 1, self.backend, None)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

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
        **API Language** - :ref:`中文 <MSResNet.forward-cn>` | :ref:`English <MSResNet.forward-en>`

        ----

        .. _MSResNet.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量

        ----

        .. _MSResNet.forward-en:

        * **English**

        :param x: floating-point image ``[N, C, H, W]`` or sequence
            ``[T, N, C, H, W]``
        :type x: torch.Tensor
        :return: classification logits shaped ``[N, num_classes]``
        :rtype: torch.Tensor
        :raises ValueError: if the input is neither four- nor five-dimensional
        """
        x = self._to_sequence(x)
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        if self.layer4 is not None:
            x = self.layer4(x)
        x = self.avgpool(self.head_lif(x)).flatten(2)
        return self.head(x).mean(0)


class MaxResNet(MSResNet):
    _block_type = _MaxResNetBlock
    _transition_block_type = _MaxResNetMaxBlock
    _head_tau = 2.0


def ms_resnet18(
    T: int = 6,
    in_channels: int = 3,
    num_classes: int = 1000,
    backend: str = "torch",
) -> MSResNet:
    r"""
    **API Language** - :ref:`中文 <ms_resnet18-cn>` | :ref:`English <ms_resnet18-en>`

    ----

    .. _ms_resnet18-cn:

    * **中文**

    构建 MS-ResNet-18。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :return: MS-ResNet-18 模型
    :rtype: MSResNet

    ----

    .. _ms_resnet18-en:

    * **English**

    Builds MS-ResNet-18.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: MS-ResNet-18 model
    :rtype: MSResNet
    """
    return MSResNet(
        T=T, in_channels=in_channels, num_classes=num_classes, backend=backend
    )


def ms_resnet34(
    T: int = 6,
    in_channels: int = 3,
    num_classes: int = 1000,
    backend: str = "torch",
) -> MSResNet:
    r"""
    **API Language** - :ref:`中文 <ms_resnet34-cn>` | :ref:`English <ms_resnet34-en>`

    ----

    .. _ms_resnet34-cn:

    * **中文**

    构建 MS-ResNet-34。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :return: MS-ResNet-34 模型
    :rtype: MSResNet

    ----

    .. _ms_resnet34-en:

    * **English**

    Builds MS-ResNet-34.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: MS-ResNet-34 model
    :rtype: MSResNet
    """
    return MSResNet(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        layers=(3, 4, 6, 3),
        backend=backend,
    )


def max_resnet18(
    T: int = 4,
    in_channels: int = 3,
    num_classes: int = 10,
    backend: str = "torch",
) -> MaxResNet:
    r"""
    **API Language** - :ref:`中文 <max_resnet18-cn>` | :ref:`English <max_resnet18-en>`

    ----

    .. _max_resnet18-cn:

    * **中文**

    构建 Max-ResNet-18 的 CIFAR 配置。该模型继承 :class:`MSResNet` 的膜电位
    shortcut，并在残差转换路径中加入 max-pool。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 内部脉冲神经元使用的后端
    :type backend: str
    :return: Max-ResNet-18 模型
    :rtype: MaxResNet

    ----

    .. _max_resnet18-en:

    * **English**

    Builds the CIFAR configuration of Max-ResNet-18. It inherits the membrane
    shortcut from :class:`MSResNet` and adds max-pooling to residual transitions.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend used by the internal spiking neurons
    :type backend: str
    :return: Max-ResNet-18 model
    :rtype: MaxResNet

    **参考文献 | Reference**

    `Spiking Neural Networks Need High Frequency Information
    <https://arxiv.org/abs/2505.18608>`_
    """
    return MaxResNet(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        layers=(3, 3, 2),
        stage_channels=(128, 256, 512),
        stem_kernel_size=3,
        stem_stride=1,
        stem_pool=False,
        backend=backend,
    )
