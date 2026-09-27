from copy import deepcopy

import torch
import torch.nn as nn

from .. import functional, layer, neuron, surrogate
from .ms_resnet import MSResNet, _MSBlock, _conv1x1

__all__ = [
    "MASNN",
    "AttMSResNet",
    "masnn_dvs128_gesture",
    "att_ms_resnet18",
]


def _ma_lif(backend: str, **overrides) -> neuron.LIFNode:
    """Default neuron of the MA-SNN DVS line.

    The author cell accumulates without input leak, fires at ``v_threshold``,
    and decays by a multiplicative factor after the reset. ``tau`` is that
    factor's time constant and the rectangular surrogate matches the author
    gradient width (``lens = 0.25``).
    """
    parameters = {
        "tau": 10.0 / 7.0,
        "v_threshold": 0.3,
        "detach_reset": True,
        "decay_input": False,
        "step_mode": "m",
        "surrogate_function": surrogate.Rect(alpha=2.0),
        "backend": backend,
    }
    parameters.update(overrides)
    return neuron.LIFNode(**parameters)


def _att_ms_lif(backend: str, **overrides) -> neuron.LIFNode:
    """Default neuron of the Att-MS-ResNet line.

    Author surrogate: ``|v - thresh| < lens`` with ``lens = 0.5``, i.e.
    ``Rect(alpha=1.0)``.
    """
    parameters = {
        "tau": 4.0 / 3.0,
        "v_threshold": 0.5,
        "detach_reset": True,
        "decay_input": False,
        "step_mode": "m",
        "surrogate_function": surrogate.Rect(alpha=1.0),
        "backend": backend,
    }
    parameters.update(overrides)
    return neuron.LIFNode(**parameters)


def _neuron_factory(spiking_neuron, default_factory, backend: str, kwargs: dict):
    """Build the per-neuron factory used by every block of a model.

    ``spiking_neuron=None`` keeps the paper neuron and lets ``kwargs`` override
    single fields. A user-supplied class is constructed from ``kwargs`` plus
    ``backend``, so the model-level backend applies to both paths; the class
    must therefore accept a ``backend`` argument, as every
    :class:`BaseNode <spikingjelly.activation_based.neuron.BaseNode>` does.
    """
    if spiking_neuron is None:
        return lambda: default_factory(backend, **deepcopy(kwargs))
    return lambda: spiking_neuron(backend=backend, **deepcopy(kwargs))


class _MaConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: int,
        cell_factory,
        attn_factory,
    ):
        super().__init__()
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.bn = layer.BatchNorm2d(out_channels, step_mode="m")
        self.pool = layer.AvgPool2d(pool, step_mode="m") if pool > 1 else None
        self.attn = attn_factory(out_channels)
        self.cell = cell_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.pool is not None:
            x = self.pool(x)
        return self.cell(self.attn(x))


class _MaFcBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        T: int,
        reduction_t: int,
        cell_factory,
    ):
        super().__init__()
        self.linear = layer.Linear(in_features, out_features, bias=True, step_mode="m")
        self.bn = layer.BatchNorm1d(out_features, step_mode="m")
        self.attn = layer.TemporalWiseAttention(T=T, reduction=reduction_t, dimension=2)
        # The author's fully connected TA applies a ReLU after the attention
        # product; TemporalWiseAttention itself does not include one.
        self.relu = nn.ReLU()
        self.cell = cell_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell(self.relu(self.attn(self.bn(self.linear(x)))))


class MASNN(nn.Module):
    def __init__(
        self,
        T: int = 60,
        in_channels: int = 2,
        num_classes: int = 11,
        input_size: tuple[int, int] = (32, 32),
        channels: tuple[int, ...] = (64, 128, 128),
        pools: tuple[int, ...] = (1, 2, 2),
        fc_hidden: int = 256,
        reduction_t: int = 5,
        reduction_c: int = 8,
        backend: str = "torch",
        spiking_neuron: callable = None,
        **kwargs,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MASNN.__init__-cn>` | :ref:`English <MASNN.__init__-en>`

        ----

        .. _MASNN.__init__-cn:

        * **中文**

        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_
        中用于 DVS128 Gesture 等事件流数据集的 MA-SNN 卷积网络。每个卷积块按
        ``Conv2d -> BatchNorm2d -> AvgPool2d -> MultiDimensionalAttention -> LIF``
        的顺序处理，随后是全连接块 ``Linear -> BatchNorm1d -> TemporalWiseAttention -> LIF``，
        最后对仿真时间维取平均得到分类结果。脉冲神经元为
        :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        （``tau=10/7``、``v_threshold=0.3``、``decay_input=False``、硬重置到 ``0``、
        :class:`Rect <spikingjelly.activation_based.surrogate.Rect>```(alpha=2.0)``），
        对应作者实现每步保留 30% 膜电位的设置；作者将发放位置重置到阈值而非
        ``0``，此处采用 ``LIFNode`` 的零重置近似，未与作者实现逐数值对齐。

        ``forward`` 接收与模型参数同设备、同浮点类型的 ``[N, C, H, W]`` 图像或
        ``[T, N, C, H, W]`` 序列，并返回 ``[N, num_classes]`` 分类结果。静态图像
        会沿时间维重复 ``T`` 次。由于时间注意力（TCSA 的 TA 与全连接层的
        TemporalWiseAttention）在构造时固定时间步数，输入序列的时间维必须等于
        构造参数 ``T``，否则抛出 ``ValueError``。
        处理相互独立的输入序列时，应调用
        :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>` 重置网络状态。

        :param T: 仿真时间步数
        :type T: int
        :param in_channels: 输入通道数
        :type in_channels: int
        :param num_classes: 分类类别数
        :type num_classes: int
        :param input_size: 输入空间尺寸 ``(H, W)``，必须能被 ``pools`` 中各元素的乘积整除
        :type input_size: tuple[int, int]
        :param channels: 各卷积块的输出通道数
        :type channels: tuple[int, ...]
        :param pools: 各卷积块平均池化的核大小；``1`` 表示不池化
        :type pools: tuple[int, ...]
        :param fc_hidden: 第一个全连接层的输出特征数
        :type fc_hidden: int
        :param reduction_t: 时间注意力压缩比，必须 ``<= T``
        :type reduction_t: int
        :param reduction_c: 通道注意力压缩比，必须 ``<=`` 各卷积块通道数
        :type reduction_c: int
        :param backend: 脉冲神经元使用的后端，对默认神经元和 ``spiking_neuron``
            同时生效。默认的 :class:`Rect <spikingjelly.activation_based.surrogate.Rect>`
            替代梯度只有 ``torch`` 后端支持；使用 ``cupy`` 或 ``triton`` 时需通过
            ``kwargs`` 传入受支持的替代梯度（如 ``ATan``）
        :type backend: str
        :param spiking_neuron: 脉冲神经元类；为 ``None`` 时使用论文默认的
            :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        :type spiking_neuron: callable
        :param kwargs: 传给脉冲神经元的额外参数；``spiking_neuron`` 为 ``None``
            时逐项覆盖论文默认值
        :type kwargs: dict
        :raises ValueError: ``channels`` 与 ``pools`` 长度不同，或 ``input_size``
            不能被池化下采样整除

        ----

        .. _MASNN.__init__-en:

        * **English**

        MA-SNN convolutional network for event-stream datasets such as DVS128
        Gesture, proposed in `Attention Spiking Neural Networks
        <https://ieeexplore.ieee.org/document/10032591>`_. Every convolution
        block processes ``Conv2d -> BatchNorm2d -> AvgPool2d ->
        MultiDimensionalAttention -> LIF``, followed by fully connected blocks
        ``Linear -> BatchNorm1d -> TemporalWiseAttention -> LIF`` and an average
        over simulation steps. The spiking neuron is a
        :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        (``tau=10/7``, ``v_threshold=0.3``, ``decay_input=False``, hard reset to
        ``0``, :class:`Rect <spikingjelly.activation_based.surrogate.Rect>`
        ``(alpha=2.0)``), matching the author setting that keeps 30% of the
        membrane potential per step. The author resets fired positions to the
        threshold instead of ``0``; the zero reset here is an approximation and
        is not numerically aligned with the author implementation.

        ``forward`` accepts a floating-point image ``[N, C, H, W]`` or sequence
        ``[T, N, C, H, W]`` on the same device and with the same dtype as the
        model parameters, and returns logits shaped ``[N, num_classes]``. A
        static image is repeated for ``T`` time steps. Temporal attention (TCSA's
        TA and the fully connected TemporalWiseAttention) fixes the number of
        time steps at construction, so the time dimension of a sequence input
        must equal ``T`` or ``forward`` raises ``ValueError``.
        Call :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>`
        between independent input sequences.

        :param T: number of simulation steps
        :type T: int
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of classes
        :type num_classes: int
        :param input_size: input spatial size ``(H, W)``; must be divisible by
            the product of ``pools``
        :type input_size: tuple[int, int]
        :param channels: output channels of every convolution block
        :type channels: tuple[int, ...]
        :param pools: average-pooling kernel size of every convolution block;
            ``1`` disables pooling
        :type pools: tuple[int, ...]
        :param fc_hidden: output features of the first fully connected layer
        :type fc_hidden: int
        :param reduction_t: temporal attention reduction ratio; must be ``<= T``
        :type reduction_t: int
        :param reduction_c: channel attention reduction ratio; must be ``<=`` the
            channel count of every convolution block
        :type reduction_c: int
        :param backend: backend of the spiking neurons, applied both to the
            default neuron and to ``spiking_neuron``. The default
            :class:`Rect <spikingjelly.activation_based.surrogate.Rect>`
            surrogate is only supported by the ``torch`` backend; pass a
            supported surrogate such as ``ATan`` through ``kwargs`` to use
            ``cupy`` or ``triton``
        :type backend: str
        :param spiking_neuron: spiking neuron class; ``None`` uses the paper
            default :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        :type spiking_neuron: callable
        :param kwargs: extra arguments for the spiking neuron; they override the
            paper defaults field by field when ``spiking_neuron`` is ``None``
        :type kwargs: dict
        :raises ValueError: if ``channels`` and ``pools`` have different lengths
            or ``input_size`` is not divisible by the pooling downsampling

        **参考文献 | Reference**

        `Attention Spiking Neural Networks
        <https://ieeexplore.ieee.org/document/10032591>`_
        """
        super().__init__()
        if len(channels) != len(pools):
            raise ValueError("channels and pools must have the same length")
        height, width = input_size
        for pool in pools:
            if pool > 1 and (height % pool or width % pool):
                raise ValueError("input_size must be divisible by the pooling factors")
            height, width = height // pool, width // pool

        self.T = T
        cell_factory = _neuron_factory(spiking_neuron, _ma_lif, backend, kwargs)

        def attn_factory(channel_count: int):
            return layer.MultiDimensionalAttention(
                T=T,
                C=channel_count,
                reduction_t=reduction_t,
                reduction_c=reduction_c,
            )

        blocks = []
        current_channels = in_channels
        for out_channels, pool in zip(channels, pools):
            blocks.append(
                _MaConvBlock(
                    current_channels, out_channels, pool, cell_factory, attn_factory
                )
            )
            current_channels = out_channels
        self.conv_blocks = nn.Sequential(*blocks)

        feature_size = channels[-1] * height * width
        self.fc_blocks = nn.Sequential(
            _MaFcBlock(feature_size, fc_hidden, T, reduction_t, cell_factory),
            _MaFcBlock(fc_hidden, num_classes, T, reduction_t, cell_factory),
        )
        functional.set_step_mode(self, "m")

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if x.ndim == 5:
            if x.shape[0] != self.T:
                raise ValueError(
                    f"expected a sequence with T={self.T} time steps, but got "
                    f"shape {tuple(x.shape)}"
                )
            return x
        raise ValueError(
            f"expected 4D image or 5D sequence input, but got shape {x.shape}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <MASNN.forward-cn>` | :ref:`English <MASNN.forward-en>`

        ----

        .. _MASNN.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量，或序列的时间维不等于 ``T``

        ----

        .. _MASNN.forward-en:

        * **English**

        :param x: floating-point image ``[N, C, H, W]`` or sequence
            ``[T, N, C, H, W]``
        :type x: torch.Tensor
        :return: classification logits shaped ``[N, num_classes]``
        :rtype: torch.Tensor
        :raises ValueError: if the input is neither four- nor five-dimensional,
            or the sequence time dimension is not ``T``
        """
        x = self._to_sequence(x)
        x = self.conv_blocks(x)
        x = x.flatten(2)
        x = self.fc_blocks(x)
        return x.mean(0)


def masnn_dvs128_gesture(
    T: int = 60,
    in_channels: int = 2,
    num_classes: int = 11,
    input_size: tuple[int, int] = (32, 32),
    backend: str = "torch",
    spiking_neuron: callable = None,
    **kwargs,
) -> MASNN:
    r"""
    **API Language** - :ref:`中文 <masnn_dvs128_gesture-cn>` | :ref:`English <masnn_dvs128_gesture-en>`

    ----

    .. _masnn_dvs128_gesture-cn:

    * **中文**

    构建作者论文中 DVS128 Gesture 配置的 MA-SNN 卷积网络：三个卷积块
    ``(64, 128, 128)``、池化 ``(1, 2, 2)``、``fc_hidden=256``、
    ``reduction_t=5``、``reduction_c=8``。

    :param T: 仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param input_size: 输入空间尺寸 ``(H, W)``
    :type input_size: tuple[int, int]
    :param backend: 脉冲神经元使用的后端；默认 ``Rect`` 替代梯度只有 ``torch``
        后端支持
    :type backend: str
    :param spiking_neuron: 脉冲神经元类；为 ``None`` 时使用论文默认神经元
    :type spiking_neuron: callable
    :param kwargs: 传给脉冲神经元的额外参数
    :type kwargs: dict
    :return: DVS128 Gesture 配置的 MA-SNN 模型
    :rtype: MASNN

    ----

    .. _masnn_dvs128_gesture-en:

    * **English**

    Builds the MA-SNN convolutional network in the DVS128 Gesture configuration
    of the paper: three convolution blocks ``(64, 128, 128)``, pooling
    ``(1, 2, 2)``, ``fc_hidden=256``, ``reduction_t=5``, and ``reduction_c=8``.

    :param T: number of simulation steps
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param input_size: input spatial size ``(H, W)``
    :type input_size: tuple[int, int]
    :param backend: backend of the spiking neurons; the default ``Rect``
        surrogate is only supported by the ``torch`` backend
    :type backend: str
    :param spiking_neuron: spiking neuron class; ``None`` uses the paper default
    :type spiking_neuron: callable
    :param kwargs: extra arguments for the spiking neuron
    :type kwargs: dict
    :return: MA-SNN model in the DVS128 Gesture configuration
    :rtype: MASNN
    """
    return MASNN(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=input_size,
        channels=(64, 128, 128),
        pools=(1, 2, 2),
        fc_hidden=256,
        reduction_t=5,
        reduction_c=8,
        backend=backend,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )


class _AttMSBlock(_MSBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        backend: str,
        downsample: nn.Module | None,
        attention: nn.Module,
        cell_factory,
    ):
        super().__init__(in_channels, out_channels, stride, backend, downsample)
        # The parent builds its own default nodes; replace them so both the
        # author neuron and a user-supplied spiking_neuron reach every block.
        self.spike1 = cell_factory()
        self.spike2 = cell_factory()
        # Author init: the second BN weight starts at 0.2 * thresh = 0.1.
        nn.init.constant_(self.bn2.weight, 0.1)
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(self.conv1(self.spike1(x)))
        out = self.bn2(self.conv2(self.spike2(out)))
        return self.attention(out) + identity


class AttMSResNet(MSResNet):
    _block_type = _AttMSBlock
    _transition_block_type = _AttMSBlock

    def __init__(
        self,
        T: int = 1,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        base_channels: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_pool: bool = False,
        stage_channels: tuple[int, ...] | None = None,
        reduction_c: int = 8,
        dropout: float = 0.2,
        backend: str = "torch",
        spiking_neuron: callable = None,
        **kwargs,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <AttMSResNet-cn>` | :ref:`English <AttMSResNet-en>`

        ----

        .. _AttMSResNet-cn:

        * **中文**

        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_
        中的 Att-MS-ResNet。在 :class:`MSResNet <spikingjelly.activation_based.model.ms_resnet.MSResNet>`
        的膜电位 shortcut 基础上，每个残差块在与 shortcut 相加之前对卷积分支施加
        通道-空间注意力（:class:`MultiDimensionalAttention
        <spikingjelly.activation_based.layer.attention.MultiDimensionalAttention>`
        的 ``use_temporal=False`` 组合，对应作者的 CSA）。与 :class:`MSResNet` 的其他差异
        均来自作者实现：shortcut 需要下采样时使用 ``AvgPool2d`` 加 stride 为 1 的 1x1 卷积，
        第二个 BatchNorm 的权重初始化为 ``0.1``，替代梯度为 ``Rect(alpha=1.0)``，分类头
        先对时间取平均、经过 ``Dropout`` 后再做线性分类。

        ``forward`` 的输入输出约定与 :class:`MSResNet` 相同：接收 ``[N, C, H, W]``
        图像（沿时间重复 ``T`` 次）或 ``[T, N, C, H, W]`` 序列，返回
        ``[N, num_classes]``。处理相互独立的输入序列时，应调用
        :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>` 重置网络状态。

        除以下参数外，构造语义与 :class:`MSResNet` 相同：

        :param T: 静态图像输入的仿真时间步数；作者的 ImageNet 配置为 ``1``
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
        :param reduction_c: 通道注意力压缩比，必须 ``<=`` 各 stage 的通道数
        :type reduction_c: int
        :param dropout: 分类头 Dropout 概率
        :type dropout: float
        :param backend: 脉冲神经元使用的后端，对默认神经元和 ``spiking_neuron``
            同时生效。默认的 :class:`Rect <spikingjelly.activation_based.surrogate.Rect>`
            替代梯度只有 ``torch`` 后端支持；使用 ``cupy`` 或 ``triton`` 时需通过
            ``kwargs`` 传入受支持的替代梯度（如 ``ATan``）
        :type backend: str
        :param spiking_neuron: 脉冲神经元类；为 ``None`` 时使用论文默认的
            :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        :type spiking_neuron: callable
        :param kwargs: 传给脉冲神经元的额外参数；``spiking_neuron`` 为 ``None``
            时逐项覆盖论文默认值，否则完全决定神经元构造（``backend`` 除外）
        :type kwargs: dict
        :raises ValueError: ``layers`` 不含三个或四个值，或 ``stage_channels``
            与 ``layers`` 长度不同

        ----

        .. _AttMSResNet-en:

        * **English**

        Att-MS-ResNet from `Attention Spiking Neural Networks
        <https://ieeexplore.ieee.org/document/10032591>`_. On top of the membrane
        shortcut of :class:`MSResNet
        <spikingjelly.activation_based.model.ms_resnet.MSResNet>`, every residual
        block applies channel-spatial attention (:class:`MultiDimensionalAttention
        <spikingjelly.activation_based.layer.attention.MultiDimensionalAttention>`
        with ``use_temporal=False``, the author's CSA) to the convolution branch
        before adding the shortcut. The remaining differences from
        :class:`MSResNet` follow the author implementation: a downsampling shortcut
        uses ``AvgPool2d`` plus a stride-1 1x1 convolution, the second BatchNorm
        weight is initialized to ``0.1``, the surrogate gradient is
        ``Rect(alpha=1.0)``, and the classification head averages over time and
        applies ``Dropout`` before the linear classifier.

        ``forward`` follows the :class:`MSResNet` conventions: it accepts an image
        ``[N, C, H, W]`` (repeated for ``T`` steps) or a sequence
        ``[T, N, C, H, W]`` and returns logits shaped ``[N, num_classes]``.
        Call :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>`
        between independent input sequences.

        Constructor semantics match :class:`MSResNet` except where noted:

        :param T: number of simulation steps used for static images; the author
            ImageNet configuration uses ``1``
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
        :param reduction_c: channel attention reduction ratio; must be ``<=`` the
            channel count of every stage
        :type reduction_c: int
        :param dropout: dropout probability of the classification head
        :type dropout: float
        :param backend: backend of the spiking neurons, applied both to the
            default neuron and to ``spiking_neuron``. The default
            :class:`Rect <spikingjelly.activation_based.surrogate.Rect>`
            surrogate is only supported by the ``torch`` backend; pass a
            supported surrogate such as ``ATan`` through ``kwargs`` to use
            ``cupy`` or ``triton``
        :type backend: str
        :param spiking_neuron: spiking neuron class; ``None`` uses the paper
            default :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>`
        :type spiking_neuron: callable
        :param kwargs: extra arguments for the spiking neuron; they override the
            paper defaults field by field when ``spiking_neuron`` is ``None``,
            and otherwise fully define its construction apart from ``backend``
        :type kwargs: dict
        :raises ValueError: if ``layers`` does not contain three or four values,
            or ``stage_channels`` and ``layers`` have different lengths

        **参考文献 | Reference**

        `Attention Spiking Neural Networks
        <https://ieeexplore.ieee.org/document/10032591>`_
        """
        self._cell_factory = _neuron_factory(
            spiking_neuron, _att_ms_lif, backend, kwargs
        )
        self._attention_factory = lambda channels: layer.MultiDimensionalAttention(
            T=T, C=channels, reduction_c=reduction_c, use_temporal=False
        )
        super().__init__(
            T=T,
            in_channels=in_channels,
            num_classes=num_classes,
            layers=layers,
            base_channels=base_channels,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            stem_pool=stem_pool,
            stage_channels=stage_channels,
            backend=backend,
        )
        self.head_lif = self._cell_factory()
        self.dropout = nn.Dropout(dropout)
        final_channels = self.head.in_features
        self.head = nn.Linear(final_channels, num_classes)
        functional.set_step_mode(self, "m")

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample_bn = layer.BatchNorm2d(out_channels, step_mode="m")
            nn.init.constant_(downsample_bn.weight, self._bn_weight)
            modules = []
            if stride != 1:
                # ceil_mode matches the stride-2 padded 3x3 convolution of the
                # residual branch for odd sizes; even sizes, the only ones the
                # author configuration reaches, are unaffected.
                modules.append(
                    layer.AvgPool2d(stride, stride, ceil_mode=True, step_mode="m")
                )
            modules.extend([_conv1x1(self.inplanes, out_channels, 1), downsample_bn])
            downsample = nn.Sequential(*modules)
        layers = [
            self._block_type(
                self.inplanes,
                out_channels,
                stride,
                self.backend,
                downsample,
                self._attention_factory(out_channels),
                self._cell_factory,
            )
        ]
        self.inplanes = out_channels
        layers.extend(
            self._block_type(
                out_channels,
                out_channels,
                1,
                self.backend,
                None,
                self._attention_factory(out_channels),
                self._cell_factory,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <AttMSResNet.forward-cn>` | :ref:`English <AttMSResNet.forward-en>`

        ----

        .. _AttMSResNet.forward-cn:

        * **中文**

        :param x: ``[N, C, H, W]`` 浮点图像或 ``[T, N, C, H, W]`` 浮点序列
        :type x: torch.Tensor
        :return: ``[N, num_classes]`` 分类结果
        :rtype: torch.Tensor
        :raises ValueError: 输入不是四维或五维张量

        ----

        .. _AttMSResNet.forward-en:

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
        return self.head(self.dropout(x.mean(0)))


def att_ms_resnet18(
    T: int = 1,
    in_channels: int = 3,
    num_classes: int = 1000,
    backend: str = "torch",
    spiking_neuron: callable = None,
    **kwargs,
) -> AttMSResNet:
    r"""
    **API Language** - :ref:`中文 <att_ms_resnet18-cn>` | :ref:`English <att_ms_resnet18-en>`

    ----

    .. _att_ms_resnet18-cn:

    * **中文**

    构建作者 ImageNet 配置的 Att-MS-ResNet-18（``T=1``、通道-空间注意力、
    ``reduction_c=8``）。更深的配置可通过 :class:`AttMSResNet` 的 ``layers``
    参数构造。

    :param T: 静态图像输入的仿真时间步数
    :type T: int
    :param in_channels: 输入通道数
    :type in_channels: int
    :param num_classes: 分类类别数
    :type num_classes: int
    :param backend: 脉冲神经元使用的后端；默认 ``Rect`` 替代梯度只有 ``torch``
        后端支持
    :type backend: str
    :param spiking_neuron: 脉冲神经元类；为 ``None`` 时使用论文默认神经元
    :type spiking_neuron: callable
    :param kwargs: 传给脉冲神经元的额外参数
    :type kwargs: dict
    :return: Att-MS-ResNet-18 模型
    :rtype: AttMSResNet

    ----

    .. _att_ms_resnet18-en:

    * **English**

    Builds Att-MS-ResNet-18 in the author ImageNet configuration (``T=1``,
    channel-spatial attention, ``reduction_c=8``). Deeper variants can be built
    through the ``layers`` parameter of :class:`AttMSResNet`.

    :param T: number of simulation steps used for static images
    :type T: int
    :param in_channels: number of input channels
    :type in_channels: int
    :param num_classes: number of classes
    :type num_classes: int
    :param backend: backend of the spiking neurons; the default ``Rect``
        surrogate is only supported by the ``torch`` backend
    :type backend: str
    :param spiking_neuron: spiking neuron class; ``None`` uses the paper default
    :type spiking_neuron: callable
    :param kwargs: extra arguments for the spiking neuron
    :type kwargs: dict
    :return: Att-MS-ResNet-18 model
    :rtype: AttMSResNet
    """
    return AttMSResNet(
        T=T,
        in_channels=in_channels,
        num_classes=num_classes,
        layers=(2, 2, 2, 2),
        backend=backend,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
