import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from .. import base, functional


__all__ = [
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'NeuNorm',
    'ThresholdDependentBatchNorm1d',
    'ThresholdDependentBatchNorm2d',
    'ThresholdDependentBatchNorm3d',
    'TemporalEffectiveBatchNorm1d',
    'TemporalEffectiveBatchNorm2d',
    'TemporalEffectiveBatchNorm3d',
    "BatchNormThroughTime1d",
    'BatchNormThroughTime2d',
    "BatchNormThroughTime3d",
]


class BatchNorm1d(nn.BatchNorm1d, base.StepModule):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode='s'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNorm1d-cn>` | :ref:`English <BatchNorm1d-en>`

        ----

        .. _BatchNorm1d-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm1d`

        ----

        .. _BatchNorm1d-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm1d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4 and x.dim() != 3:
                raise ValueError(f'expected x with shape [T, N, C, L] or [T, N, C], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)


class BatchNorm2d(nn.BatchNorm2d, base.StepModule):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode='s'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNorm2d-cn>` | :ref:`English <BatchNorm2d-en>`

        ----

        .. _BatchNorm2d-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm2d`

        ----

        .. _BatchNorm2d-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm2d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)


class BatchNorm3d(nn.BatchNorm3d, base.StepModule):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode='s'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNorm3d-cn>` | :ref:`English <BatchNorm3d-en>`

        ----

        .. _BatchNorm3d-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm3d`

        ----

        .. _BatchNorm3d-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm3d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)


class NeuNorm(base.MemoryModule):

    def __init__(
        self, in_channels: int, height: int, width: int, k: float = 0.9,
        shared_across_channels: bool = False, step_mode: str = 's'
    ):
        r"""
        **API Language:**
        :ref:`中文 <NeuNorm-cn>` | :ref:`English <NeuNorm-en>`

        ----

        .. _NeuNorm-cn:

        * **中文**

        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_
        中提出的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如： ``Conv2d -> LIF -> NeuNorm`` 。

        要求输入的尺寸是 ``[batch_size, in_channels, height, width]``。

        ``in_channels`` 是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        ``k`` 是动量项系数，相当于论文中的 :math:`k_{\tau 2}`。

        论文中的 :math:`\frac{v}{F}` 会根据 :math:`k_{\tau 2} + vF = 1` 自动算出。

        :param in_channels: 输入数据的通道数
        :type in_channels: int

        :param height: 输入数据的宽
        :type height: int

        :param width: 输入数据的高
        :type width: int

        :param k: 动量项系数
        :type k: float

        :param shared_across_channels: 可学习的权重 ``w`` 是否在通道这一维度上共享。设置为 ``True`` 可以大幅度节省内存
        :type shared_across_channels: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _NeuNorm-en:

        * **English**

        The NeuNorm layer is proposed in
        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_.

        It should be placed after spiking neurons behind convolution layer, e.g., ``Conv2d -> LIF -> NeuNorm`` .

        The input should be a 4-D tensor with ``shape = [batch_size, in_channels, height, width]``.

        ``in_channels`` is the channels of input，which is :math:`F` in the paper.

        ``k`` is the momentum factor，which is :math:`k_{\tau 2}` in the paper.

        :math:`\frac{v}{F}` will be calculated by :math:`k_{\tau 2} + vF = 1` autonomously.

        :param in_channels: channels of input
        :type in_channels: int

        :param height: height of input
        :type height: int

        :param width: height of width
        :type width: int

        :param k: momentum factor
        :type k: float

        :param shared_across_channels: whether the learnable parameter ``w`` is
            shared over channel dim. If set ``True``, the consumption of memory
            can decrease largely
        :type shared_across_channels: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.step_mode = step_mode
        self.register_memory('x', 0.)
        self.k0 = k
        self.k1 = (1. - self.k0) / in_channels ** 2
        if shared_across_channels:
            self.w = nn.Parameter(Tensor(1, height, width))
        else:
            self.w = nn.Parameter(Tensor(in_channels, height, width))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def single_step_forward(self, in_spikes: Tensor):
        self.x = self.k0 * self.x + self.k1 * in_spikes.sum(dim=1, keepdim=True)
        # x.shape = [batch_size, 1, height, width]
        return in_spikes - self.w * self.x

    def extra_repr(self) -> str:
        return f'shape={self.w.shape}'


class _ThresholdDependentBatchNormBase(_BatchNorm, base.MultiStepModule):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = 'm'
        self.alpha = alpha
        self.v_th = v_th
        assert self.affine, "ThresholdDependentBatchNorm needs to set `affine = True`!"
        torch.nn.init.constant_(self.weight, alpha * v_th)

    def forward(self, x_seq):
        return functional.seq_to_ann_forward(x_seq, super().forward)


class ThresholdDependentBatchNorm1d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <ThresholdDependentBatchNorm1d.__init__-cn>` | :ref:`English <ThresholdDependentBatchNorm1d.__init__-en>`

        ----

        .. _ThresholdDependentBatchNorm1d.__init__-cn:

        * **中文**

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_
        一文提出的 Threshold-Dependent Batch Normalization (tdBN)。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， tdBN 需要跨时间步求统计量。

        :param alpha: 由网络结构决定的超参数
        :type alpha: float

        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        ----

        .. _ThresholdDependentBatchNorm1d.__init__-en:

        * **English**

        The Threshold-Dependent Batch Normalization (tdBN) proposed in
        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because tdBN needs to calculate statistics across time steps.

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float

        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 4 - 1 or input.dim() == 3 - 1  # [T * N, C, L]


class ThresholdDependentBatchNorm2d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <ThresholdDependentBatchNorm2d.__init__-cn>` | :ref:`English <ThresholdDependentBatchNorm2d.__init__-en>`

        ----

        .. _ThresholdDependentBatchNorm2d.__init__-cn:

        * **中文**

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_
        一文提出的 Threshold-Dependent Batch Normalization (tdBN)。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， tdBN 需要跨时间步求统计量。

        :param alpha: 由网络结构决定的超参数
        :type alpha: float

        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        ----

        .. _ThresholdDependentBatchNorm2d.__init__-en:

        * **English**

        The Threshold-Dependent Batch Normalization (tdBN) proposed in
        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because tdBN needs to calculate statistics across time steps.

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float

        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 5 - 1  # [T * N, C, H, W]


class ThresholdDependentBatchNorm3d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <ThresholdDependentBatchNorm3d.__init__-cn>` | :ref:`English <ThresholdDependentBatchNorm3d.__init__-en>`

        ----

        .. _ThresholdDependentBatchNorm3d.__init__-cn:

        * **中文**

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_
        一文提出的 Threshold-Dependent Batch Normalization (tdBN)。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， tdBN 需要跨时间步求统计量。

        :param alpha: 由网络结构决定的超参数
        :type alpha: float

        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        ----

        .. _ThresholdDependentBatchNorm3d.__init__-en:

        * **English**

        The Threshold-Dependent Batch Normalization (tdBN) proposed in
        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because tdBN needs to calculate statistics across time steps.

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float

        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 6 - 1  # [T * N, C, H, W, D]


class _TemporalEffectiveBatchNormBase(_BatchNorm, base.MultiStepModule):

    def __init__(self, T: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = "m"
        self.scale = nn.Parameter(torch.ones([T]))

    def forward(self, x_seq: torch.Tensor):
        y_seq = functional.seq_to_ann_forward(x_seq, super().forward)
        return y_seq * self.scale.view(-1, *[1 for _ in range(y_seq.dim() - 1)])


class TemporalEffectiveBatchNorm1d(_TemporalEffectiveBatchNormBase):

    def __init__(self, T: int, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <TemporalEffectiveBatchNorm1d-cn>` | :ref:`English <TemporalEffectiveBatchNorm1d-en>`

        ----

        .. _TemporalEffectiveBatchNorm1d-cn:

        * **中文**

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_
        一文提出的 Temporal Effective Batch Normalization (TEBN)。

        TEBN在多步模式的BN的基础上，给每个时刻的输出增加一个可学习的缩放。
        若多步模式BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，
        其中 ``k[t]`` 是可学习的参数。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， TEBN 需要跨时间步求统计量。

        :param T: 总时间步数
        :type T: int

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        ----

        .. _TemporalEffectiveBatchNorm1d-en:

        * **English**

        Temporal Effective Batch Normalization (TEBN) proposed by
        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native multi-step BN.
        Denote the output at time step ``t`` of the native multi-step BN as ``y[t]``,
        then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because TEBN needs to calculate statistics across time steps.

        :param T: the number of time-steps
        :type T: int

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.
        """
        super().__init__(T, *args, **kwargs)

    def _check_input_dim(self, input):
        if input.ndim not in [3, 2]:
            raise ValueError(
                f"expect input shape [T*N, C, L] or [T*N, C], but get {input.shape}"
            )


class TemporalEffectiveBatchNorm2d(_TemporalEffectiveBatchNormBase):

    def __init__(self, T: int, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <TemporalEffectiveBatchNorm2d-cn>` | :ref:`English <TemporalEffectiveBatchNorm2d-en>`

        ----

        .. _TemporalEffectiveBatchNorm2d-cn:

        * **中文**

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_
        一文提出的 Temporal Effective Batch Normalization (TEBN)。

        TEBN在多步模式的BN的基础上，给每个时刻的输出增加一个可学习的缩放。
        若多步模式BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，
        其中 ``k[t]`` 是可学习的参数。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， TEBN 需要跨时间步求统计量。

        :param T: 总时间步数
        :type T: int

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        ----

        .. _TemporalEffectiveBatchNorm2d-en:

        * **English**

        Temporal Effective Batch Normalization (TEBN) proposed by
        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native multi-step BN.
        Denote the output at time step ``t`` of the native multi-step BN as ``y[t]``,
        then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because tdBN needs to calculate statistics across time steps.

        :param T: the number of time-steps
        :type T: int

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.
        """
        super().__init__(T, *args, **kwargs)

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError(
                f"expect input shape [T*N, C, H, W], but get {input.shape}"
            )


class TemporalEffectiveBatchNorm3d(_TemporalEffectiveBatchNormBase):

    def __init__(self, T: int, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <TemporalEffectiveBatchNorm3d-cn>` | :ref:`English <TemporalEffectiveBatchNorm3d-en>`

        ----

        .. _TemporalEffectiveBatchNorm3d-cn:

        * **中文**

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_
        一文提出的 Temporal Effective Batch Normalization (TEBN)。

        TEBN在多步模式的BN的基础上，给每个时刻的输出增加一个可学习的缩放。
        若多步模式BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，
        其中 ``k[t]`` 是可学习的参数。

        .. warning::

            只支持多步运行模式 ``step_mode = "m"`` 。这是因为， TEBN 需要跨时间步求统计量。

        :param T: 总时间步数
        :type T: int

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        ----

        .. _TemporalEffectiveBatchNorm3d-en:

        * **English**

        Temporal Effective Batch Normalization (TEBN) proposed by
        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native multi-step BN.
        Denote the output at time step ``t`` of the native multi-step BN as ``y[t]``,
        then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        .. warning::

            Only supports multi-step running mode ``step_mode = "m"`` .
            This is because TEBN needs to calculate statistics across time steps.

        :param T: the number of time-steps
        :type T: int

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.
        """
        super().__init__(T, *args, **kwargs)

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError(
                f"expect input shape [T*N, C, H, W], but get {input.shape}"
            )


class _BatchNormThroughTimeBase(base.MemoryModule):

    bn_type = _BatchNorm

    def __init__(
        self,
        T: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode: str = 's'
    ):
        super().__init__()
        self.bn_list = nn.ModuleList(
            self.bn_type(num_features, eps, momentum, affine, track_running_stats)
            for _ in range(T)
        )
        for bn in self.bn_list:
            bn.bias = None
        self.T = T
        self.step_mode = step_mode
        self.register_memory("t", -1)

    def single_step_forward(self, x: torch.Tensor):
        self.t = self.t + 1
        print(f"Call bn_list[{self.t}]")
        return self.bn_list[self.t](x)


class BatchNormThroughTime1d(_BatchNormThroughTimeBase):

    bn_type = nn.BatchNorm1d

    def __init__(
        self,
        T: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode: str = 's'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNormThroughTime1d-cn>` | :ref:`English <BatchNormThroughTime1d-en>`

        ----

        .. _BatchNormThroughTime1d-cn:

        * **中文**

        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_
        一文提出的 Batch Normalization Through Time (BNTT)。

        BNTT为每个时间步配置一个单独的单步BN。这 ``T`` 个BN的可学习参数以及统计量都相互独立。

        .. note::

            BNTT 能以单步或多步模式运行，其状态 ``t`` 标注了当前时间步。
            每次调用 ``single_step_forward()`` (包括通过 ``multi_step_forward()``
            间接调用的情况)， ``t`` 将加1。 ``t`` 将被用来索引对应时间步的BN。

            因此，记得在完成 ``T`` 个时间步的计算后，调用 ``reset()`` 来重制 ``t`` 。

        :param T: 总时间步数
        :type T: int

        :param step_mode: 运行模式，'s'代表单步模式，'m'代表多步模式
        :type step_mode: str

        其余参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        ----

        .. _BatchNormThroughTime1d-en:

        * **English**

        Batch Normalization Through Time (BNTT) proposed by
        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_ .

        BPTT assigns a separate single-step BN to each of the ``T`` time steps.
        The learnable parameters and statistics of these BNs are independent of each other.

        .. note::

            BNTT can run in single-step or multi-step mode, and its state ``t``
            marks the current time step. Every time you call ``single_step_forward()``
            (including indirect calling through ``multi_step_forward()``), ``t``
            will be incremented by 1. ``t`` will be used to index the BN
            corresponding to the current time step.

            Therefore, remember to call ``reset()`` method after completing ``T`` time
            steps so as to reset ``t`` .

        :param T: the number of time-steps
        :type T: int

        :param step_mode: running mode. 's' for single-step mode, 'm' for multi-step mode
        :type step_mode: str

        Other parameters are same with those of :class:`torch.nn.BatchNorm1d`.
        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)


class BatchNormThroughTime2d(_BatchNormThroughTimeBase):

    bn_type = nn.BatchNorm2d

    def __init__(
        self,
        T: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode: str = 's'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNormThroughTime2d-cn>` | :ref:`English <BatchNormThroughTime2d-en>`

        ----

        .. _BatchNormThroughTime2d-cn:

        * **中文**

        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_
        一文提出的 Batch Normalization Through Time (BNTT)。

        BNTT为每个时间步配置一个单独的单步BN。这 ``T`` 个BN的可学习参数以及统计量都相互独立。

        .. note::

            BNTT 能以单步或多步模式运行，其状态 ``t`` 标注了当前时间步。
            每次调用 ``single_step_forward()`` (包括通过 ``multi_step_forward()``
            间接调用的情况)， ``t`` 将加1。 ``t`` 将被用来索引对应时间步的BN。

            因此，记得在完成 ``T`` 个时间步的计算后，调用 ``reset()`` 来重制 ``t`` 。

        :param T: 总时间步数
        :type T: int

        :param step_mode: 运行模式，'s'代表单步模式，'m'代表多步模式
        :type step_mode: str

        其余参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        ----

        .. _BatchNormThroughTime2d-en:

        * **English**

        Batch Normalization Through Time (BNTT) proposed by
        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_ .

        BPTT assigns a separate single-step BN to each of the ``T`` time steps.
        The learnable parameters and statistics of these BNs are independent of each other.

        .. note::

            BNTT can run in single-step or multi-step mode, and its state ``t``
            marks the current time step. Every time you call ``single_step_forward()``
            (including indirect calling through ``multi_step_forward()``), ``t``
            will be incremented by 1. ``t`` will be used to index the BN
            corresponding to the current time step.

            Therefore, remember to call ``reset()`` method after completing ``T`` time
            steps so as to reset ``t`` .

        :param T: the number of time-steps
        :type T: int

        :param step_mode: running mode. 's' for single-step mode, 'm' for multi-step mode
        :type step_mode: str

        Other parameters are same with those of :class:`torch.nn.BatchNorm2d`.
        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)


class BatchNormThroughTime3d(_BatchNormThroughTimeBase):

    bn_type = nn.BatchNorm3d

    def __init__(
        self,
        T: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        step_mode: str = 's'
    ):
        """
        **API Language:**
        :ref:`中文 <BatchNormThroughTime3d-cn>` | :ref:`English <BatchNormThroughTime3d-en>`

        ----

        .. _BatchNormThroughTime3d-cn:

        * **中文**

        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_
        一文提出的 Batch Normalization Through Time (BNTT)。

        BNTT为每个时间步配置一个单独的单步BN。这 ``T`` 个BN的可学习参数以及统计量都相互独立。

        .. note::

            BNTT 能以单步或多步模式运行，其状态 ``t`` 标注了当前时间步。
            每次调用 ``single_step_forward()`` (包括通过 ``multi_step_forward()``
            间接调用的情况)， ``t`` 将加1。 ``t`` 将被用来索引对应时间步的BN。

            因此，记得在完成 ``T`` 个时间步的计算后，调用 ``reset()`` 来重制 ``t`` 。

        :param T: 总时间步数
        :type T: int

        :param step_mode: 运行模式，'s'代表单步模式，'m'代表多步模式
        :type step_mode: str

        其余参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        ----

        .. _BatchNormThroughTime3d-en:

        * **English**

        Batch Normalization Through Time (BNTT) proposed by
        `Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full>`_ .

        BPTT assigns a separate single-step BN to each of the ``T`` time steps.
        The learnable parameters and statistics of these BNs are independent of each other.

        .. note::

            BNTT can run in single-step or multi-step mode, and its state ``t``
            marks the current time step. Every time you call ``single_step_forward()``
            (including indirect calling through ``multi_step_forward()``), ``t``
            will be incremented by 1. ``t`` will be used to index the BN
            corresponding to the current time step.

            Therefore, remember to call ``reset()`` method after completing ``T`` time
            steps so as to reset ``t`` .

        :param T: the number of time-steps
        :type T: int

        :param step_mode: running mode. 's' for single-step mode, 'm' for multi-step mode
        :type step_mode: str

        Other parameters are same with those of :class:`torch.nn.BatchNorm3d`.
        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)