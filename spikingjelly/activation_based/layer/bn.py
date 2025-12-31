import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from .. import base, functional


class BatchNorm1d(nn.BatchNorm1d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm1d-en>`

        .. _BatchNorm1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm1d`

        * :ref:`中文 API <BatchNorm1d-cn>`

        .. _BatchNorm1d-en:

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
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm2d-en>`

        .. _BatchNorm2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm2d`

        * :ref:`中文 API <BatchNorm2d-cn>`

        .. _BatchNorm2d-en:

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
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm3d-en>`

        .. _BatchNorm3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm3d`

        * :ref:`中文 API <BatchNorm3d-cn>`

        .. _BatchNorm3d-en:

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
    def __init__(self, in_channels, height, width, k=0.9, shared_across_channels=False, step_mode='s'):
        """
        * :ref:`API in English <NeuNorm.__init__-en>`

        .. _NeuNorm.__init__-cn:

        :param in_channels: 输入数据的通道数

        :param height: 输入数据的宽

        :param width: 输入数据的高

        :param k: 动量项系数

        :param shared_across_channels: 可学习的权重 ``w`` 是否在通道这一维度上共享。设置为 ``True`` 可以大幅度节省内存

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_ 中提出\\
        的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如：

        ``Conv2d -> LIF -> NeuNorm``

        要求输入的尺寸是 ``[batch_size, in_channels, height, width]``。

        ``in_channels`` 是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        ``k`` 是动量项系数，相当于论文中的 :math:`k_{\\tau 2}`。

        论文中的 :math:`\\frac{v}{F}` 会根据 :math:`k_{\\tau 2} + vF = 1` 自动算出。

        * :ref:`中文API <NeuNorm.__init__-cn>`

        .. _NeuNorm.__init__-en:

        :param in_channels: channels of input

        :param height: height of input

        :param width: height of width

        :param k: momentum factor

        :param shared_across_channels: whether the learnable parameter ``w`` is shared over channel dim. If set ``True``,
            the consumption of memory can decrease largely

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        The NeuNorm layer is proposed in `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_.

        It should be placed after spiking neurons behind convolution layer, e.g.,

        ``Conv2d -> LIF -> NeuNorm``

        The input should be a 4-D tensor with ``shape = [batch_size, in_channels, height, width]``.

        ``in_channels`` is the channels of input，which is :math:`F` in the paper.

        ``k`` is the momentum factor，which is :math:`k_{\\tau 2}` in the paper.

        :math:`\\frac{v}{F}` will be calculated by :math:`k_{\\tau 2} + vF = 1` autonomously.

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
        self.x = self.k0 * self.x + self.k1 * in_spikes.sum(dim=1,
                                                            keepdim=True)  # x.shape = [batch_size, 1, height, width]
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
        assert self.step_mode == 'm', "ThresholdDependentBatchNormBase can only be used in the multi-step mode!"
        return functional.seq_to_ann_forward(x_seq, super().forward)


class ThresholdDependentBatchNorm1d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm1d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm1d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 4 - 1 or input.dim() == 3 - 1  # [T * N, C, L]


class ThresholdDependentBatchNorm2d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm2d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm2d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 5 - 1  # [T * N, C, H, W]


class ThresholdDependentBatchNorm3d(_ThresholdDependentBatchNormBase):

    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm3d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm3d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 6 - 1  # [T * N, C, H, W, D]


class _TemporalEffectiveBatchNormBase(base.MemoryModule):

    bn_instance = _BatchNorm

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        super().__init__()

        self.bn = self.bn_instance(num_features, eps, momentum, affine, track_running_stats, step_mode)
        self.scale = nn.Parameter(torch.ones([T]))
        self.register_memory('t', 0)

    def single_step_forward(self, x: torch.Tensor):
        return self.bn(x) * self.scale[self.t]


class TemporalEffectiveBatchNorm1d(_TemporalEffectiveBatchNormBase):

    bn_instance = BatchNorm1d

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm1d-en>`

        .. _TemporalEffectiveBatchNorm1d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm1d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm1d-cn>`

        .. _TemporalEffectiveBatchNorm1d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm1d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, L]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1)


class TemporalEffectiveBatchNorm2d(_TemporalEffectiveBatchNormBase):

    bn_instance = BatchNorm2d

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm2d-en>`

        .. _TemporalEffectiveBatchNorm2d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm2d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm2d-cn>`

        .. _TemporalEffectiveBatchNorm2d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm2d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, H, W]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1)


class TemporalEffectiveBatchNorm3d(_TemporalEffectiveBatchNormBase):

    bn_instance = BatchNorm3d

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm3d-en>`

        .. _TemporalEffectiveBatchNorm3d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm3d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm3d-cn>`

        .. _TemporalEffectiveBatchNorm3d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm3d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, H, W, D]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1, 1)

