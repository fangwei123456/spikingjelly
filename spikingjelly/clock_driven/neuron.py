from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from . import surrogate, base
import math
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike


class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + X[t]

        * :ref:`中文API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + X[t]
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class MultiStepIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepIFNode.__init__-en>`

        .. _MultiStepIFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.IFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepIFNode.__init__-cn>`

        .. _MultiStepIFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.IFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'

        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)


            self.spike_seq, self.v_seq = neuron_kernel.MultiStepIFNodePTT.apply(
                x_seq, self.v, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})
        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class MultiStepLIFNode(LIFNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepLIFNode.__init__-en>`

        .. _MultiStepLIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.LIFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepLIFNode.__init__-cn>`

        .. _MultiStepLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.LIFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(tau, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)


            self.spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq, self.v, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError


class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <ParametricLIFNode.__init__-en>`

        .. _ParametricLIFNode.__init__-cn:

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        * :ref:`中文API <ParametricLIFNode.__init__-cn>`

        .. _ParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :param init_tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        """

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()


class MultiStepParametricLIFNode(ParametricLIFNode):
    def __init__(self, init_tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepParametricLIFNode.__init__-en>`

        .. _MultiStepParametricLIFNode.__init__-cn:

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        多步版本的 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

            .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepParametricLIFNode.__init__-cn>`

        .. _MultiStepParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :param init_tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.
        """
        super().__init__(init_tau, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)


            self.spike_seq, self.v_seq = neuron_kernel.MultiStepParametricLIFNodePTT.apply(
                x_seq, self.v, self.w.sigmoid(), self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError

class QIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <QIFNode.__init__-en>`

        .. _QIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_c: 关键电压
        :type v_c: float

        :param a0: 
        :type a0: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Quadratic Integrate-and-Fire 神经元模型，一种非线性积分发放神经元模型，也是指数积分发放神经元(Exponential Integrate-and-Fire)的近似版本。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))

        * :ref:`中文API <QIFNode.__init__-cn>`

        .. _QIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_c: critical voltage
        :type v_c: float

        :param a0: 
        :type a0: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Quadratic Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an approximation of the Exponential Integrate-and-Fire model.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))
        """
                 
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau


class EIFNode(BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <EIFNode.__init__-en>`

        .. _EIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param delta_T: 陡峭度参数
        :type delta_T: float

        :param theta_rh: 基强度电压阈值
        :type theta_rh: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Exponential Integrate-and-Fire 神经元模型，一种非线性积分发放神经元模型，是由HH神经元模型(Hodgkin-Huxley model)简化后推导出的一维模型。在 :math:`\\Delta_T\\to 0` 时退化为LIF模型。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)

        * :ref:`中文API <EIFNode.__init__-cn>`

        .. _EIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param delta_T: sharpness parameter
        :type delta_T: float

        :param theta_rh: rheobase threshold
        :type theta_rh: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Exponential Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an one-dimensional model derived from the Hodgkin-Huxley model. It degenerates to the LIF model when :math:`\\Delta_T\\to 0`
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)
        """
                 
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def neuronal_charge(self, x: torch.Tensor):
        
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)
        
        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp((self.v - self.theta_rh) / self.delta_T)) / self.tau