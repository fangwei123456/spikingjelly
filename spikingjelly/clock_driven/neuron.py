from abc import abstractmethod
import torch
import torch.nn as nn
from . import surrogate, base
import math

class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold=1., v_reset=0., surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param detach_reset: whether detach the computation graph of reset

        This class is the base class of differentiable spiking neurons.
        """
        super().__init__()
        self.register_buffer('v_threshold', torch.as_tensor(v_threshold))
        if v_reset is None:
            self.register_buffer('v_reset', None)
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_buffer('v_reset', torch.as_tensor(v_reset))
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

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
            if self.v_threshold == 1.:
                self.v = self.v - spike
            else:
                self.v = self.v - spike * self.v_threshold
        else:
            if self.v_reset == 0.:
                self.v = (1. - spike) * self.v
            else:
                self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param x: 输入到神经元的电压增量

        :return: 神经元的输出脉冲

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param x: increment of voltage inputted to neurons

        :return: out spikes of neurons

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class IFNode(BaseNode):
    def __init__(self, v_threshold=1., v_reset=0., surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        """
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + X[t]

        * :ref:`中文API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + X[t]
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v += x

class LIFNode(BaseNode):
    def __init__(self, tau=100., v_threshold=1., v_reset=0., surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数。``tau`` 对于这一层的所有神经元都是共享的

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离


        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset})

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant. ``tau`` is shared by all neurons in this layer

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset})
        """
        assert tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_buffer('tau', torch.as_tensor(tau))

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v += (x - self.v) / self.tau
        else:
            self.v += (x - (self.v - self.v_reset)) / self.tau

class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1., v_reset=0., surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数。``tau`` 对于这一层的所有神经元都是共享的

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`
 提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset})

        其中 :math:`\frac{1}{\tau} = {\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant. ``tau`` is shared by all neurons in this layer

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>` and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset})

        where :math:`\frac{1}{\tau} = {\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        assert init_tau > 1.
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))


    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v += (x - self.v) * self.w.sigmoid()
        else:
            self.v += (x - (self.v - self.v_reset)) * self.w.sigmoid()

