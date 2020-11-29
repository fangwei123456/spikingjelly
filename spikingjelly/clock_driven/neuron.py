from abc import abstractmethod
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate

class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
        '''
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``h``, ``v`` ``s``，分别记录充电后的电压、释放脉冲后的电压、释放的脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param detach_reset: whether detach the computation graph of reset 
        
        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``h`` for recording membrane potential after charging,
            ``v`` for recording membrane potential after firing and ``s`` for recording output spikes.
            And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        This class is the base class of differentiable spiking neurons.
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.monitor = monitor_state
        self.reset()


    @abstractmethod
    def neuronal_charge(self, dv: torch.Tensor):
        '''
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        '''
        raise NotImplementedError

    def neuronal_fire(self):
        '''
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        '''
        if self.monitor:
            if self.monitor['h'].__len__() == 0:
                # 补充在0时刻的电压
                if self.v_reset is None:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy() * 0)
                else:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy() * self.v_reset)

        self.spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            self.monitor['s'].append(self.spike.data.cpu().numpy().copy())

    def neuronal_reset(self):
        '''
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        '''
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            self.v = (1 - spike) * self.v - spike * self.v_threshold
        else:
            self.v = (1 - spike) * self.v + spike * self.v_reset

        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())


    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def set_monitor(self, monitor_state=True):
        '''
        * :ref:`API in English <BaseNode.set_monitor-en>`

        .. _BaseNode.set_monitor-cn:

        :param monitor_state: ``True`` 或 ``False``，表示开启或关闭monitor

        :return: None

        设置开启或关闭monitor。

        * :ref:`中文API <BaseNode.set_monitor-cn>`

        .. _BaseNode.set_monitor-en:

        :param monitor_state: ``True`` or ``False``, which indicates turn on or turn off the monitor

        :return: None

        Turn on or turn off the monitor.
        '''
        if monitor_state:
            self.monitor = {'h': [], 'v': [], 's': []}
        else:
            self.monitor = False


    def forward(self, dv: torch.Tensor):
        '''

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param dv: 输入到神经元的电压增量

        :return: 神经元的输出脉冲

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param dv: increment of voltage inputted to neurons

        :return: out spikes of neurons

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        '''
        self.neuronal_charge(dv)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

    def reset(self):
        '''
        * :ref:`API in English <BaseNode.reset-en>`

        .. _BaseNode.reset-cn:

        :return: None

        重置神经元为初始状态，也就是将电压设置为 ``v_reset``。
        如果子类的神经元还含有其他状态变量，需要在此函数中将这些状态变量全部重置。

        * :ref:`中文API <BaseNode.reset-cn>`

        .. _BaseNode.reset-en:

        :return: None

        Reset neurons to initial states, which means that set voltage to ``v_reset``.
        Note that if the subclass has other stateful variables, these variables should be reset by this function.
        '''
        if self.v_reset is None:
            self.v = 0.0
        else:
            self.v = self.v_reset

        self.spike = None

        if self.monitor:
            self.monitor = {'h': [], 'v': [], 's': []}


class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
        '''
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``h``, ``v`` ``s``，分别记录充电后的电压、释放脉冲后的电压、释放的脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)

        * :ref:`中文API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``h`` for recording membrane potential after charging,
            ``v`` for recording membrane potential after firing and ``s`` for recording output spikes.
            And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)

    def neuronal_charge(self, dv: torch.Tensor):
        self.v += dv

class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False,
                 monitor_state=False):
        '''
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数。``tau`` 对于这一层的所有神经元都是共享的

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``h``, ``v`` ``s``，分别记录充电后的电压、释放脉冲后的电压、释放的脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant. ``tau`` is shared by all neurons in this layer


        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``h`` for recording membrane potential after charging,
            ``v`` for recording membrane potential after firing and ``s`` for recording output spikes.
            And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.tau = tau

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau}'

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) / self.tau
        else:
            self.v += (dv - (self.v - self.v_reset)) / self.tau