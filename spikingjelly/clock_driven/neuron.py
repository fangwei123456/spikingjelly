import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate, accelerating, layer
import math

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
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
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
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        This class is the base class of differentiable spiking neurons.
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        self.surrogate_function = surrogate_function
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

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
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def spiking(self):
        '''
        * :ref:`API in English <BaseNode.spiking-en>`

        .. _BaseNode.spiking-cn:

        :return: 神经元的输出脉冲

        根据当前神经元的电压、阈值、重置电压，计算输出脉冲，并更新神经元的电压。

        * :ref:`中文API <BaseNode.spiking-cn>`

        .. _BaseNode.spiking-en:

        :return: out spikes of neurons

        Calculate out spikes of neurons and update neurons' voltage by their current voltage, threshold voltage and reset voltage.

        '''
        
        spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            if self.monitor['v'].__len__() == 0:
                # 补充在0时刻的电压
                if self.v_reset is None:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * 0)
                else:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * self.v_reset)

            self.monitor['v'].append(self.v.data.cpu().numpy().copy())
            self.monitor['s'].append(spike.data.cpu().numpy().copy())

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
            
        if self.v_reset is None:
            if self.surrogate_function.spiking:
                self.v = accelerating.soft_voltage_transform(self.v, spike_d, self.v_threshold)
            else:
                self.v = self.v - spike_d * self.v_threshold
        else:
            if self.surrogate_function.spiking:
                self.v = accelerating.hard_voltage_transform(self.v, spike_d, self.v_reset)
            else:
                self.v = self.v * (1 - spike_d) + self.v_reset * spike_d

        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())

        return spike

    def forward(self, dv: torch.Tensor):
        '''

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param dv: 输入到神经元的电压增量

        :return: 神经元的输出脉冲

        子类需要实现这一函数。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param dv: increment of voltage inputted to neurons

        :return: out spikes of neurons

        Subclass should implement this function.

        '''
        raise NotImplementedError

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
            self.v = 0
        else:
            self.v = self.v_reset
        if self.monitor:
            self.monitor = {'v': [], 's': []}


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
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
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
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()

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
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
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
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
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

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) / self.tau
        else:
            self.v += (dv - (self.v - self.v_reset)) / self.tau
        return self.spiking()

class PLIFNode(BaseNode):
    @staticmethod
    def piecewise_exp(w: torch.Tensor):
        if w.item() >= 0:
            return 1 - (- w).exp() / 2
        else:
            return w.exp() / 2

    @staticmethod
    def inverse_piecewise_exp(init_tau: float):
        if init_tau > 2:
            return math.log(2 / init_tau)
        elif init_tau < 2:
            return math.log(init_tau / (2 * init_tau - 2))
        else:
            return 0.0

    @staticmethod
    def sigmoid(w: torch.Tensor):
        return w.sigmoid()

    @staticmethod
    def inverse_sigmoid(init_tau: float):
        return - math.log(init_tau - 1)

    @staticmethod
    def reciprocal_abs_plus_1(w: torch.Tensor):
        return 1 / (1 + w.abs())

    @staticmethod
    def inverse_reciprocal_abs_plus_1(init_tau: float):
        return init_tau - 1


    def __init__(self, init_tau=2.0, clamp=False, clamp_function=None, inverse_clamp_function=None, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False,
                 monitor_state=False):
        '''
        * :ref:`API in English <PLIFNode.__init__-en>`

        .. _PLIFNode.__init__-cn:

        :param init_tau: 初始的 ``tau``

        :param clamp: 本层神经元中可学习的参数为``w``,当 ``clamp == False`` 时，``self.v`` 的更新按照 ``self.v += (dv - (self.v - self.v_reset)) * self.w``；
            当 ``clamp == True`` 时，``self.v`` 的更新按照 ``self.v += (dv - (self.v - self.v_reset)) * clamp_function(self.w)``，
            且 ``self.w`` 的初始值为 ``inverse_clamp_function(init_tau)``

        :param clamp_function: 通常是取值 ``(0,1)`` 的一个函数，当 ``clamp == True``，在前向传播时，``tau = 1 / clamp_function(self.w)``。

        :param inverse_clamp_function: ``clamp_function`` 的反函数。这个参数只在 ``clamp == True`` 时生效

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        `Leaky integrate-and-fire spiking neuron with learnable membrane time parameter <https://arxiv.org/abs/2007.05785>`_ 提出的Parametric
        LIF神经元模型，时间常数 ``tau`` 可学习的LIF神经元。其阈下神经动力学方程与LIF神经元相同：

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        需要注意的是，对于同一层神经元，它们的 ``tau`` 是共享的。

        .. tip::
            LIF神经元的电压更新代码为

            ``self.v += (dv - (self.v - self.v_reset)) / self.tau``

            为了防止出现除以0的情况，PLIF神经元没有使用除法，而是用乘法代替（``clamp == False`` 时）：

            ``self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))``

            ``self.v += (dv - (self.v - self.v_reset)) * self.w``

        * :ref:`中文API <PLIFNode.__init__-cn>`

        .. _PLIFNode.__init__-en:

        :param init_tau: initial value of ``tau``

        :param clamp: the learnable parameter is ``w`. When ``clamp == False``, the update of ``self.v`` is ``self.v += (dv - (self.v - self.v_reset)) * self.w``;
            when ``clamp == True``, the update of ``self.v`` is ``self.v += (dv - (self.v - self.v_reset)) * clamp_function(self.w)``,
            and the initial value of ``self.w`` is ``inverse_clamp_function(init_tau)``

        :param clamp_function: can be a function range ``(0,1)``. When ``clamp == True``, ``tau = 1 / clamp_function(self.w)``
            during forward.

        :param inverse_clamp_function: inverse function of ``clamp_function``. This param only takes effect when ``clamp == True``

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Parametric LIF neuron that is proposed in `Leaky integrate-and-fire spiking neuron with learnable membrane time parameter <https://arxiv.org/abs/2007.05785>`_.
        The membrane time constant ``tau`` of PLIF neuron is learnable. The subthreshold neural dynamics of the PLIF neuron
        is same with that of the LIF neuron:

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        Note that ``tau`` is shared by all neurons in this layer.

        .. tip::
            The code of voltage update is as followed:

            ``self.v += (dv - (self.v - self.v_reset)) / self.tau``

            To avoid division by zero, the code for the PLIF neuron uses multiplication substitute for division (when
            ``clamp == False``):

            ``self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))``

            ``self.v += (dv - (self.v - self.v_reset)) * self.w``
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.clamp = clamp
        if self.clamp:
            self.clamp_function = clamp_function
            init_w = inverse_clamp_function(init_tau)
            self.w = nn.Parameter(torch.tensor([init_w], dtype=torch.float))
            assert abs(self.tau() - init_tau) < 1e-4, print('tau:', self.tau(), 'init_tau', init_tau)

        else:
            self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.clamp:
            self.v += (dv - (self.v - self.v_reset)) * self.clamp_function(self.w)
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w
        return self.spiking()

    def tau(self):
        if self.clamp:
            return 1 / self.clamp_function(self.w.data).item()
        else:
            return 1 / self.w.data.item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}, clamp={self.clamp}'

class RIFNode(BaseNode):
    def __init__(self, init_w=-1e-3, amplitude=None, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
        '''
        * :ref:`API in English <RIFNode.__init__-en>`

        .. _RIFNode.__init__-cn:

        :param init_w: 初始的自连接权重

        :param amplitude: 对自连接权重的限制。若为 ``None``，则不会对权重有任何限制；
            若为一个 ``float``，会限制权重在 ``(- amplitude, amplitude)`` 范围内；
            若为一个 ``tuple``，会限制权重在 ``(amplitude[0], amplitude[1])`` 范围内。
            权重的限制是通过套上sigmoid函数进行限幅，然后进行线性变换来实现。

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        Recurrent Integrate-and-Fire 神经元模型。与Parametric LIF神经元模型类似，但有微妙的区别，自连接权重不会作用于输入。其阈下神经动力学方程为：

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = w(V(t) - V_{reset}) + R_{m}I(t)

        其中 :math:`w` 是自连接权重，权重是可以学习的。对于同一层神经元，它们的 :math:`w` 是共享的。

        * :ref:`中文API <RIFNode.__init__-cn>`

        .. _RIFNode.__init__-en:

        :param init_w: initial self connection weight

        :param amplitude: Restriction on self connection weight. If ``None``, there won't be any restriction on weight;
            if ``amplitude`` is a ``float``, the weight will be restricted in ``(- amplitude, amplitude)``;
            if ``amplitude`` is a ``tuple``, the weight will be restricted in ``(amplitude[0], amplitude[1])``.
            This restriction is implemented by a sigmoid function and a linear transform.

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Recurrent Integrate-and-Fire neuron. It is very similar with the Parametric LIF neuron. But there is a tricky
        difference that the self connection will not apply to input. The subthreshold neural dynamics of the PLIF neuron

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = w(V(t) - V_{reset}) + R_{m}I(t)

        :math:`w` is the self connection weight. The weight is learnable. And it is shared by all neurons in this layer.
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.amplitude = amplitude
        if isinstance(self.amplitude, int):
            self.amplitude = float(self.amplitude)

        if self.amplitude is None:
            self.g = nn.Parameter(torch.tensor([init_w], dtype=torch.float))
        elif isinstance(self.amplitude, float):
            self.g = math.log((amplitude + init_w) / (amplitude - init_w))
            self.g = nn.Parameter(torch.tensor([self.g], dtype=torch.float))
            # (self.w.sigmoid() * 2 - 1 ) * self.amplitude == init_w
        else:
            self.g = math.log((init_w - amplitude[0]) / (amplitude[1] - init_w))
            self.g = nn.Parameter(torch.tensor([self.g], dtype=torch.float))
            # self.w.sigmoid() * (self.amplitude[1] - self.amplitude[0]) + self.amplitude[0] == init_w

    def w(self):
        '''
        * :ref:`API in English <RIFNode.w-en>`

        .. _RIFNode.w-cn:

        :return: 自连接权重

        * :ref:`中文API <RIFNode.w-cn>`

        .. _RIFNode.w-en:

        :return: the self connection weight
        '''
        if self.amplitude is None:
            return self.g.data
        elif isinstance(self.amplitude, float):
            return (self.g.data.sigmoid() * 2 - 1) * self.amplitude
        else:
            return self.g.data.sigmoid() * (self.amplitude[1] - self.amplitude[0]) + self.amplitude[0]

    def extra_repr(self):

        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, w={self.w()}'

    def forward(self, dv: torch.Tensor):
        if self.amplitude is None:
            self.v += (self.v - self.v_reset) * self.g + dv
        elif isinstance(self.amplitude, float):
            self.v += (self.v - self.v_reset) * ((self.g.sigmoid() * 2 - 1) * self.amplitude) + dv
        else:
            self.v += (self.v - self.v_reset) * \
                     (self.g.sigmoid() * (self.amplitude[1] - self.amplitude[0]) + self.amplitude[0]) * self.amplitude + dv


        return self.spiking()

class AdaptThresholdNode(nn.Module):
    def __init__(self, neuron_shape, tau_m: float, tau_adp: float, v_threshold_baseline=1.0, v_threshold_range=1.8, v_reset=0.0, surrogate_function=surrogate.Erf(), monitor_state=False, dt=1.0):
        '''
        * :ref:`API in English <AdaptThresholdNode.__init__-en>`

        .. _AdaptThresholdNode.__init__-cn:

        :param neuron_shape: 神经元张量的形状
        :type neuron_shape: array_like
        :param tau_m: 膜电位时间常数
        :type tau_m: float
        :param tau_adp: 阈值时间常数
        :type tau_adp: float
        :param v_threshold_baseline: 最小阈值，也为初始阈值 :math:`b_0` ，默认为1.0
        :type v_threshold_baseline: float 
        :param v_threshold_range: 决定阈值变化范围的参数 :math:`\\beta` ，默认为1.8。控制阈值的范围为 :math:`[b_0,b_0+\\beta]`
        :type v_threshold_range: float
        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；如果设置为 ``None``，则电压会被减去 ``v_threshold``，默认为0.0
        :type v_reset: float
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param detach_reset: 是否将reset过程的计算图分离，默认为surrogate.Erf()
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。还需要注意，``self.reset()`` 函数会清空这些链表， 默认为False
        :type monitor_state: bool
        :param dt: 神经元的仿真间隔时间参数, 默认为1.0
        :type dt: float

        `Effective and Efficient Computation with Multiple-timescale Spiking Recurrent Neural Networks <https://arxiv.org/abs/2005.11633>`_ 中提出的自适应阈值神经元模型。在LIF神经元的基础上增加了一个阈值的动态方程：

        .. math::

            \\begin{align}
            \\eta_t&=\\rho\\eta_{t-1}+(1-\\rho)S_{t-1},\\\\
            \\theta_t&=b_0+\\beta\\eta_t,
            \\end{align}
        
        其中 :math:`\\eta_t` 为t时刻的阈值增幅，:math:`\\rho` 为阈值动态方程中由 ``tau_adp`` 决定的时间常数。:math:`\\theta_t` 为t时刻的电压阈值。

        所有神经元动态方程的时间常数均为\ **可学习**\ 的网络参数。

        .. hint::
            不同于该模块中的其它神经元层，同层的各神经元不共享时间常数。

        * :ref:`中文API <AdaptThresholdNode.__init__-cn>`

        .. _AdaptThresholdNode.__init__-en:

        :param neuron_shape: Shape of neuron tensor
        :type neuron_shape: array_like
        :param tau_m: Membrane potential time-constant
        :type tau_m: float
        :param tau_adp: Threshold time-constant
        :type tau_adp: float
        :param v_threshold_baseline: Minimal threshold, also the initial threshold :math:`b_0`, defaults to 1.0
        :type v_threshold_baseline: float
        :param v_threshold_range: Parameter :math:`\\beta` determining the range of threshold to :math:`[b_0,b_0+\\beta]` , defaults to 1.8
        :type v_threshold_range: float
        :param v_reset: Reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``, defaults to 0.0
        :type v_reset: float
        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param detach_reset: whether detach the computation graph of reset, defaults to surrogate.Erf()
        :param monitor_state: Whether to turn on the monitor, defaults to False
        :type monitor_state: bool
        :param dt: Simulation interval constant of neurons, defaults to 1.0
        :type dt: float

        An neuron model with adaptive threshold proposed in `Effective and Efficient Computation with Multiple-timescale Spiking Recurrent Neural Networks <https://arxiv.org/abs/2005.11633>`_. Compared to vanilla LIF neuron, an additional dynamic equation of threshold is added:

        .. math::

            \\begin{align}
            \\eta_t & = \\rho\\eta_{t-1}+(1-\\rho)S_{t-1},\\\\
            \\theta_t & = b_0+\\beta\\eta_t,
            \\end{align}
        
        where :math:`\\eta_t` is the growth of threshold at timestep t, :math:`\\rho` is the time-constant determined by ``tau_adp`` in threshold dynamic. :math:`\\theta_t` is the threshold at timestep t.

        All time constants in neurons' dynamics are **learnable** network parameters.

        .. admonition:: Hint
            :class: hint

            Different from other types of neuron in this module, time-constant is NOT shared in the same layer.
        '''

        super().__init__()
        self.neuron_shape = neuron_shape
        self.b_0 = v_threshold_baseline
        self.b = 0
        self.v_reset = v_reset
        self.beta = v_threshold_range
        self.tau_m = nn.Parameter(torch.full(neuron_shape, fill_value=tau_m, dtype=torch.float))
        self.tau_adp = nn.Parameter(torch.full(neuron_shape, fill_value=tau_adp, dtype=torch.float))
        self.dt = dt
        self.last_spike = torch.rand(neuron_shape)

        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        self.v_threshold = self.b_0
        
        self.surrogate_function = surrogate_function
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def extra_repr(self):
        return f'v_threshold_baseline={self.b_0}, v_threshold_range={self.beta}, v_reset={self.v_reset}'

    def set_monitor(self, monitor_state=True):
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def spiking(self):
        spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            if self.monitor['v'].__len__() == 0:
                # 补充在0时刻的电压
                if self.v_reset is None:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * 0)
                else:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * self.v_reset)

            self.monitor['v'].append(self.v.data.cpu().numpy().copy())
            self.monitor['s'].append(spike.data.cpu().numpy().copy())

        if self.v_reset is None:
            if self.surrogate_function.spiking:
                self.v = accelerating.soft_voltage_transform(self.v, spike, self.v_threshold)
            else:
                self.v = self.v - spike * self.v_threshold
        else:
            if self.surrogate_function.spiking:
                self.v = accelerating.hard_voltage_transform(self.v, spike, self.v_reset)
            else:
                self.v = self.v * (1 - spike) + self.v_reset * spike

        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())

        return spike

    def forward(self, dv: torch.Tensor):
        alpha = torch.exp(-self.dt / self.tau_m)
        rho = torch.exp(-self.dt / self.tau_adp)

        self.b = rho * self.b + (1 - rho) * self.last_spike
        self.v_threshold = self.b_0 + self.beta * self.b
        self.v = self.v * alpha + (1 - alpha) * dv

        spike = self.spiking()

        self.last_spike = spike

        return spike

    def reset(self):

        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        self.v_threshold = self.b_0
        self.b = 0
        self.last_spike = torch.rand(self.neuron_shape)
        if self.monitor:
            self.monitor = {'v': [], 's': []}
