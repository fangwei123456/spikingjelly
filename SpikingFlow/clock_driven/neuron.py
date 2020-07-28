import torch
import torch.nn as nn
import torch.nn.functional as F
from SpikingFlow.clock_driven import surrogate, accelerating
import math

class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        '''
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If ``v_reset=None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If not ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        This class is the base class of differentiable spiking neurons.
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
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
        return 'v_threshold={}, v_reset={}'.format(
            self.v_threshold, self.v_reset
        )

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
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())
            self.monitor['s'].append(spike.data.cpu().numpy().copy())

        if self.v_reset is None:
            self.v = accelerating.soft_voltage_transform(self.v, spike, self.v_threshold)
        else:
            self.v = accelerating.hard_voltage_transform(self.v, spike, self.v_reset)

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
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        '''
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

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

        :param v_reset: reset voltage of neurons. If ``v_reset=None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If not ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, monitor_state)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()


class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(),
                 monitor_state=False):
        '''
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数。``tau`` 对于这一层的所有神经元都是共享的

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

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

        :param v_reset: reset voltage of neurons. If ``v_reset=None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If not ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, monitor_state)
        self.tau = tau

    def extra_repr(self):
        return 'v_threshold={}, v_reset={}, tau={}'.format(
            self.v_threshold, self.v_reset, self.tau
        )
    def forward(self, dv: torch.Tensor):
        self.v += (dv - (self.v - self.v_reset)) / self.tau
        return self.spiking()

class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, decay=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        '''
        * :ref:`API in English <PLIFNode.__init__-en>`

        .. _PLIFNode.__init__-cn:

        :param init_tau: 初始的 ``tau``

        :param decay: 为 ``True`` 时会限制 ``tau`` 的取值恒大于1，使得神经元不会给自身充电；为 ``False`` 时不会有任何限制

        :param v_threshold: 神经元的阈值电压

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数

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

            为了防止出现除以0的情况，PLIF神经元没有使用除法，而是用乘法代替：

            ``self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))``

            ``self.v += (dv - (self.v - self.v_reset)) * self.w``

        * :ref:`中文API <PLIFNode.__init__-cn>`

        .. _PLIFNode.__init__-en:

        :param init_tau: initial value of ``tau``

        :param decay: If ``True``, ``tau`` will be restricted to larger than 1, making sure that neurons will not charge themselves.
            If ``False``, there won't be any restriction on ``tau``.

        :param v_threshold: threshold voltage of neurons

        :param v_reset: reset voltage of neurons. If ``v_reset=None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If not ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

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

            To avoid division by zero, the code for the PLIF neuron uses multiplication substitute for division:

            ``self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))``

            ``self.v += (dv - (self.v - self.v_reset)) * self.w``
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, monitor_state)
        self.decay = decay
        if self.decay:
            self.w = nn.Parameter(torch.tensor([math.log(1 / (init_tau - 1))], dtype=torch.float))
            # self.w.sigmoid() == init_tau
        else:
            self.w = nn.Parameter(1 / torch.tensor([init_tau], dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.decay:
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w
        return self.spiking()

    def tau(self):
        if self.decay:
            return 1 / self.w.data.sigmoid().item()
        else:
            return 1 / self.w.data.item()

    def extra_repr(self):
        return 'v_threshold={}, v_reset={}, tau={}, decay={}'.format(
            self.v_threshold, self.v_reset, self.tau(), self.decay
        )

class RIFNode(BaseNode):
    def __init__(self, init_w=-1e-3, amplitude=None, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
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

        :param v_reset: reset voltage of neurons. If ``v_reset=None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If not ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation

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
        super().__init__(v_threshold, v_reset, surrogate_function, monitor_state)
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

        return 'v_threshold={}, v_reset={}, w={}'.format(
            self.v_threshold, self.v_reset, self.w()
        )

    def forward(self, dv: torch.Tensor):
        if self.amplitude is None:
            self.v += (self.v - self.v_reset) * self.g + dv
        elif isinstance(self.amplitude, float):
            self.v += (self.v - self.v_reset) * ((self.g.sigmoid() * 2 - 1) * self.amplitude) + dv
        else:
            self.v += (self.v - self.v_reset) * \
                     (self.g.sigmoid() * (self.amplitude[1] - self.amplitude[0]) + self.amplitude[0]) * self.amplitude + dv


        return self.spiking()
