import torch
import torch.nn as nn
import torch.nn.functional as F
from SpikingFlow.clock_driven import surrogate, accelerating
import math

class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为None，当神经元释放脉冲后，电压会被重置为v_reset；如果设置为None，则电压会被减去阈值
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
                        若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                        转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

        可微分SNN神经元的基类神经元。

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
        :param monitor_state: True或False，表示开启或关闭monitor
        :return: None

        设置开启或关闭monitor。
        '''
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def spiking(self):
        '''
        :return: 神经元的输出脉冲

        根据当前神经元的电压、阈值、重置电压，计算输出脉冲，并更新神经元的电压。
        '''
        spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy())
            self.monitor['s'].append(spike.data.cpu().numpy())

        if self.v_reset is None:
            self.v = accelerating.soft_vlotage_transform(self.v, spike, self.v_threshold)
        else:
            self.v = accelerating.hard_voltage_transform(self.v, spike, self.v_reset)

        return spike

    def forward(self, dv: torch.Tensor):
        '''
        :param dv: 输入到神经元的电压增量
        :return: 神经元的输出脉冲

        子类需要实现这一函数。
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        重置神经元为初始状态，也就是将电压设置为v_reset。

        如果子类的神经元还含有其他状态变量，需要在此函数中将这些状态变量全部重置。
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
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
                        若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                        转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

        IF神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减：

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)

        电压一旦达到阈值v_threshold则放出脉冲，同时电压归位到重置电压v_reset。
        '''
        super().__init__(v_threshold, v_reset, surrogate_function, monitor_state)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()


class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(),
                 monitor_state=False):
        '''
        :param tau: 膜电位时间常数，越大则充电越慢
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
                        若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                        转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

        LIF神经元模型，可以看作是带漏电的积分器：

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减。
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
        :param init_tau: 初始的tau
        :param decay: 为 ``True`` 时会限制 ``tau`` 的取值恒大于1，使得神经元不会给自身充电；为 ``False`` 时不会有任何限制
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param monitor: 是否设置监视器来保存神经元的电压和释放的脉冲。
                        若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                        转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

        Parametric LIF神经元模型，时间常数tau可学习的LIF神经元：

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减。对于同一层神经元，它们的tau是共享的。

        .. tip::
            LIF神经元的电压更新方程为：

            ``self.v += (dv - (self.v - self.v_reset)) / self.tau``

            为了防止出现除以0的情况，PLIF神经元没有使用除法，而是用乘法代替：

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

    def extra_repr(self):
        if self.decay:
            tau = 1 / self.w.data.sigmoid()
        else:
            tau = 1 / self.w.data

        return 'v_threshold={}, v_reset={}, tau={}'.format(
            self.v_threshold, self.v_reset, tau
        )

class RIFNode(BaseNode):
    def __init__(self, init_w=-1e-3, amplitude=None, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        '''
        :param init_w: 初始的自连接权重
        :param amplitude: 对自连接权重的限制。若为 ``None``，则不会对权重有任何限制；
                            若为一个 ``float``，会限制权重在 ``(- amplitude, amplitude)`` 范围内；
                            若为一个 ``tuple``，会限制权重在 ``(amplitude[0], amplitude[1])`` 范围内。
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
                        若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                        转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

        Recurrent IF神经元模型。与Parametric LIF神经元模型类似，但有微妙的区别，自连接权重不会作用于输入。其膜电位更新方程为：

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = w(V(t) - V_{reset}) + R_{m}I(t)

        其中 :math:`w` 是自连接权重，权重是可以学习的。对于同一层神经元，它们的 :math:`w` 是共享的。

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
        :return: 返回自连接权重
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
