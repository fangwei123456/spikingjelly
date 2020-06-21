import torch
import torch.nn as nn
import torch.nn.functional as F
import SpikingFlow.softbp.soft_pulse_function as soft_pulse_function


class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, pulse_soft=soft_pulse_function.Sigmoid()):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为None，当神经元释放脉冲后，电压会被重置为v_reset；如果设置为None，则电压会被减去阈值
        :param pulse_soft: 反向传播时用来计算脉冲函数梯度的替代函数，即软脉冲函数

        softbp包中，可微分SNN神经元的基类神经元

        可微分SNN神经元，在前向传播时输出真正的脉冲（离散的0和1）。脉冲的产生过程可以看作是一个\\
        阶跃函数：

        .. math::
            S = \\Theta(V - V_{threshold})

            其中\\Theta(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            0, & x < 0
            \\end{cases}

        :math:`\\Theta(x)` 是一个不可微的函数，用一个形状与其相似的函数 :math:`\\sigma(x)`，即代码中的pulse_soft去近\\
        似它的梯度。默认的pulse_soft = SpikingFlow.softbp.soft_pulse_function.Sigmoid()，\\
        在反向传播时用 :math:`\\sigma'(x)` 来近似 :math:`\\Theta'(x)`，这样就可以使用梯度下降法来更新SNN了

        前向传播使用 :math:`\\Theta(x)`，反向传播时按前向传播为 :math:`\\sigma(x)` 来计算梯度，在PyTorch中很容易实现，参见\\
        这个类的spiking()函数
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        self.pulse_soft = pulse_soft


    def spiking(self):
        '''
        :return: 神经元的输出脉冲

        根据当前神经元的电压、阈值、重置电压，计算输出脉冲，并更新神经元的电压
        '''

        if self.training:
            spike_hard = (self.v >= self.v_threshold).float()
            spike_soft = self.pulse_soft(self.v - self.v_threshold)
            if self.v_reset is None:
                v_hard = self.v - spike_hard * self.v_threshold
                v_soft = self.v - spike_soft * self.v_threshold
            else:
                v_hard = self.v_reset * spike_hard + self.v * (1 - spike_hard)
                v_soft = self.v_reset * spike_soft + self.v * (1 - spike_soft)
            self.v = v_soft + (v_hard - v_soft).detach_()
            return spike_soft + (spike_hard - spike_soft).detach_()
        else:
            spike_hard = (self.v >= self.v_threshold).float()
            if self.v_reset is None:
                self.v = self.v - spike_hard * self.v_threshold
            else:
                self.v = self.v_reset * spike_hard + self.v * (1 - spike_hard)
            return spike_hard


    def forward(self, dv: torch.Tensor):
        '''
        :param dv: 输入到神经元的电压增量
        :return: 神经元的输出脉冲

        子类需要实现这一函数
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        重置神经元为初始状态，也就是将电压设置为v_reset

        如果子类的神经元还含有其他状态变量，需要在此函数中将这些状态变量全部重置
        '''
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, pulse_soft=soft_pulse_function.Sigmoid()):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param pulse_soft: 反向传播时用来计算脉冲函数梯度的替代函数，即软脉冲函数


        IF神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)

        电压一旦达到阈值v_threshold则放出脉冲，同时电压归位到重置电压v_reset
        '''
        super().__init__(v_threshold, v_reset, pulse_soft)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()



class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, pulse_soft=soft_pulse_function.Sigmoid()):
        '''
        :param tau: 膜电位时间常数，越大则充电越慢
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param pulse_soft: 反向传播时用来计算脉冲函数梯度的替代函数，即软脉冲函数

        LIF神经元模型，可以看作是带漏电的积分器

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减
        '''
        super().__init__(v_threshold, v_reset, pulse_soft)
        self.tau = tau

    def forward(self, dv: torch.Tensor):
        self.v += (dv - (self.v - self.v_reset)) / self.tau
        return self.spiking()




class PLIFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, pulse_soft=soft_pulse_function.Sigmoid()):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压
        :param pulse_soft: 反向传播时用来计算脉冲函数梯度的替代函数，即软脉冲函数

        Parametric LIF神经元模型，时间常数tau可学习的LIF神经元。对于同一层神经元，它们的tau是共享的

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减
        '''
        super().__init__(v_threshold, v_reset, pulse_soft)
        self.tau = nn.Parameter(torch.ones(size=[1]) / 2)

    def forward(self, dv: torch.Tensor):
        self.v += (dv - (self.v - self.v_reset)) * self.tau
        return self.spiking()
