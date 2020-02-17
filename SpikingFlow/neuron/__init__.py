import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNode(nn.Module):
    def __init__(self, shape, r, v_threshold, v_reset=0.0, device='cpu'):
        '''
        :param shape: 输出的shape，可以看作是神经元的数量
        :param r: 膜电阻，可以是一个float，表示所有神经元的膜电阻均为这个float。
                 也可以是形状为shape的tensor，这样就指定了每个神经元的膜电阻
        :param v_threshold: 阈值电压，可以是一个float，也可以是tensor
        :param v_reset: 重置电压，可以是一个float，也可以是tensor
                        注意，更新过程中会确保电压不低于v_reset，因而电压低于v_reset时会被截断为v_reset
        :param device: 数据所在的设备

        时钟驱动（逐步仿真）的神经元基本模型

        这些神经元都是在t时刻接收电流i作为输入，与膜电阻r相乘，得到dv = i * r

        之后self.v += dv，然后根据神经元自身的属性，决定是否发放脉冲

        需要注意的是，所有的神经元模型都遵循如下约定:

        1. 电压会被截断到[v_reset, v_threshold]

        2. 数据经过一个BaseNode，需要1个dt的时间。可以参考simulating包的流水线的设计

        3. t-dt时刻电压没有达到阈值，t时刻电压达到了阈值，则到t+dt时刻才会放出脉冲。这是为了方便查看波形图，
           如果不这样设计，若t-dt时刻电压为0.1，v_threshold=1.0，v_reset=0.0, t时刻增加了0.9，直接在t时刻发放脉冲，则从波形图
           上看，电压从0.1直接跳变到了0.0，不利于进行数据分析

        '''
        super().__init__()
        self.shape = shape
        self.device = device

        assert isinstance(r, float) or isinstance(r, torch.Tensor)

        if isinstance(r, torch.Tensor):
            assert r.shape == shape
            self.r = r.to(device)
        else:
            self.r = r

        assert isinstance(v_threshold, float) or isinstance(v_threshold, torch.Tensor)

        if isinstance(v_threshold, torch.Tensor):
            assert v_threshold.shape == shape
            self.v_threshold = v_threshold.to(device)
        else:
            self.v_threshold = v_threshold

        assert isinstance(v_reset, float) or isinstance(v_reset, torch.Tensor)
        if isinstance(v_reset, torch.Tensor):
            assert v_reset.shape == shape
            self.v_reset = v_reset.to(device)
        else:
            self.v_reset = v_reset

        self.v = torch.ones(size=shape, dtype=torch.float, device=device) * v_reset

    def __str__(self):
        '''
        :return: 字符串，内容为'shape ' + str(self.shape) + '\nr ' + str(self.r) + '\nv_threshold ' + str(self.v_threshold) + '\nv_reset ' + str(self.v_reset)

        在print这个类时的输出

        子类可以重写这个函数，实现对新增成员变量的打印
        '''

        return 'shape ' + str(self.shape) + '\nr ' + str(self.r) + '\nv_threshold ' + str(
            self.v_threshold) + '\nv_reset ' + str(self.v_reset)

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return:out_spike: shape与self.shape相同，输出脉冲

        接受电流输入，更新膜电位的电压，并输出脉冲（如果过阈值）
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        将所有状态变量全部设置为初始值，作为基类即为将膜电位v设置为v_reset

        对于子类，如果存在除了v以外其他状态量的神经元，应该重写此函数
        '''
        self.v = torch.ones(size=self.shape, dtype=torch.float, device=self.device) * self.v_reset


class IFNode(BaseNode):
    def __init__(self, shape, r, v_threshold, v_reset=0.0, device='cpu'):
        '''
        :param shape: 输出的shape，可以看作是神经元的数量
        :param r: 膜电阻，可以是一个float，表示所有神经元的膜电阻均为这个float。
                 也可以是形状为shape的tensor，这样就指定了每个神经元的膜电阻
        :param v_threshold: 阈值电压，可以是一个float，也可以是tensor
        :param v_reset: 重置电压，可以是一个float，也可以是tensor。
                        注意，更新过程中会确保电压不低于v_reset，因而电压低于v_reset时会被截断为v_reset
        :param device: 数据所在的设备


        IF神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)

        电压一旦达到阈值v_threshold则下一个时刻放出脉冲，同时电压归位到重置电压v_reset

        测试代码

        .. code-block:: python

            if_node = neuron.IFNode([1], r=1.0, v_threshold=1.0)
            v = []
            for i in range(1000):
                if_node(0.01)
                v.append(if_node.v.item())

            pyplot.plot(v)
            pyplot.show()

        '''
        super().__init__(shape, r, v_threshold, v_reset, device)
        self.next_out_spike = torch.zeros(size=shape, dtype=torch.bool, device=device)

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return: out_spike: shape与self.shape相同，输出脉冲
        '''
        out_spike = self.next_out_spike

        self.v += self.r * i
        self.next_out_spike = (self.v >= self.v_threshold)
        self.v[self.next_out_spike] = self.v_threshold

        self.v[self.v < self.v_reset] = self.v_reset

        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset
        return out_spike


class LIFNode(BaseNode):
    def __init__(self, shape, r, v_threshold, v_reset=0.0, tau=1.0, device='cpu'):
        '''
        :param shape: 输出的shape，可以看作是神经元的数量
        :param r: 膜电阻，可以是一个float，表示所有神经元的膜电阻均为这个float。
                 也可以是形状为shape的tensor，这样就指定了每个神经元的膜电阻
        :param v_threshold: 阈值电压，可以是一个float，也可以是tensor
        :param v_reset: 重置电压，可以是一个float，也可以是tensor
                        注意，更新过程中会确保电压不低于v_reset，因而电压低于v_reset时会被截断为v_reset
        :param tau: 膜电位时间常数，越大则充电越慢
                    对于频率编码而言，tau越大，神经元对“频率”的感知和测量也就越精准
                    在分类任务中，增大tau在一定范围内能够显著增加正确率
        :param device: 数据所在的设备

        LIF神经元模型，可以看作是带漏电的积分器

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减

        .. code-block:: python

            v_decay = -(self.v - self.v_reset)
            self.v += (self.r * i + v_decay) / self.tau

        电压一旦达到阈值v_threshold则下一个时刻放出脉冲，同时电压归位到重置电压v_reset


        测试代码

        .. code-block:: python

            lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
            v = []

            for i in range(1000):
                if i < 500:
                    lif_node(0.1)
                else:
                    lif_node(0)
                v.append(lif_node.v.item())

            pyplot.plot(v)
            pyplot.show()
        '''
        super().__init__(shape, r, v_threshold, v_reset, device)
        self.tau = tau
        self.next_out_spike = torch.zeros(size=shape, dtype=torch.bool, device=device)

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return: out_spike: shape与self.shape相同，输出脉冲
        '''
        out_spike = self.next_out_spike

        v_decay = -(self.v - self.v_reset)
        self.v += (self.r * i + v_decay) / self.tau
        self.next_out_spike = (self.v >= self.v_threshold)
        self.v[self.next_out_spike] = self.v_threshold
        self.v[self.v < self.v_reset] = self.v_reset

        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset

        return out_spike

    def __str__(self):
        '''
        :return: None
        '''
        return super().__str__() + '\ntau ' + str(self.tau)
