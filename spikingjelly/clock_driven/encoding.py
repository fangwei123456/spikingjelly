import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import base
from abc import abstractmethod


class StatelessEncoder(nn.Module):
    pass


class StatefulEncoder(base.MemoryModule):
    def __init__(self, T: int):
        super().__init__()
        assert isinstance(T, int) and T >= 1
        self.T = T
        self.register_memory('spike', None)
        self.register_memory('t', 0)

    def forward(self, x):
        t = self.t
        self.t += 1
        if self.t >= self.T:
            self.t = 0

        return self.spike[t]

    @abstractmethod
    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'T={self.T}'


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: torch.Tensor):
        super().__init__(spike.shape[0])
        self.encode(spike)

    def encode(self, spike: torch.Tensor):
        self.spike = spike


class LatencyEncoder(StatefulEncoder):
    def __init__(self, T: int, function_type='linear'):
        super().__init__(T)

        if function_type == 'log':
            self.alpha = math.exp(T - 1.) - 1.
        elif function_type != 'linear':
            raise NotImplementedError

        self.type = function_type

    def encode(self, x: torch.Tensor):
        if self.type == 'log':
            t_f = (self.T - 1. - torch.log(self.alpha * x + 1.)).round().long()
        else:
            t_f = ((self.T - 1.) * (1. - x)).round().long()

        self.spike = F.one_hot(t_f, num_classes=self.max_spike_time).float()




class PoissonEncoder(BaseEncoder):
    def __init__(self):
        '''
        泊松频率编码，输出脉冲可以看作是泊松流，发放脉冲的概率即为刺激强度，要求刺激强度已经被归一化到[0, 1]。

        示例代码：

        .. code-block:: python

            pe = encoding.PoissonEncoder()
            x = torch.rand(size=[8])
            print(x)
            for i in range(10):
                print(pe(x))
        '''
        super().__init__()

    def forward(self, x):
        '''
        :param x: 要编码的数据，任意形状的tensor，要求x的数据范围必须在[0, 1]

        将输入数据x编码为脉冲，脉冲发放的概率即为对应位置元素的值。
        '''
        out_spike = torch.rand_like(x).le(x)
        # torch.rand_like(x)生成与x相同shape的介于[0, 1)之间的随机数， 这个随机数小于等于x中对应位置的元素，则发放脉冲
        return out_spike


class GaussianTuningCurveEncoder(BaseEncoder):
    def __init__(self, x_min, x_max, tuning_curve_num, max_spike_time, device='cpu'):
        '''
        :param x_min: float，或者是shape=[M]的tensor，表示M个特征的最小值
        :param x_max: float，或者是shape=[M]的tensor，表示M个特征的最大值
        :param tuning_curve_num: 编码每个特征使用的高斯函数（调谐曲线）数量
        :param max_spike_time: 最大脉冲发放时间，所有数据都会被编码到[0, max_spike_time - 1]范围内的脉冲发放时间
        :param device: 数据所在设备

        Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.

        高斯调谐曲线编码，一种时域编码方法。

        首先生成tuning_curve_num个高斯函数，这些高斯函数的对称轴在数据范围内均匀排列，对于每一个输入x，计算tuning_curve_num个\
        高斯函数的值，使用这些函数值线性地生成tuning_curve_num个脉冲发放时间。

        待编码向量是M维tensor，也就是有M个特征。

        1个M维tensor会被编码成shape=[M, tuning_curve_num]的tensor，表示M * tuning_curve_num个神经元的脉冲发放时间。

        需要注意的是，编码一次数据，经过max_spike_time步仿真，才能进行下一次的编码。

        示例代码：

        .. code-block:: python

            x = torch.rand(size=[3, 2])
            tuning_curve_num = 10
            max_spike_time = 20
            ge = encoding.GaussianTuningCurveEncoder(x.min(0)[0], x.max(0)[0], tuning_curve_num=tuning_curve_num, max_spike_time=max_spike_time)
            ge(x)
            for i in range(max_spike_time):
                print(ge.step())
        '''
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        assert tuning_curve_num > 2
        self.tuning_curve_num = tuning_curve_num
        assert isinstance(max_spike_time, int) and max_spike_time > 1
        self.max_spike_time = max_spike_time
        self.device = device
        if isinstance(x_min, torch.Tensor):
            self.mu = torch.zeros(size=[x_min.shape[0], tuning_curve_num], dtype=torch.float, device=self.device)
        else:
            self.mu = torch.zeros(size=[1, tuning_curve_num], dtype=torch.float, device=self.device)

        # 生成tuning_curve_num个高斯函数的方差和均值
        self.sigma = 1 / 1.5 * (x_max - x_min) / (tuning_curve_num - 2)
        for i in range(tuning_curve_num):
            self.mu[:, i] = x_min + (2 * i - 3) / 2 * (x_max - x_min) / (tuning_curve_num - 2)

        self.spike_time = 0
        self.out_spike = 0
        self.index = 0

    def forward(self, x):
        '''
        :param x: 要编码的数据，shape=[batch_size, M]

        将输入数据x编码为脉冲。
        '''
        assert self.index == 0
        self.spike_time = torch.zeros(size=[x.shape[0], x.shape[1], self.tuning_curve_num], dtype=torch.float,
                                      device=self.device)
        for i in range(self.tuning_curve_num):
            self.spike_time[:, :, i] = torch.exp(
                -torch.pow(x - self.mu[:, i], 2) / 2 / (self.sigma ** 2))  # 数值在[0, 1]之间
        self.spike_time = (-(self.max_spike_time - 1) * self.spike_time + (
                self.max_spike_time - 1)).round().long()  # [batch_size, M, tuning_curve_num]
        self.out_spike = F.one_hot(self.spike_time,
                                   num_classes=self.max_spike_time).bool()  # [batch_size, M, tuning_curve_num, max_spike_time]
        # 太晚发放的脉冲（最后时刻的脉冲）认为全部是0
        self.out_spike[:, :, :, -1].zero_()

    def step(self):
        '''
        :return: out_spike[index]

        初始化时index=0，每调用一次，index则自增1，index为max_spike_time时修改为0。
        '''
        index = self.index
        self.index += 1
        if self.index == self.max_spike_time:
            self.index = 0

        return self.out_spike[:, :, :, index]

    def reset(self):
        '''
        :return: None

        重置GaussianTuningCurveEncoder的所有状态变量（包括spike_time，out_spike，index）为初始值0。
        '''
        self.spike_time = 0
        self.out_spike = 0
        self.index = 0


class IntervalEncoder(BaseEncoder):
    def __init__(self, T_in, shape, device='cpu'):
        '''
        :param T_in: 脉冲发放的间隔
        :param shape: 输出形状
        :param device: 输出脉冲所在的设备

        每隔 ``T_in`` 个步长就发放一次脉冲的编码器。
        '''
        super().__init__()
        self.t = 0
        self.T_in = T_in
        self.out_spike = [torch.zeros(size=shape, device=device, dtype=torch.bool),
                          torch.ones(size=shape, device=device, dtype=torch.bool)]

    def step(self):
        if self.t == self.T_in:
            self.t = 0
            return self.out_spike[1]
        else:
            self.t += 1
            return self.out_spike[0]

    def reset(self):
        self.t = 0


class WeightedPhaseEncoder(BaseEncoder):
    def __init__(self, phase, period, device='cpu'):
        '''
        :param phase: 一个周期内用于编码的相数
        :type phase: int
        :param device: 输出脉冲所在的设备
        :type device: str

        Kim J, Kim H, Huh S, et al. Deep neural networks with weighted spikes[J]. Neurocomputing, 2018, 311: 373-386.

        带权的相位编码，一种基于二进制表示的编码方法。

        将输入按照二进制各位展开，从高位到低位遍历输入进行脉冲编码。相比于频率编码，每一位携带的信息量更多。编码相位数为 :math:`K` 时，可以对于处于区间 :math:`[0, 1-2^{-K}]` 的数进行编码。以下为原始论文中的示例：

        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | Phase (K=8)                      | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              |
        +==================================+================+================+================+================+================+================+================+================+
        | Spike weight :math:`\omega(t)`   | 2\ :sup:`-1`   | 2\ :sup:`-2`   | 2\ :sup:`-3`   | 2\ :sup:`-4`   | 2\ :sup:`-5`   | 2\ :sup:`-6`   | 2\ :sup:`-7`   | 2\ :sup:`-8`   |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 192/256                          | 1              | 1              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 1/256                            | 0              | 0              | 0              | 0              | 0              | 0              | 0              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 128/256                          | 1              | 0              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 255/256                          | 1              | 1              | 1              | 1              | 1              | 1              | 1              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        '''
        super().__init__()
        self.t = 0
        self.phase = phase
        self.device = device

    def forward(self, x):
        '''
        :param x: 要编码的数据，shape=[batch_size, *]

        将输入数据x编码为一个周期内的脉冲。
        '''
        assert (x >= 0).all() and (x <= 1 - 2 ** (-self.phase)).all()
        inputs = x.copy()
        self.out_spike = torch.empty((self.phase,) + x.shape, device=self.device)  # 编码为[phase, batch_size, *]
        w = 0.5
        for i in range(self.phase):
            self.out_spike[i] = inputs >= w
            inputs -= w * self.out_spike[i]
            w *= 0.5

    def step(self):
        out = self.out_spike[self.t]
        self.t += 1
        if self.t == self.phase:
            self.t = 0
        return out

    def reset(self):
        self.t = 0
