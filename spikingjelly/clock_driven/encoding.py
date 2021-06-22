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
        self.T = spike.shape[0]


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

        self.spike = F.one_hot(t_f, num_classes=self.max_spike_time).to(x)


class PoissonEncoder(StatelessEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike


class GaussianTuningCurveEncoder(StatefulEncoder):
    def __init__(self, x_min, x_max, tuning_curve_num, T):
        super().__init__(T)
        self.x_min = x_min
        self.x_max = x_max
        assert tuning_curve_num > 2
        self.tuning_curve_num = tuning_curve_num
        if isinstance(x_min, torch.Tensor):
            self.mu = torch.zeros(size=[x_min.shape[0], tuning_curve_num], dtype=torch.float, device=x_min.device)
        else:
            self.mu = torch.zeros(size=[1, tuning_curve_num], dtype=torch.float)

        # 生成tuning_curve_num个高斯函数的方差和均值
        self.sigma = 1 / 1.5 * (x_max - x_min) / (tuning_curve_num - 2)
        for i in range(tuning_curve_num):
            self.mu[:, i] = x_min + (2 * i - 3) / 2 * (x_max - x_min) / (tuning_curve_num - 2)

    def encode(self, x: torch.Tensor):
        self.to(x.device)

        t_f = torch.zeros(size=[x.shape[0], x.shape[1], self.tuning_curve_num], dtype=torch.float,
                                      device=x.device)

        for i in range(self.tuning_curve_num):
            self.spike_time[:, :, i] = torch.exp(-torch.pow(x - self.mu[:, i], 2) / 2 / (self.sigma ** 2))  # 数值在[0, 1]之间

        t_f = (-(self.T - 1) * t_f + (self.T - 1)).round().long()  # [batch_size, M, tuning_curve_num]

        self.spike = F.one_hot(t_f, num_classes=self.T).float()  # [batch_size, M, tuning_curve_num, T]
        # 太晚发放的脉冲（最后时刻的脉冲）认为全部是0
        self.spike[:, :, :, -1].zero_()
        self.spike = self.spike.permute(3, 0, 1, 2)



class WeightedPhaseEncoder(StatefulEncoder):
    def __init__(self, T: int):
        '''
        :param T: 一个周期内用于编码的相数
        :type T: int
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
        super().__init__(T)

    def encode(self, x: torch.Tensor):
        '''
        :param x: 要编码的数据，shape=[batch_size, *]

        将输入数据x编码为一个周期内的脉冲。
        '''
        assert (x >= 0).all() and (x <= 1 - 2 ** (-self.phase)).all()
        inputs = x.clone()
        self.spike = torch.empty((self.phase,) + x.shape, device=x.device)  # 编码为[phase, batch_size, *]
        w = 0.5
        for i in range(self.phase):
            self.spike[i] = inputs >= w
            inputs -= w * self.spike[i]
            w *= 0.5
