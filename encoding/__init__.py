import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self):
        '''
        所有编码器的基类
        :param device: 数据所在的设备
        '''
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def step(self):
        # 对于某些编码器（例如GaussianTuningCurveEncoder），编码一次x，需要经过多步仿真才能将数据输出，这种情况下则用step来获取每一步的数据
        raise NotImplementedError


class ConstantEncoder(BaseEncoder):
    # 将输入简单转化为脉冲，输入中大于0的位置输出1，其他位置输出0
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        '''
        :param x: tensor
        :return: x.bool()
        '''
        assert list(x.shape) == self.shape
        return x.bool()

class GaussianTuningCurveEncoder(BaseEncoder):
    def __init__(self, x_min, x_max, tuning_curve_num, max_spike_time, device='cpu'):
        '''
        Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.

        高斯调谐曲线编码，待编码向量是M维tensor，也就是有M个特征。
        1个M维tensor会被编码成shape=[M, tuning_curve_num]的tensor，表示M * tuning_curve_num个神经元的脉冲发放时间
        需要注意的是，编码一次数据，经过max_spike_time步仿真，才能进行下一次的编码。
        :param x_min: shape=[M]，M个特征的最小值
        :param x_max: shape=[M]，M个特征的最大值
        :param tuning_curve_num: 编码每个特征使用的高斯函数（调谐曲线）数量
        :param max_spike_time: 最大脉冲发放时间，所有数据都会被编码到[0, max_spike_time - 1]范围内的脉冲发放时间
        :param device: 数据所在设备

        示例代码
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
        self.mu = torch.zeros(size=[x_min.shape[0], tuning_curve_num], dtype=torch.float, device=self.device)

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

    def step(self):
        index = self.index
        self.index += 1
        if self.index == self.max_spike_time:
            self.index = 0
        return self.out_spike[:, :, :, index]





