import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class BaseEncoder(nn.Module):
    def __init__(self):
        '''
        所有编码器的基类
        '''
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def step(self):
        # 对于某些编码器（例如GaussianTuningCurveEncoder），编码一次x，需要经过多步仿真才能将数据输出，这种情况下则用step来获取每一步的数据
        raise NotImplementedError

    def reset(self):
        pass


class ConstantEncoder(BaseEncoder):
    # 将输入简单转化为脉冲，输入中大于0的位置输出1，其他位置输出0
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        '''
        :param x: tensor
        :return: x.bool()
        '''
        return x.bool()

class PeriodicEncoder(BaseEncoder):
    def __init__(self, out_spike):
        '''
        给定out_spike后，周期性的输出out_spike的编码器
        :param out_spike: shape=[T, *]，PeriodicEncoder会不断的输出out_spike[0], out_spike[1], ..., out_spike[T-1], out_spike[0]...
        '''
        super().__init__()
        assert out_spike.dtype == torch.bool
        self.out_spike = out_spike
        self.T = out_spike.shape[0]
        self.index = 0

    def forward(self, x):
        # 使用step代替forward函数，可以替代STDPModule中的neuron_module，来指导连接权重的学习
        return self.step()

    def step(self):
        index = self.index
        self.index += 1
        if self.index == self.T:
            self.index = 0
        return self.out_spike[index]

    def set_out_spike(self, out_spike):
        assert out_spike.dtype == torch.bool
        self.out_spike = out_spike
        self.T = out_spike.shape[0]
        self.index = 0

    def reset(self):
        self.index = 0




class LatencyEncoder(BaseEncoder):
    def __init__(self, max_spike_time, function_type='linear', device='cpu'):
        '''
        延迟编码，刺激强度越大，脉冲发放越早。要求刺激强度已经被归一化到[0, 1]
        脉冲发放时间t_i与刺激强度x_i满足
        type='linear'
             t_i = (t_max - 1) * (1 - x_i)

        type='log'
            t_i = (t_max - 1) - ln(alpha * x_i + 1)
            alpha满足(t_max - 1) - ln(alpha * 1 + 1) = 0
            这导致此编码器很容易发生溢出，因为alpha = math.exp(max_spike_time - 1) - 1，当max_spike_time较大时alpha极大
        :param max_spike_time: 最晚脉冲发放时间
        :param function_type: 'linear'或'log'
        :param device: 数据所在设备

        示例代码
    x = torch.rand(size=[3, 2])
    max_spike_time = 20
    le = encoding.LatencyEncoder(max_spike_time)

    le(x)
    print(x)
    print(le.spike_time)
    for i in range(max_spike_time):
        print(le.step())
        '''
        super().__init__()
        self.device = device
        assert isinstance(max_spike_time, int) and max_spike_time > 1

        self.max_spike_time = max_spike_time
        if function_type == 'log':
            self.alpha = math.exp(max_spike_time - 1) - 1
        elif function_type != 'linear':
            raise NotImplementedError

        self.type = function_type

        self.spike_time = 0
        self.out_spike = 0
        self.index = 0

    def forward(self, x):
        '''
        :param x: 要编码的数据，任意形状的tensor，要求x的数据范围必须在[0, 1]
        '''
        if self.type == 'log':
            self.spike_time = (self.max_spike_time - 1 - torch.log(self.alpha * x + 1)).round().long()
        else:
            self.spike_time = (self.max_spike_time - 1) * (1 - x).round().long()

        self.out_spike = F.one_hot(self.spike_time,
                                       num_classes=self.max_spike_time).bool()  # [batch_size, M, tuning_curve_num, max_spike_time]


    def step(self):
        index = self.index
        self.index += 1
        if self.index == self.max_spike_time:
            self.index = 0

        return torch.index_select(self.out_spike, self.out_spike.dim() - 1, torch.tensor([index], device=self.device).long())

    def reset(self):
        self.spike_time = 0
        self.out_spike = 0
        self.index = 0

class PoissonEncoder(BaseEncoder):
    def __init__(self):
        '''
        泊松频率编码，输出脉冲可以看作是泊松流，发放脉冲的概率即为刺激强度，要求刺激强度已经被归一化到[0, 1]

        示例代码
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
        '''
        out_spike = torch.rand_like(x).le(x)
        # torch.rand_like(x)生成与x相同shape的介于[0, 1]之间的随机数， 这个随机数小于等于x中对应位置的元素，则发放脉冲
        return out_spike




class GaussianTuningCurveEncoder(BaseEncoder):
    def __init__(self, x_min, x_max, tuning_curve_num, max_spike_time, device='cpu'):
        '''
        Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.

        高斯调谐曲线编码，待编码向量是M维tensor，也就是有M个特征。
        1个M维tensor会被编码成shape=[M, tuning_curve_num]的tensor，表示M * tuning_curve_num个神经元的脉冲发放时间
        需要注意的是，编码一次数据，经过max_spike_time步仿真，才能进行下一次的编码。
        :param x_min: float，或者是shape=[M]的tensor，表示M个特征的最小值
        :param x_max: float，或者是shape=[M]的tensor，表示M个特征的最大值
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
        index = self.index
        self.index += 1
        if self.index == self.max_spike_time:
            self.index = 0

        return self.out_spike[:, :, :, index]

    def reset(self):
        self.spike_time = 0
        self.out_spike = 0
        self.index = 0



