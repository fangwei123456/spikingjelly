import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class BaseEncoder(nn.Module):
    def __init__(self):
        '''
        所有编码器的基类。编码器将输入数据（例如图像）编码为脉冲数据。
        '''
        super().__init__()

    def forward(self, x):
        '''
        :param x: 要编码的数据
        :return: 编码后的脉冲，或者是None

        将x编码为脉冲。少数编码器（例如ConstantEncoder）可以将x编码成时长为1个dt的脉冲，在这种情况下，本函数返回编码后的脉冲。

        多数编码器（例如PeriodicEncoder），都是把x编码成时长为n个dt的脉冲out_spike，out_spike.shape=[n, *]。

        因此编码一次后，需要调用n次step()函数才能将脉冲全部发放完毕，第index此调用step()会得到out_spike[index]。
        '''
        raise NotImplementedError

    def step(self):
        '''
        :return: 1个dt的脉冲

        多数编码器（例如PeriodicEncoder），编码一次x，需要经过多步仿真才能将数据输出，这种情况下则用step来获取每一步的数据。
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        将编码器的所有状态变量设置为初始状态。对于有状态的编码器，需要重写这个函数。
        '''
        pass

class PeriodicEncoder(BaseEncoder):
    def __init__(self, out_spike):
        '''
        :param out_spike: shape=[T, *]，PeriodicEncoder会不断的输出out_spike[0], out_spike[1], ..., out_spike[T-1],
                          out_spike[0], out_spike[1], ...

        给定out_spike后，周期性的输出out_spike[0], out_spike[1], ..., out_spike[T-1]的编码器。
        '''
        super().__init__()
        assert out_spike.dtype == torch.bool
        self.out_spike = out_spike
        self.T = out_spike.shape[0]
        self.index = 0

    def forward(self, x):
        '''
        :param x: 输入数据，实际上并不需要输入数据，因为out_spike在初始化时已经被指定了
        :return: 调用step()后得到的返回值
        '''
        return self.step()

    def step(self):
        '''
        :return: out_spike[index]

        初始化时index=0，每调用一次，index则自增1，index为T时修改为0。
        '''
        index = self.index
        self.index += 1
        if self.index == self.T:
            self.index = 0
        return self.out_spike[index]

    def set_out_spike(self, out_spike):
        '''
        :param out_spike: 新设定的out_spike，必须是torch.bool
        :return: None

        重新设定编码器的输出脉冲self.out_spike为out_spike。
        '''
        assert out_spike.dtype == torch.bool
        self.out_spike = out_spike
        self.T = out_spike.shape[0]
        self.index = 0

    def reset(self):
        '''
        :return: None

        重置编码器的状态变量，对于PeriodicEncoder而言将索引index置0即可。
        '''
        self.index = 0




class LatencyEncoder(BaseEncoder):
    def __init__(self, max_spike_time, function_type='linear', device='cpu'):
        '''
        :param max_spike_time: 最晚（最大）脉冲发放时间
        :param function_type: 'linear'或'log'
        :param device: 数据所在设备

        延迟编码，刺激强度越大，脉冲发放越早。要求刺激强度已经被归一化到[0, 1]。

        脉冲发放时间 :math:`t_i` 与刺激强度 :math:`x_i` 满足：

        type='linear'
            .. math::
                t_i = (t_{max} - 1) * (1 - x_i)

        type='log'
            .. math::
                t_i = (t_{max} - 1) - ln(alpha * x_i + 1)

        :math:`alpha` 满足：

        .. math::
            (t_{max} - 1) - ln(alpha * 1 + 1) = 0
        这导致此编码器很容易发生溢出，因为

        .. math::
            alpha = exp(t_{max} - 1) - 1

        当 :math:`t_{max}` 较大时 :math:`alpha` 极大。

        示例代码：

        .. code-block:: python

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

        将输入数据x编码为max_spike_time个时刻的max_spike_time个脉冲。
        '''

        # 将输入数据转换为不同时刻发放的脉冲
        if self.type == 'log':
            self.spike_time = (self.max_spike_time - 1 - torch.log(self.alpha * x + 1)).round().long()
        else:
            self.spike_time = (self.max_spike_time - 1) * (1 - x).round().long()

        self.out_spike = F.one_hot(self.spike_time,
                                       num_classes=self.max_spike_time).bool()  # [*, max_spike_time]

        self.out_spike.transpose_(0, self.out_spike.shape[-1])  # [*, max_spike_time] -> [max_spike_time, *]


    def step(self):
        '''
        :return: out_spike[index]

        初始化时index=0，每调用一次，index则自增1，index为max_spike_time时修改为0。

        '''
        index = self.index
        self.index += 1
        if self.index == self.max_spike_time:
            self.index = 0

        return self.out_spike[self.index]

    def reset(self):
        '''
        :return: None

        重置LatencyEncoder的所有状态变量（包括spike_time，out_spike，index）为初始值0。
        '''
        self.spike_time = 0
        self.out_spike = 0
        self.index = 0

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



