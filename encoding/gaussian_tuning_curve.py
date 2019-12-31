import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.
'''

class GaussianEncoder:
    def __init__(self, x_min, x_max, neuron_num, device='cpu'):
        self.x_min = x_min  # 要编码的数据的最小值
        self.x_max = x_max
        self.neuron_num = neuron_num  # 编码使用的神经元数量
        self.mu = torch.zeros(size=[neuron_num], dtype=torch.float, device=device)
        self.sigma = 1 / 1.5 * (x_max - x_min) / (neuron_num - 2)
        for i in range(neuron_num):
            self.mu[i] = x_min + (2 * i - 3) / 2 * (x_max - x_min) / (neuron_num - 2)


    def encode(self, x, max_spike_time=10, discard_late=False, T=500):
        """
        x是shape=[N]的tensor，M个神经元，x中的每个值都被编码成neuron_num个神经元的脉冲发放时间，也就是一个[neuron_num]的tensor
        因此，x的编码结果为shape=[N, neuron_num]的tensor，第j行表示的是x_j的编码结果
        记第i个高斯函数为f_i，则高斯函数作用后结果应该为
        f_0(x_0), f_1(x_0), ...
        f_0(x_1), f_1(x_1),...
        """
        ret = x.repeat([self.neuron_num, 1])  # [neuron_num, N]
        """
        [x0, x1, x2, ...
         x0, x1, x2, ...]
        """
        for i in range(self.neuron_num):
            ret[i] = torch.exp(-torch.pow(ret[i] - self.mu[i], 2) / 2 / (self.sigma**2))
        """
        [f_0(x0), f_0(x1), ...
         f_1(x0), f_1(x1), ...]
        接下来进行取整，函数值从[1,0]对应脉冲发放时间[0, max_spike_time]，计算时会取整
        discard_late==True则认为脉冲发放时间大于max_spike_time*9/10的不会导致激活，设置成仿真周期，表示不发放脉冲
        """
        ret = -max_spike_time * ret + max_spike_time
        ret = torch.round(ret)
        if discard_late:
            ret[ret > max_spike_time * 9 / 10] = T
        return ret.t()  # x的编码结果为shape=[N, neuron_num]的tensor，第j行表示的是x_j的编码结果。返回的dtype=float32