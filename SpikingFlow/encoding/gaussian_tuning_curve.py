import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianEncoder:
    def __init__(self, x_min, x_max, neuron_num):
        '''
        Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.

        高斯调谐曲线编码，待编码向量是M维tensor，也就是有M个特征。
        1个M维tensor会被编码成shape=[M, neuron_num]的tensor，表示M * neuron_num个脉冲发放时间
        :param x_min: shape=[M]，M个特征的最小值
        :param x_max: shape=[M]，M个特征的最大值
        :param neuron_num: 编码每个特征使用的神经元数量
        '''
        self.x_min = x_min
        self.x_max = x_max
        self.neuron_num = neuron_num
        self.mu = torch.zeros(size=[x_min.shape[0], neuron_num], dtype=torch.float)
        self.sigma = 1 / 1.5 * (x_max - x_min) / (neuron_num - 2)
        for i in range(neuron_num):
            self.mu[:, i] = x_min + (2 * i - 3) / 2 * (x_max - x_min) / (neuron_num - 2)


    def encode(self, x, max_spike_time=10, discard_late=False, T=500):
        '''
        :param x: 要编码的数据，shape=[N, M]
        :param max_spike_time: 最大脉冲发放时间，所有数据都会被编码到[0, max_spike_time]范围内的脉冲发放时间
        :param discard_late: 如果为True，则认为脉冲发放时间大于max_spike_time*9/10的不会导致激活，设置成仿真周期，表示不发放脉冲
        :param T: 仿真周期
        :return: 编码后的数据，shape=[N, M, neuron_num]
        '''
        ret = torch.zeros(size=[x.shape[0], x.shape[1], self.neuron_num])
        for i in range(self.neuron_num):
            ret[:, :, i] = torch.exp(-torch.pow(x - self.mu[:, i], 2) / 2 / (self.sigma**2))
        ret = -max_spike_time * ret + max_spike_time
        if discard_late:
            ret[ret > max_spike_time * 9 / 10] = T
        return ret.round()
