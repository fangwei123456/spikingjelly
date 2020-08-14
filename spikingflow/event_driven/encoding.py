import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianTuning:
    def __init__(self, n, m, x_min: torch.Tensor, x_max: torch.Tensor):
        '''
        :param n: 特征的数量，int
        :param m: 编码一个特征所使用的神经元数量，int
        :param x_min: n个特征的最小值，shape=[n]的tensor
        :param x_max: n个特征的最大值，shape=[n]的tensor

        Bohte S M, Kok J N, La Poutre J A, et al. Error-backpropagation in temporally encoded networks of spiking \
        neurons[J]. Neurocomputing, 2002, 48(1): 17-37. 中提出的高斯调谐曲线编码方式

        编码器所使用的变量所在的device与x_min.device一致
        '''
        assert m > 2
        self.m = m
        self.n = n
        i = torch.arange(1, m+1).unsqueeze(0).repeat(n, 1).float().to(x_min.device)  # shape=[n, m]
        self.mu = x_min.unsqueeze(-1).repeat(1, m) + \
                  (2 * i - 3) / 2 * \
                  (x_max.unsqueeze(-1).repeat(1, m) - x_min.unsqueeze(-1).repeat(1, m)) / (m - 2)  # shape=[n, m]
        self.sigma2 = (1 / 1.5 * (x_max - x_min) / (m - 2)).unsqueeze(-1).square().repeat(1, m)  # shape=[n, m]

        # print('mu\n', self.mu)
        # print('sigma2\n', self.sigma2)

    def encode(self, x: torch.Tensor, max_spike_time=50):
        '''
        :param x: shape=[batch_size, n, k]，batch_size个数据，每个数据含有n个特征，每个特征中有k个数据
        :param max_spike_time: 最大（最晚）脉冲发放时间，也可以称为编码时间窗口的长度
        :return: out_spikes, shape=[batch_size, n, k, m]，将每个数据编码成了m个神经元的脉冲发放时间
        '''
        x_shape = x.shape
        # shape: [batch_size, n, k] -> [batch_size, k, n]
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, x.shape[2])  # shape=[batch_size * k, n]
        x = x.unsqueeze(-1).repeat(1, 1, self.m)  # shape=[batch_size * k, n, m]
        # 计算x对应的m个高斯函数值
        y = torch.exp(- (x - self.mu).square() / 2 / self.sigma2)  # shape=[batch_size * k, n, m]
        out_spikes = (max_spike_time * (1 - y)).round()
        out_spikes[out_spikes >= max_spike_time] = -1  # -1表示无脉冲发放
        # shape: [batch_size * k, n, m] -> [batch_size, k, n, m]
        out_spikes = out_spikes.view(x_shape[0], x_shape[2], out_spikes.shape[1], out_spikes.shape[2])
        out_spikes = out_spikes.permute(0, 2, 1, 3)  # shape: [batch_size, k, n, m] -> [batch_size, n, k, m]
        return out_spikes



