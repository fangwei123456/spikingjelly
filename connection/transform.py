import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeCurrent(nn.Module):
    def __init__(self, amplitude=1):
        '''
        无记忆
        输入脉冲，输出与脉冲形状完全相同、离散的、大小为amplitude的电流
        :param amplitude: 电流的大小
        '''
        super().__init__()
        self.amplitude = amplitude

    def forward(self, in_spike):
        return in_spike.float() * self.amplitude

class ExpDecayCurrent(nn.Module):
    def __init__(self, tau, amplitude=1):
        '''
        有记忆
        若当前时刻到达一个脉冲，则电流增加amplitude；否则电流按指数衰减
        :param tau: 时间常数，衰减系数
        :param amplitude: 电流的大小
        '''
        super().__init__()
        self.tau = tau
        self.amplitude = amplitude
        self.i = 0

    def forward(self, in_spike):
        in_spike_float = in_spike.float()
        i_decay = -self.i / self.tau
        self.i += i_decay * (1 - in_spike_float) + self.amplitude * in_spike_float
        return self.i





