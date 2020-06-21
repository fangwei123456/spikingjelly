import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConnection(nn.Module):
    def __init__(self):
        '''
        所有突触的基类

        突触，输入和输出均为电流，将脉冲转换为电流的转换器定义在connection.transform中
        '''
        super().__init__()

    def forward(self, x):
        '''
        :param x: 输入电流
        :return: 输出电流
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        将突触内的所有状态变量重置为初始状态
        '''
        pass


class ConstantDelay(BaseConnection):
    def __init__(self, delay_time=1):
        '''
        :param delay_time: int，表示延迟时长

        具有固定延迟delay_time的突触，t时刻的输入，在t+1+delay_time时刻才能输出
        '''
        super().__init__()
        assert isinstance(delay_time, int) and delay_time > 0
        self.delay_time = delay_time
        self.queue = []

    def forward(self, x):
        '''
        :param x: 输入电流
        :return: 输出电流

        t时刻的输入，在t+1+delay_time时刻才能输出
        '''
        self.queue.append(x)
        if self.queue.__len__() > self.delay_time:
            return self.queue.pop()
        else:
            return torch.zeros_like(x)

    def reset(self):
        '''
        :return: None

        重置状态变量为初始值，对于ConstantDelay，将保存之前时刻输入的队列清空即可
        '''
        self.queue.clear()

class Linear(BaseConnection):
    def __init__(self, in_num, out_num, device='cpu'):
        '''
        :param in_num: 输入数量
        :param out_num: 输出数量
        :param device: 数据所在设备

        线性全连接层，输入是[batch_size, *, in_num]，输出是[batch_size, *, out_num]

        连接权重矩阵为 :math:`W`，输入为 :math:`x`，输出为 :math:`y`，则

        .. math::
            y = xW^T
        '''
        super().__init__()
        self.w = torch.rand(size=[out_num, in_num], device=device) / 128

    def forward(self, x):
        '''
        :param x: 输入电流，shape=[batch_size, *, in_num]
        :return: 输出电流，shape=[batch_size, *, out_num]
        '''
        return torch.matmul(x, self.w.t())


class GaussianLinear(BaseConnection):
    def __init__(self, in_num, out_num, std, device='cpu'):
        '''
        :param in_num: 输入数量
        :param out_num: 输出数量
        :param std: 噪声的标准差
        :param device: 数据所在设备

        带高斯噪声的线性全连接层，噪声是施加在输出端的，所以可以对不同的神经元产生不同的随机噪声。
        维度上，输入是[batch_size, *, in_num]，输出是[batch_size, *, out_num]。

        连接权重矩阵为 :math:`W`，输入为 :math:`x`，输出为 :math:`y`，标准差为std的噪声为 :math:`e`, 则

        .. math::
            y = xW^T + e
        '''
        super().__init__()
        self.out_num = out_num
        self.w = torch.rand(size=[out_num, in_num], device=device) / 128
        self.std = torch.tensor(std, device=device)
        self.device = device

    def forward(self, x):
        current = torch.matmul(x, self.w.t())
        noise = torch.randn(self.out_num, device=self.device)*self.std
        return current+noise





