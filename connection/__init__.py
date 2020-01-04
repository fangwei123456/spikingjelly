import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConnection(nn.Module):
    def __init__(self, learning=False):
        super().__init__()
        self.learning = learning  # True表示参与学习过程

    def forward(self, x):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError


class Linear(BaseConnection):
    def __init__(self, in_num, out_num, learning, device='cpu'):
        '''
        线性全连接层，输入是[batch_size, *, in_num]，输出是[batch_size, *, out_num]
        :param in_num: 输入数量
        :param out_num: 输出数量
        '''
        super().__init__(self, learning)
        self.w = torch.rand(size=[out_num, in_num], device=device)

    def forward(self, x):
        return torch.matmul(x, self.w.t())

    def update(self, delta_w):
        self.w += delta_w
