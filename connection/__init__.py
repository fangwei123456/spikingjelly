import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConnection(nn.Module):
    def __init__(self):
        '''
        所有突触的基类
        '''
        super().__init__()

    def forward(self, x):
        raise NotImplementedError




class Linear(BaseConnection):
    def __init__(self, in_num, out_num, device='cpu'):
        '''
        线性全连接层，输入是[batch_size, *, in_num]，输出是[batch_size, *, out_num]
        :param in_num: 输入数量
        :param out_num: 输出数量
        :param device: 数据所在设备
        '''
        super().__init__()
        self.w = torch.rand(size=[out_num, in_num], device=device)

    def forward(self, x):
        return torch.matmul(x, self.w.t())







