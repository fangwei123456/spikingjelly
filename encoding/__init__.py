import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, shape, device='cpu'):
        '''
        所有编码器的基类
        :param shape: 输出的shape
        :param device: 数据所在的设备
        '''
        super().__init__()
        self.shape = shape
        self.device = device

    def forward(self, x):
        raise NotImplementedError


class ConstantEncoder(BaseEncoder):
    # 将输入简单转化为脉冲，输入中大于0的位置输出1，其他位置输出0
    def __init__(self, shape, device='cpu'):
        super().__init__(shape, device)

    def forward(self, x: torch.Tensor):
        '''
        :param x: tensor
        :return: mask
        '''
        assert list(x.shape) == self.shape
        return x.bool()