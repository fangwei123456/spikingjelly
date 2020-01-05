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
    # 输出常数的编码器
    def __init__(self, shape, device='cpu'):
        super().__init__(shape, device)

    def forward(self, mask):
        '''
        :param mask: 与输出相同shape的bool类型的tensor
        :return: mask
        '''
        assert mask.dtype == torch.bool
        return mask