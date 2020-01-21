import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
这个py文件包含各种 :math:`v(t)` 函数
'''
def exp_decay_kernel(t, tau=15.0, tau_s=15.0 / 4):
    t_ = F.relu(t)
    return torch.exp(-t_ / tau) - torch.exp(-t_ / tau_s)

def grad_exp_decay_kernel(t, tau=15.0, tau_s=15.0 / 4):
    t_ = F.relu(t)
    return -torch.exp(-t_ / tau) / tau + torch.exp(-t_ / tau_s) / tau_s