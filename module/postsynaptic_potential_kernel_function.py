import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def exp_decay_kernel(t, tau=15.0, tau_s=15.0 / 4):
    '''
    postsynaptic potentials
    :param t:
    :param tau:
    :param tau_s:
    :return:
    '''
    t_ = F.relu(t)
    return torch.exp(-t_ / tau) - torch.exp(-t_ / tau_s)

def grad_exp_decay_kernel(t, tau=15.0, tau_s=15.0 / 4):
    '''
    postsynaptic potentials
    :param t:
    :param tau:
    :param tau_s:
    :return:
    '''
    t_ = F.relu(t)
    return -torch.exp(-t_ / tau) / tau + torch.exp(-t_ / tau_s) / tau_s