import torch
import torch.nn as nn
import torch.nn.functional as F

def stdp_learning_rule(connection_type, param_shape, *args):
    if connection_type == 'Linear':
        # STDP的情况下，*args应该是突触前后神经元的脉冲发放时间
        w_shape = param_shape
        pre_spike_time = args[0]
        post_spike_time = args[1]


    return 0