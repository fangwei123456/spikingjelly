import torch.nn as nn
import torch
import numpy as np

class VoltageHook(nn.Module):
    def __init__(self, scale=1.0, momemtum=0.1, mode='MaxNorm'):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.mode = mode
        self.num_batches_tracked = 0
        self.momentum = momemtum

    def forward(self, x):
        if self.mode == 'MaxNorm':
            s_t = x.max().detach()
        else:
            s_t = torch.tensor(np.percentile(x.detach().cpu(), 99))
        
        if self.num_batches_tracked == 0:
            self.scale = s_t
        else:
            self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
        self.num_batches_tracked += x.shape[0]
        # print(self.num_batches_tracked, self.scale.item())
        return x

class VoltageScaler(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x):
        return x * self.scale

    def extra_repr(self):
        return '%f' % self.scale.item()