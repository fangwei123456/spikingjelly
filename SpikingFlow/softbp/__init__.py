import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNode(nn.Module):
    def __init__(self, v_threshold, v_reset):
        super().__init__()
        self.v = 0
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    @staticmethod
    def pulse_soft(x):
        return torch.sigmoid(x)

    def spiking(self):
        if self.training:
            spike_hard = (self.v >= self.v_threshold).float()
            spike_soft = self.pulse_soft(self.v - self.v_threshold)
            v_hard = self.v_reset * spike_hard + self.v * (1 - spike_hard)
            v_soft = self.v_reset * spike_soft + self.v * (1 - spike_soft)
            self.v = v_soft + (v_hard - v_soft).detach_()
            return spike_soft + (spike_hard - spike_soft).detach_()
        else:
            spike_hard = (self.v >= self.v_threshold).float()
            self.v = self.v_reset * spike_hard + self.v * (1 - spike_hard)
            return spike_hard

    def forward(self, dv: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        self.v = 0

class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__(v_threshold, v_reset)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()



class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
        super().__init__(v_threshold, v_reset)
        self.tau = tau

    def forward(self, dv: torch.Tensor):
        self.v += (dv + -(self.v - self.v_reset)) / self.tau
        return self.spiking()

