from abc import abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn

from .. import surrogate, base


__all__ = ["ILCBaseNode", "ILCIFNode", "ILCLIFNode", "ILCCUBALIFNode"]


class ILCBaseNode(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.out_pop_dim = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1.0 - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike)
            if t < T - 1:
                x_seq[t + 1] = x_seq[t + 1] + self.conn(
                    spike.view(-1, self.act_dim, self.dec_pop_dim)
                ).view(-1, self.out_pop_dim)

        return torch.stack(spike_seq)


class ILCIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class ILCLIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_decay: float = 0.75,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v * self.v_decay + x


class ILCCUBALIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)
