from typing import Optional

import torch

from ..reset import voltage
from ..surrogate import _surrogate_spike


def step(
    x, v, threshold: float, reset: Optional[float], surrogate_function, detach: bool
):
    h = v + x
    spike = surrogate_function(h - threshold)
    return spike, voltage(h, spike, threshold, reset, detach), h


def multi_step(
    x_seq,
    v,
    threshold: float,
    reset: Optional[float],
    surrogate_function,
    detach: bool,
    store_v_seq: bool = False,
):
    spikes = []
    voltages = []
    for x in x_seq:
        spike, v, _ = step(x, v, threshold, reset, surrogate_function, detach)
        spikes.append(spike)
        if store_v_seq:
            voltages.append(v)
    return torch.stack(spikes), v, torch.stack(voltages) if store_v_seq else None


def _forward_impl(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    surrogate_function = lambda x: _surrogate_spike(x, alpha, surrogate_id)
    return _sequence(
        x_seq, v_init, threshold, reset, detach_reset, store_v_seq, surrogate_function
    )


def _sequence(
    x_seq, v_init, threshold, reset, detach_reset, store_v_seq, surrogate_function
):
    spikes, voltages, charged = [], [], []
    v = v_init
    for x in x_seq:
        spike, v, h = step(x, v, threshold, reset, surrogate_function, detach_reset)
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(charged),
    )
