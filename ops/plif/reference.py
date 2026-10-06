from typing import Optional

import torch

from ..reset import voltage


def step(
    x,
    v,
    w,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    surrogate_function,
    detach: bool,
):
    q = w.sigmoid()
    reset_value = 0.0 if reset is None else reset
    h = (
        v + (x - (v - reset_value)) * q
        if decay_input
        else v - (v - reset_value) * q + x
    )
    spike = surrogate_function(h - threshold)
    return spike, voltage(h, spike, threshold, reset, detach), h


def multi_step(
    x_seq,
    v,
    w,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    surrogate_function,
    detach: bool,
    store_v_seq: bool = False,
):
    spikes = []
    voltages = []
    for x in x_seq:
        spike, v, _ = step(
            x, v, w, decay_input, threshold, reset, surrogate_function, detach
        )
        spikes.append(spike)
        if store_v_seq:
            voltages.append(v)
    return torch.stack(spikes), v, torch.stack(voltages) if store_v_seq else None


def _forward_impl(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ..surrogate import _surrogate_spike

    q = w.float().sigmoid()
    reset_value = 0.0 if reset is None else reset
    surrogate_function = lambda x: _surrogate_spike(x, alpha, surrogate_id)
    spikes, voltages, charged = [], [], []
    v = v_init
    for x in x_seq:
        h = (
            v + (x.float() - (v - reset_value)) * q
            if decay_input
            else v - (v - reset_value) * q + x.float()
        )
        spike = surrogate_function(h - threshold)
        v = voltage(h, spike, threshold, reset, detach_reset)
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(charged),
    )
