from typing import Optional

import torch

from ..reset import voltage
from ..surrogate import _surrogate_spike


def charge(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    reset: Optional[float],
) -> torch.Tensor:
    reset_value = 0.0 if reset is None else reset
    if decay_input:
        return v + (x - (v - reset_value)) / tau
    return v - (v - reset_value) / tau + x


def step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    surrogate_function,
    detach_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = charge(x, v, tau, decay_input, reset)
    spike = surrogate_function(h - threshold)
    v_next = voltage(h, spike, threshold, reset, detach_reset)
    return spike, v_next, h


def multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    surrogate_function,
    detach_reset: bool,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    spikes = []
    voltages = []
    for x in x_seq:
        spike, v, _ = step(
            x,
            v,
            tau,
            decay_input,
            threshold,
            reset,
            surrogate_function,
            detach_reset,
        )
        spikes.append(spike)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        v,
        torch.stack(voltages) if store_v_seq else None,
    )


def _forward_impl(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    surrogate_function = lambda x: _surrogate_spike(x, alpha, surrogate_id)
    spikes = []
    voltages = []
    charged = []
    v = v_init
    for x in x_seq:
        spike, v, h = step(
            x,
            v,
            tau,
            decay_input,
            threshold,
            reset,
            surrogate_function,
            detach_reset,
        )
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(charged),
    )
