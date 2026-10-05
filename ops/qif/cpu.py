from typing import Optional

import torch

from ..surrogate import _surrogate_gradient
from .autograd import _check, _check_backward


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(
        x,
        v,
        tau,
        rest,
        critical,
        a0,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    spikes, voltages, charged, previous = ([], [], [], [])
    for current in x.float():
        previous.append(v)
        h = v + (current + a0 * (v - rest) * (v - critical)) / tau
        spike = (h >= threshold).float()
        v = h - spike * threshold if reset is None else spike * reset + (1 - spike) * h
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v.contiguous(),
        torch.stack(charged),
        torch.stack(previous),
    )


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    previous: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(
        gs,
        gv,
        h,
        previous,
        tau,
        rest,
        critical,
        a0,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    gx = torch.empty_like(gs, memory_format=torch.contiguous_format)
    cv = torch.zeros_like(h[0])
    for t in range(h.shape[0] - 1, -1, -1):
        sg = _surrogate_gradient(h[t] - threshold, alpha, surrogate_id)
        if reset is None:
            dr = 1 if detach_reset else 1 - threshold * sg
        else:
            dr = 1 - (h[t] >= threshold).float()
            if not detach_reset:
                dr = dr + (reset - h[t]) * sg
        incoming_v = cv + (gv[t] if store_v_seq else gv if t == h.shape[0] - 1 else 0)
        gh = gs[t].float() * sg + incoming_v * dr
        dh = 1 + a0 * (2 * previous[t] - rest - critical) / tau
        gx[t] = gh / tau
        cv = gh * dh
    return (gx, cv.contiguous())
