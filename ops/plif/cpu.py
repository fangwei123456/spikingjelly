from typing import Optional

import torch

from ..surrogate import _surrogate_gradient
from .autograd import _check_backward, _check_forward


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    q = w.float().sigmoid()
    reset_value = 0.0 if reset is None else reset
    spikes, voltages, charged = [], [], []
    for current in x.float():
        h = (
            v + (current - (v - reset_value)) * q
            if decay_input
            else v - (v - reset_value) * q + current
        )
        spike = (h >= threshold).to(torch.float32)
        v = h - spike * threshold if reset is None else spike * reset + (1 - spike) * h
        spikes.append(spike.to(x.dtype))
        if store_v_seq:
            voltages.append(v)
        charged.append(h)
    return (
        torch.stack(spikes),
        (torch.stack(voltages) if store_v_seq else v.contiguous()),
        torch.stack(charged),
    )


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, x, v, w, h, threshold, reset, alpha, store_v_seq, surrogate_id
    )
    q = w.float().sigmoid()
    reset_value = 0.0 if reset is None else reset
    gx = torch.empty_like(x, memory_format=torch.contiguous_format)
    carry = torch.zeros_like(v)
    gq = torch.zeros_like(v)
    for t in range(x.shape[0] - 1, -1, -1):
        sg = _surrogate_gradient(h[t] - threshold, alpha, surrogate_id)
        if reset is None:
            reset_grad = 1 if detach_reset else 1 - threshold * sg
        else:
            reset_grad = 1 - (h[t] >= threshold).to(h.dtype)
            if not detach_reset:
                reset_grad = reset_grad + (reset - h[t]) * sg
        gh = (
            gs[t].float() * sg
            + ((gv[t] if store_v_seq else (gv if t == h.shape[0] - 1 else 0)) + carry)
            * reset_grad
        )
        gx[t] = gh * q if decay_input else gh
        carry = gh - gh * q
        previous = v
        if t > 0:
            spike = (h[t - 1] >= threshold).to(h.dtype)
            previous = (
                h[t - 1] - spike * threshold
                if reset is None
                else spike * reset + (1 - spike) * h[t - 1]
            )
        factor = (
            x[t].float() - (previous - reset_value)
            if decay_input
            else -(previous - reset_value)
        )
        gq = gq + gh * factor
    return gx, carry.contiguous(), (gq.sum() * q * (1 - q)).to(w.dtype)
