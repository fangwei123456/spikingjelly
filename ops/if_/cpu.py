from typing import Optional

import torch

from ..surrogate import _surrogate_gradient
from ..validation import _check_gradients, _check_inputs


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_inputs(x, v, threshold, reset, alpha, surrogate_id)
    spikes, voltages, charged = [], [], []
    for current in x.float():
        h = v + current
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
    h: torch.Tensor,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_gradients(gs, gv, h, threshold, reset, alpha, store_v_seq, surrogate_id)
    gx = torch.empty_like(h, dtype=gs.dtype, memory_format=torch.contiguous_format)
    carry = torch.zeros_like(h[0])
    for t in range(h.shape[0] - 1, -1, -1):
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
        gx[t] = gh
        carry = gh
    return gx, carry.contiguous()
