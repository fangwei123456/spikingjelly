"""Torch reference for CLIF's complete state transition."""

import torch

from spikingjelly.activation_based import surrogate

from ..surrogate import _SURROGATE_IDS

_SURROGATES = {
    index: getattr(surrogate, name) for name, index in _SURROGATE_IDS.items()
}
_SURROGATES[7] = surrogate.Rect


def step(x, v, m, tau, threshold, spike_function):
    h = v - (v - 0.0) / tau + x
    decayed_m = m * torch.sigmoid(h / tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    return spike, h - spike * (threshold + torch.sigmoid(m)), m


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    leak_v: torch.Tensor,
    m: torch.Tensor,
    tau: float,
    threshold: float,
    alpha: float,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .autograd import _check

    _check(x, (v, leak_v, m), tau, threshold, alpha, surrogate_id)
    h = v - leak_v / tau + x
    decayed_m = m * torch.sigmoid(h / tau)
    if surrogate_id == 8:
        spike = surrogate.heaviside(h - threshold)
    else:
        spike = _SURROGATES[surrogate_id].spiking_function(h - threshold, alpha)
    m_next = decayed_m + spike
    v_next = h - spike * (threshold + torch.sigmoid(m_next))
    return spike, v_next, m_next, h
