"""Torch equations for projection fallback and rematerialized neuron gradients."""

import torch

from .surrogate import _surrogate_gradient


def _rematerialize(x, v, threshold, reset, *, tau=None, decay_input=True):
    spikes, charged = [], []
    voltage = v
    for current in x:
        if tau is None:
            h = voltage + current
        else:
            resting = voltage if reset is None else voltage - reset
            h = voltage + (
                (current - resting) / tau if decay_input else current - resting / tau
            )
        spike = (h >= threshold).to(x.dtype)
        voltage = (
            h - spike * threshold
            if reset is None
            else torch.where(spike.bool(), reset, h)
        )
        spikes.append(spike)
        charged.append(h)
    return torch.stack(spikes), torch.stack(charged), voltage


def _neuron_backward(
    gs,
    gv,
    h,
    sg,
    threshold,
    reset,
    detach,
    alpha,
    surrogate_id,
    *,
    tau=None,
    decay_input=True,
):
    gradients = []
    carry = gv
    for index in range(h.shape[0] - 1, -1, -1):
        value = h[index]
        derivative = (
            sg[index]
            if surrogate_id < 0
            else _surrogate_gradient(value - threshold, alpha, surrogate_id)
        )
        if reset is None:
            dr = 1 if detach else 1 - threshold * derivative
        else:
            dr = 1 - (value >= threshold).to(h.dtype)
            if not detach:
                dr = dr + (reset - value) * derivative
        gh = gs[index] * derivative + carry * dr
        gradients.append(gh / tau if tau is not None and decay_input else gh)
        carry = gh if tau is None else gh * (1 - 1 / tau)
    return torch.stack(gradients[::-1]), carry
