import torch

from ..reset import voltage
from ..surrogate import _surrogate_spike


def step(x, v, tau, rest, theta, delta, threshold, reset, surrogate_function, detach):
    h = v + (x + rest - v + delta * torch.exp((v - theta) / delta)) / tau
    spike = surrogate_function(h - threshold)
    return spike, voltage(h, spike, threshold, reset, detach), h


def multi_step(
    x_seq,
    v,
    tau,
    rest,
    theta,
    delta,
    threshold,
    reset,
    surrogate_function,
    detach,
    store_v_seq=False,
):
    spikes, voltages, previous, charged = [], [], [], []
    for x in x_seq:
        previous.append(v)
        spike, v, h = step(
            x, v, tau, rest, theta, delta, threshold, reset, surrogate_function, detach
        )
        spikes.append(spike)
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(charged),
        torch.stack(previous),
    )


def _forward_impl(
    x,
    v,
    tau,
    rest,
    theta,
    delta,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq,
    surrogate_id,
):
    surrogate_function = lambda value: _surrogate_spike(value, alpha, surrogate_id)
    spikes, voltages, charged, previous = [], [], [], []
    for current in x.float():
        previous.append(v)
        spike, v, h = step(
            current,
            v,
            tau,
            rest,
            theta,
            delta,
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
        torch.stack(voltages) if store_v_seq else v.contiguous(),
        torch.stack(charged),
        torch.stack(previous),
    )
