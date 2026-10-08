import torch

from ..reset import voltage
from ..surrogate import _surrogate_spike


def step(x, v, tau, a0, rest, critical, threshold, reset, surrogate_function, detach):
    h = v + (x + a0 * (v - rest) * (v - critical)) / tau
    spike = surrogate_function(h - threshold)
    return spike, voltage(h, spike, threshold, reset, detach), h


def multi_step(
    x_seq,
    v,
    tau,
    a0,
    rest,
    critical,
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
            x, v, tau, a0, rest, critical, threshold, reset, surrogate_function, detach
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
    critical,
    a0,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq,
    surrogate_id,
):
    surrogate_function = lambda value: _surrogate_spike(value, alpha, surrogate_id)
    return _registered_forward(
        x,
        v,
        tau,
        rest,
        critical,
        a0,
        threshold,
        reset,
        detach_reset,
        surrogate_function,
        store_v_seq,
    )


def _registered_forward(
    x,
    v,
    tau,
    rest,
    critical,
    a0,
    threshold,
    reset,
    detach_reset,
    surrogate_function,
    store_v_seq,
):
    spikes, voltages, charged, previous = [], [], [], []
    for current in x.float():
        previous.append(v)
        spike, v, h = step(
            current,
            v,
            tau,
            a0,
            rest,
            critical,
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
