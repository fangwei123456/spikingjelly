import torch

from ..reset import voltage
from ..surrogate import _surrogate_spike


def step(
    x,
    v,
    w,
    tau,
    rest,
    critical,
    a0,
    a,
    b,
    tau_w,
    threshold,
    reset,
    surrogate_function,
    detach,
):
    h = v + (x + a0 * (v - rest) * (v - critical) - w) / tau
    spike = surrogate_function(h - threshold)
    w_next = w + (a * (h - rest) - w) / tau_w + b * spike
    return spike, voltage(h, spike, threshold, reset, detach), w_next, h


def multi_step(
    x_seq,
    v,
    w,
    tau,
    rest,
    critical,
    a0,
    a,
    b,
    tau_w,
    threshold,
    reset,
    surrogate_function,
    detach,
    store_v_seq=False,
):
    spikes, voltages, recoveries, charged, previous = [], [], [], [], []
    for x in x_seq:
        previous.append(v)
        spike, v, w, h = step(
            x,
            v,
            w,
            tau,
            rest,
            critical,
            a0,
            a,
            b,
            tau_w,
            threshold,
            reset,
            surrogate_function,
            detach,
        )
        spikes.append(spike)
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
            recoveries.append(w)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(recoveries) if store_v_seq else w,
        torch.stack(charged),
        torch.stack(previous),
    )


def _forward_impl(
    x,
    v,
    w,
    tau,
    rest,
    critical,
    a0,
    a,
    b,
    tau_w,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq,
    surrogate_id,
):
    surrogate_function = lambda value: _surrogate_spike(value, alpha, surrogate_id)
    spikes, voltages, recoveries, charged, previous = [], [], [], [], []
    for current in x.float():
        previous.append(v)
        spike, v, w, h = step(
            current,
            v,
            w,
            tau,
            rest,
            critical,
            a0,
            a,
            b,
            tau_w,
            threshold,
            reset,
            surrogate_function,
            detach_reset,
        )
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
            recoveries.append(w)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v.contiguous(),
        torch.stack(recoveries) if store_v_seq else w.contiguous(),
        torch.stack(charged),
        torch.stack(previous),
    )
