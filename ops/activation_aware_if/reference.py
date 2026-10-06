import torch

from ..reset import voltage
from .validation import _check


def step(x, v, threshold, offset, reset, surrogate_function, detach):
    h = v + x
    spike = surrogate_function(h + offset - threshold)
    return spike, voltage(h, spike, threshold, reset, detach)


def multi_step(
    x_seq,
    v,
    threshold,
    offset,
    reset,
    surrogate_function,
    detach,
    store_v_seq=False,
):
    outputs, voltages = [], []
    for x in x_seq:
        spike, v = step(x, v, threshold, offset, reset, surrogate_function, detach)
        outputs.append(spike)
        if store_v_seq:
            voltages.append(v)
    return torch.stack(outputs), torch.stack(voltages) if store_v_seq else v


def _forward_impl(x_seq, v, threshold, offset, channels, inner, reset, store_v_seq):
    _check(x_seq, v, threshold, offset, channels, inner, reset, store_v_seq)
    if channels <= 0 or inner <= 0 or v.numel() % (channels * inner):
        raise ValueError(
            "channel dimensions must be positive and divide the state size"
        )
    if threshold.numel() not in (1, channels) or offset.numel() not in (1, channels):
        raise ValueError("threshold and offset must be scalar or match channel count")
    indices = (
        torch.arange(v.numel(), device=x_seq.device) // inner % channels
    ).reshape(v.shape)
    threshold = (
        threshold.reshape(-1)[indices]
        if threshold.numel() > 1
        else threshold.reshape(())
    )
    offset = offset.reshape(-1)[indices] if offset.numel() > 1 else offset.reshape(())
    outputs, voltages = [], []
    for current in x_seq.float():
        h = v + current
        spike = (h + offset >= threshold).to(torch.float32)
        v = h - spike * threshold if reset is None else spike * reset + (1 - spike) * h
        outputs.append(spike.to(x_seq.dtype))
        if store_v_seq:
            voltages.append(v)
    return torch.stack(outputs), torch.stack(
        voltages
    ) if store_v_seq else v.contiguous()
