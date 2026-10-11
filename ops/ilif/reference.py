import torch

from .autograd import _check


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, v, tau, count, lower, upper, threshold, detach_reset, store_v_seq)
    spikes, voltages, charged = [], [], []
    decay = 1.0 - 1.0 / tau
    for current in x.float():
        h = decay * v + current
        scaled = h / threshold
        hard_spike = scaled.clamp(0.0, count).round()
        window = ((scaled >= lower) & (scaled <= upper)).to(h.dtype) + h * 0.0
        spike = hard_spike + window * (scaled - scaled.detach())
        reset_spike = spike.detach() if detach_reset else spike
        v = h - reset_spike * threshold
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v.contiguous(),
        torch.stack(charged),
    )


def step(x, v, tau, count, lower, upper, threshold, detach_reset=False):
    decay = 1.0 - 1.0 / tau
    h = decay * v + x.to(torch.promote_types(x.dtype, v.dtype))
    scaled = h / threshold
    hard_spike = scaled.clamp(0.0, count).round()
    window = ((scaled >= lower) & (scaled <= upper)).to(h.dtype) + h * 0.0
    spike = hard_spike + window * (scaled - scaled.detach())
    reset_spike = spike.detach() if detach_reset else spike
    return spike.to(x.dtype), h - reset_spike * threshold, h


def multi_step(
    x_seq,
    v,
    tau,
    count,
    lower,
    upper,
    threshold,
    detach_reset=False,
    store_v_seq=False,
):
    spikes, voltages = [], []
    for x in x_seq:
        spike, v, _ = step(x, v, tau, count, lower, upper, threshold, detach_reset)
        spikes.append(spike)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v,
        torch.stack(voltages) if store_v_seq else None,
    )
