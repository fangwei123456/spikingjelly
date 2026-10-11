import torch

from .autograd import _check, _check_backward


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
    spikes, voltages, charged = ([], [], [])
    for current in x.float():
        h = (1 - 1 / tau) * v + current
        spike = (h / threshold).clamp(0, count).round()
        v = h - spike * threshold
        spikes.append(spike.to(x.dtype))
        charged.append(h)
        if store_v_seq:
            voltages.append(v)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if store_v_seq else v.contiguous(),
        torch.stack(charged),
    )


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, h, tau, count, lower, upper, threshold, detach_reset, store_v_seq
    )
    gx = torch.empty_like(gs, memory_format=torch.contiguous_format)
    cv = torch.zeros_like(h[0])
    for t in range(h.shape[0] - 1, -1, -1):
        scaled = h[t] / threshold
        sg = ((scaled >= lower) & (scaled <= upper)).float() / threshold
        dr = 1 if detach_reset else 1 - threshold * sg
        incoming_v = cv + (gv[t] if store_v_seq else gv if t == h.shape[0] - 1 else 0)
        gh = gs[t].float() * sg + incoming_v * dr
        dh = 1 - 1 / tau
        gx[t] = gh
        cv = gh * dh
    return (gx, cv.contiguous())
