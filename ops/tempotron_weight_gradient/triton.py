import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..tempotron_roots.triton import _phi


@triton.jit(do_not_specialize=["count"])
def _gradient(
    T,
    I,
    ORDER,
    BATCH,
    RANK,
    PEAK,
    ACCEPT,
    G,
    NORM,
    OUT,
    count,
    inputs: tl.constexpr,
    outputs: tl.constexpr,
    slow: tl.constexpr,
    delta: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slow = tl.full((), slow, tl.float64)
    delta = tl.full((), delta, tl.float64)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out = tl.program_id(1)
    live = lane < count
    pos = tl.load(ORDER + lane, live, 0)
    batch = tl.load(BATCH + lane, live, 0)
    rank = tl.load(RANK + lane, live, 0)
    time = tl.load(T + pos, live, 0).to(tl.float64)
    column = tl.load(I + pos, live, 0)
    peak = tl.load(PEAK + batch * outputs + out, live, 0)
    accepted = tl.load(ACCEPT + batch * outputs + out, live, 0)
    g = tl.load(G + batch * outputs + out, live, 0)
    valid = live & (rank < accepted) & (time < peak) & (g != 0)
    delay = tl.where(valid, peak - time, 0.0)
    norm = tl.load(NORM)
    value = libdevice.exp(-delay / slow) * _phi(delay, delta) / norm
    tl.atomic_add(OUT + out * inputs + column, g * value, valid, sem="relaxed")


def _forward_impl(
    times,
    indices,
    weight,
    order,
    offsets,
    peak,
    accepted,
    grad_max,
    slow,
    delta,
    normalizer,
    *,
    _kernel_wrapper=None,
):
    gradient = torch.zeros(weight.shape, device=weight.device, dtype=torch.float64)
    if order.numel():
        batches = torch.repeat_interleave(
            torch.arange(peak.shape[0], device=times.device),
            offsets.diff(),
            output_size=order.numel(),
        )
        ranks = torch.arange(order.numel(), device=times.device) - offsets[batches]
        wrapper = _kernel_wrapper or (lambda k: k)
        with torch.cuda.device(times.device):
            wrapper(_gradient)[(triton.cdiv(order.numel(), 256), weight.shape[0])](
                times.contiguous(),
                indices.contiguous(),
                order,
                batches,
                ranks,
                peak.contiguous(),
                accepted.contiguous(),
                grad_max.contiguous(),
                normalizer,
                gradient,
                order.numel(),
                weight.shape[1],
                weight.shape[0],
                slow,
                delta,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return gradient
