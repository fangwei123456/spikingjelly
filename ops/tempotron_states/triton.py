import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..tempotron_roots.triton import _phi


@triton.jit
def _compose(a, b, c, d, e, aa, bb, cc, dd, ee):
    return aa * a, aa * b + bb * c, cc * c, aa * d + bb * e + dd, cc * e + ee


@triton.jit(do_not_specialize=["width"])
def _states(
    T,
    IN,
    V,
    Q,
    ORIGIN,
    NORM,
    OUT_V,
    OUT_Q,
    width,
    outputs: tl.constexpr,
    slow: tl.constexpr,
    fast: tl.constexpr,
    delta: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slow = tl.full((), slow, tl.float64)
    fast = tl.full((), fast, tl.float64)
    delta = tl.full((), delta, tl.float64)
    lane = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    batch = row // outputs
    valid = lane < width
    now = tl.load(T + batch * width + lane, valid, 0)
    origin = tl.load(ORIGIN + batch)
    previous = tl.load(T + batch * width + lane - 1, valid & (lane > 0), 0)
    previous = tl.where(lane == 0, origin, previous)
    elapsed = tl.where(valid, now - previous, 0.0)
    a = libdevice.exp(-elapsed / slow)
    b = a * _phi(elapsed, delta)
    c = libdevice.exp(-elapsed / fast)
    d = tl.full((BLOCK,), 0.0, tl.float64)
    e = tl.load(IN + row * width + lane, valid, 0)
    a, b, c, d, e = tl.associative_scan((a, b, c, d, e), 0, _compose)
    voltage = tl.load(V + row)
    current = tl.load(Q + row)
    norm = tl.load(NORM)
    tl.store(OUT_V + row * width + lane, (a * voltage + b * current + d) / norm, valid)
    tl.store(OUT_Q + row * width + lane, c * current + e, valid)


def _forward_impl(
    times,
    incoming,
    voltage,
    current,
    origin,
    slow,
    fast,
    delta,
    normalizer,
    *,
    _kernel_wrapper=None,
):
    out_voltage = torch.empty(
        incoming.shape, dtype=incoming.dtype, device=incoming.device
    )
    out_current = torch.empty_like(out_voltage)
    if incoming.numel():
        wrapper = _kernel_wrapper or (lambda k: k)
        with torch.cuda.device(times.device):
            wrapper(_states)[(incoming.shape[0] * incoming.shape[1],)](
                times.contiguous(),
                incoming.contiguous(),
                voltage.contiguous(),
                current.contiguous(),
                origin.contiguous(),
                normalizer,
                out_voltage,
                out_current,
                times.shape[1],
                incoming.shape[1],
                slow,
                fast,
                delta,
                triton.next_power_of_2(times.shape[1]),
                num_warps=8,
                enable_fp_fusion=False,
            )
    return out_voltage, out_current
