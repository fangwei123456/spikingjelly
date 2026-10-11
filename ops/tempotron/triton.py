import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _phi(u, delta):
    z = delta * u
    series = u * (1 - z / 2 + z * z / 6)
    return tl.where(
        tl.abs(z) < 1.4901161193847656e-8, series, -libdevice.expm1(-z) / delta
    )


@triton.jit
def _value(v, q, u, slow, delta, norm):
    decay = libdevice.exp(-u / slow)
    return v * decay / norm + q * (decay * _phi(u, delta) / norm)


@triton.jit(do_not_specialize=["n"])
def _solve(
    V,
    Q,
    L,
    U,
    M,
    N,
    OUT,
    ERROR,
    n,
    slow: tl.constexpr,
    fast: tl.constexpr,
    delta: tl.constexpr,
    threshold: tl.constexpr,
    iterations: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slow = tl.full((), slow, tl.float64)
    fast = tl.full((), fast, tl.float64)
    delta = tl.full((), delta, tl.float64)
    threshold = tl.full((), threshold, tl.float64)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < n
    mask = tl.load(M + index, valid, False)
    norm = tl.load(N)
    voltage = tl.where(mask, tl.load(V + index, valid, 0), 0.0)
    current = tl.where(mask, tl.load(Q + index, valid, 0), 0.0)
    low = tl.where(mask, tl.load(L + index, valid, 0), 0.0)
    high = tl.where(mask, tl.load(U + index, valid, 0), 0.0)
    absolute = 1.7763568394002505e-15 * tl.minimum(high - low, fast)
    absolute = tl.maximum(absolute, 4.9406564584124654e-324)
    for _ in range(iterations):
        middle = low + (high - low) / 2
        tolerance = absolute + 1.7763568394002505e-15 * tl.maximum(
            tl.abs(low), tl.abs(high)
        )
        moving = mask & ((high - low) > tolerance) & (middle > low) & (middle < high)
        value = _value(voltage, current, middle, slow, delta, norm)
        below = value < threshold
        low = tl.where(moving & below, middle, low)
        high = tl.where(moving & ~below, middle, high)
    tolerance = absolute + 1.7763568394002505e-15 * tl.maximum(
        tl.abs(low), tl.abs(high)
    )
    # A representable midpoint is sufficient to detect an unconverged bracket.
    middle = low + (high - low) / 2
    unconverged = mask & ((high - low) > tolerance) & (middle > low) & (middle < high)
    value = _value(voltage, current, high, slow, delta, norm)
    raw = value * norm
    q = current * libdevice.exp(-high / fast)
    slope = (q - raw / slow) / norm
    scale = (
        1 + threshold + (tl.abs(voltage) + tl.abs(current * _phi(high, delta))) / norm
    )
    residual = 1.4210854715202004e-14 * scale + tl.abs(slope) * tolerance
    bad = mask & (
        (value != value)
        | (tl.abs(value) == float("inf"))
        | (tl.abs(value - threshold) > residual)
    )
    tl.store(OUT + index, high, valid)
    tl.store(ERROR + index, tl.where(unconverged, 1, tl.where(bad, 2, 0)), valid)


def _roots_impl(
    voltage,
    current,
    lower,
    upper,
    mask,
    slow,
    fast,
    delta,
    normalizer,
    threshold,
    duration,
    *,
    _kernel_wrapper=None,
):
    output = torch.empty(upper.shape, device=upper.device, dtype=upper.dtype)
    error = torch.empty(upper.shape, device=upper.device, dtype=torch.int32)
    iterations = (
        math.ceil(
            max(0.0, math.log2(duration) - math.log2(fast))
            - math.log2(8 * torch.finfo(torch.float64).eps)
        )
        + 2
    )
    if not upper.numel():
        return output
    wrapper = _kernel_wrapper or (lambda k: k)
    with torch.cuda.device(voltage.device):
        wrapper(_solve)[(triton.cdiv(upper.numel(), 128),)](
            voltage.contiguous(),
            current.contiguous(),
            lower.contiguous(),
            upper.contiguous(),
            mask.contiguous(),
            normalizer,
            output,
            error,
            upper.numel(),
            slow,
            fast,
            delta,
            threshold,
            iterations,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if bool((error == 1).any()):
        raise RuntimeError("Tempotron threshold root did not converge.")
    if bool((error == 2).any()):
        raise RuntimeError("Tempotron threshold root has an invalid voltage residual.")
    return output


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


def _states_impl(
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


def _weight_gradient_impl(
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
