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


def _forward_impl(
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
