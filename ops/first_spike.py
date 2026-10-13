"""Private forward arithmetic for bracketed first-spike roots."""

import torch

from spikingjelly.logger import logger

try:
    import triton
    import triton.language as tl
except (ImportError, OSError) as error:
    triton = None
    logger.info("First-spike Triton kernels unavailable: {}", error)


if triton is not None:

    @triton.jit(do_not_specialize=["n"])
    def _propose(L, H, T, V, D, M, OUT, FLAGS, STATUS, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < n
        lo, hi, trial = (
            tl.load(L + i, valid, 0),
            tl.load(H + i, valid, 0),
            tl.load(T + i, valid, 0),
        )
        value, slope = tl.load(V + i, valid, 0), tl.load(D + i, valid, 0)
        invalid = (
            (value != value)
            | (tl.abs(value) == float("inf"))
            | (slope != slope)
            | (tl.abs(slope) == float("inf"))
        )
        moving = tl.load(M + i, valid, False) & ~invalid
        value, slope = tl.where(invalid, 0, value), tl.where(invalid, 1, slope)
        exact = moving & (value == 0)
        lo = tl.where(exact | (moving & (value < 0)), trial, lo)
        hi = tl.where(exact | (moving & (value >= 0)), trial, hi)
        transverse = moving & (slope > 0)
        step = tl.where(transverse, value, 0) / tl.where(transverse, slope, 1)
        proposal = trial - step
        tiny = tl.full((), 1.401298464324817e-45, tl.float32)
        eps = tl.full((), 9.5367431640625e-7, tl.float32)
        near = transverse & (lo < hi) & (tl.abs(step) <= tiny + eps * tl.abs(trial))
        radius = (tiny + eps * tl.abs(proposal)) / 4
        lower = tl.where(near, tl.maximum(lo, proposal - radius), lo)
        upper = tl.where(near, tl.minimum(hi, proposal + radius), lo)
        tl.store(OUT + i, lo, valid)
        tl.store(OUT + n + i, hi, valid)
        tl.store(OUT + 2 * n + i, proposal, valid)
        tl.store(OUT + 3 * n + i, tl.abs(step), valid)
        tl.store(OUT + 4 * n + i, lower, valid)
        tl.store(OUT + 5 * n + i, upper, valid)
        tl.store(FLAGS + i, transverse, valid)
        tl.store(FLAGS + n + i, near, valid)
        status = tl.where(invalid & valid, 2, tl.where(near & valid, 1, 0))
        tl.store(STATUS + tl.program_id(0), tl.max(status, 0))

    @triton.jit(do_not_specialize=["n"])
    def _finish(
        L,
        H,
        P,
        S,
        OLD,
        T,
        TRANSVERSE,
        NEAR,
        LL,
        HH,
        FL,
        FH,
        OUT,
        n,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < n
        lo, hi = tl.load(L + i, valid, 0), tl.load(H + i, valid, 0)
        lower, upper = tl.load(LL + i, valid, 0), tl.load(HH + i, valid, 0)
        certified = (
            tl.load(NEAR + i, valid, False)
            & (lower <= upper)
            & (tl.load(FL + i, valid, 0) <= 0)
            & (tl.load(FH + i, valid, 0) >= 0)
        )
        lo, hi = tl.where(certified, lower, lo), tl.where(certified, upper, hi)
        proposal, step = tl.load(P + i, valid, 0), tl.load(S + i, valid, 0)
        trial, previous = tl.load(T + i, valid, 0), tl.load(OLD + i, valid, 0)
        newton = (
            tl.load(TRANSVERSE + i, valid, False)
            & (proposal == proposal)
            & (tl.abs(proposal) != float("inf"))
            & (proposal > lo)
            & (proposal < hi)
            & (proposal != trial)
            & (step <= previous / 2)
        )
        next_step = tl.where(newton, step, (hi - lo) / 2)
        next_trial = tl.where(newton, proposal, lo + (hi - lo) / 2)
        tl.store(OUT + i, lo, valid)
        tl.store(OUT + n + i, hi, valid)
        tl.store(OUT + 2 * n + i, next_trial, valid)
        tl.store(OUT + 3 * n + i, next_step, valid)

    @triton.jit(do_not_specialize=["n", "width", "queries", "origin_stride"])
    def _cubic(
        T,
        P,
        C,
        ORIGIN,
        DURATION,
        V,
        D,
        n,
        width,
        queries,
        origin_stride,
        outputs: tl.constexpr,
        SINGLE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < n
        row = i if SINGLE else i // queries
        interval = tl.load(P + i, valid, 0) // 3
        origin = tl.load(ORIGIN + row // outputs * origin_stride + interval, valid, 0)
        duration = tl.load(DURATION)
        x = (tl.load(T + i, valid, 0) - origin) / duration
        index = (row * width + interval) * 4
        c0 = tl.load(C + index, valid, 0)
        c1 = tl.load(C + index + 1, valid, 0)
        c2 = tl.load(C + index + 2, valid, 0)
        c3 = tl.load(C + index + 3, valid, 0)
        value = ((c3 * x + c2) * x + c1) * x + c0 - 1
        slope = ((3 * c3 * x + 2 * c2) * x + c1) / duration
        tl.store(V + i, value, valid)
        tl.store(D + i, slope, valid)


def _newton_proposal(lo, hi, trial, value, slope, moving):
    if lo.is_cuda and triton is not None:
        output = lo.new_empty((6, *lo.shape))
        flags = moving.new_empty((2, *moving.shape))
        status = torch.empty(
            (triton.cdiv(lo.numel(), 256),), device=lo.device, dtype=torch.int32
        )
        _propose[(triton.cdiv(lo.numel(), 256),)](
            lo,
            hi,
            trial,
            value.contiguous(),
            slope.contiguous(),
            moving,
            output,
            flags,
            status,
            lo.numel(),
            256,
            enable_fp_fusion=False,
        )
        result = int(status.max())
        if result == 2:
            raise RuntimeError("response returned a non-finite residual or slope.")
        return (*output.unbind(), *flags.unbind(), result == 1)
    if not bool(torch.isfinite(value).all() & torch.isfinite(slope).all()):
        raise RuntimeError("response returned a non-finite residual or slope.")
    exact = moving & (value == 0)
    lo = torch.where(exact | (moving & (value < 0)), trial, lo)
    hi = torch.where(exact | (moving & (value >= 0)), trial, hi)
    transverse = moving & (slope > 0)
    step = torch.where(transverse, value, 0) / torch.where(transverse, slope, 1)
    proposal = trial - step
    tolerance = 1.401298464324817e-45 + 8 * torch.finfo(torch.float32).eps * trial.abs()
    near = transverse & (lo < hi) & (step.abs() <= tolerance)
    radius = (
        1.401298464324817e-45 + 8 * torch.finfo(torch.float32).eps * proposal.abs()
    ) / 4
    lower = torch.where(near, torch.maximum(lo, proposal - radius), lo)
    upper = torch.where(near, torch.minimum(hi, proposal + radius), lo)
    return (
        lo,
        hi,
        proposal,
        step.abs(),
        lower,
        upper,
        transverse,
        near,
        bool(near.any()),
    )


def _newton_finish(
    lo, hi, proposal, step, previous, trial, transverse, near, lower, upper, f_lo, f_hi
):
    if lo.is_cuda and triton is not None:
        output = lo.new_empty((4, *lo.shape))
        _finish[(triton.cdiv(lo.numel(), 256),)](
            lo,
            hi,
            proposal,
            step,
            previous,
            trial,
            transverse,
            near,
            lower,
            upper,
            f_lo.contiguous(),
            f_hi.contiguous(),
            output,
            lo.numel(),
            256,
            enable_fp_fusion=False,
        )
        return output.unbind()
    certified = near & (lower <= upper) & (f_lo <= 0) & (f_hi >= 0)
    lo = torch.where(certified, lower, lo)
    hi = torch.where(certified, upper, hi)
    newton = (
        transverse
        & torch.isfinite(proposal)
        & (proposal > lo)
        & (proposal < hi)
        & (proposal != trial)
        & (step <= previous / 2)
    )
    return (
        lo,
        hi,
        torch.where(newton, proposal, lo + (hi - lo) / 2),
        torch.where(newton, step, (hi - lo) / 2),
    )


def _cubic_value(time, piece, coefficients, origins, duration):
    if time.is_cuda and triton is not None:
        value, slope = torch.empty_like(time), torch.empty_like(time)
        _cubic[(triton.cdiv(time.numel(), 256),)](
            time.contiguous(),
            piece.contiguous(),
            coefficients,
            origins,
            duration,
            value,
            slope,
            time.numel(),
            coefficients.shape[-2],
            time.shape[-1],
            origins.stride(0),
            coefficients.shape[1],
            time.shape[-1] == 1,
            256,
            enable_fp_fusion=False,
        )
        return value, slope
    interval = piece // 3
    x = (time - origins.gather(-1, interval)) / duration
    c0, c1, c2, c3 = (coefficients[..., k].gather(-1, interval) for k in range(4))
    return ((c3 * x + c2) * x + c1) * x + c0 - 1, (
        (3 * c3 * x + 2 * c2) * x + c1
    ) / duration
