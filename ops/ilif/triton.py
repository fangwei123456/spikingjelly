import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..triton_layout import (
    _block_minor,
    _neuron_grid,
    _neuron_indices,
    _spatial_offsets,
    _time_offset,
    _triton_layout_args,
)
from .autograd import _check, _check_backward


@triton.jit
def _forward_kernel(
    X,
    V,
    S,
    VO,
    H,
    T,
    N,
    tau,
    count,
    threshold,
    TRACE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tau = tl.cast(tau, tl.float32)
    count = tl.cast(count, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    SIZES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[0]
    STRIDES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[1]
    MINOR: tl.constexpr = 1 if LAYOUT is None else tl.constexpr(LAYOUT).value[2]
    if LAYOUT is None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = n < N
    else:
        n, mask = _neuron_indices(tl.constexpr(LAYOUT).value[3], BLOCK, SIZES, MINOR)
    X_offset = _spatial_offsets(n, SIZES, STRIDES, 0)
    V_offset = _spatial_offsets(n, SIZES, STRIDES, 1)
    S_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    VO_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    H_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    v = tl.load(V + V_offset, mask, 0)
    for t in range(T):
        current = tl.load(X + X_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
            tl.float32
        )
        h = (1.0 - 1.0 / tau) * v + current
        # Keep FP32 workspace stores before low-precision spikes: Triton 3.3's
        # loop layout conversion otherwise fails for FP16 voltage traces.
        tl.store(H + H_offset + _time_offset(t, N, STRIDES, 4), h, mask)
        spike = libdevice.rint(
            tl.minimum(tl.maximum(tl.div_rn(h, threshold), 0.0), count)
        )
        v = h - spike * threshold
        tl.store(S + S_offset + _time_offset(t, N, STRIDES, 2), spike, mask)
        if TRACE:
            tl.store(VO + VO_offset + _time_offset(t, N, STRIDES, 3), v, mask)
    if not TRACE:
        tl.store(VO + VO_offset, v, mask)


@triton.jit
def _backward_kernel(
    GS,
    GV,
    H,
    GX,
    GV0,
    T,
    N,
    tau,
    lower,
    upper,
    threshold,
    DETACH: tl.constexpr,
    TRACE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tau = tl.cast(tau, tl.float32)
    lower = tl.cast(lower, tl.float32)
    upper = tl.cast(upper, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    SIZES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[0]
    STRIDES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[1]
    MINOR: tl.constexpr = 1 if LAYOUT is None else tl.constexpr(LAYOUT).value[2]
    if LAYOUT is None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = n < N
    else:
        n, mask = _neuron_indices(tl.constexpr(LAYOUT).value[3], BLOCK, SIZES, MINOR)
    GS_offset = _spatial_offsets(n, SIZES, STRIDES, 0)
    GV_offset = _spatial_offsets(n, SIZES, STRIDES, 1)
    H_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    GX_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    GV0_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    cv = tl.full(n.shape, 0.0, tl.float32)
    for step in range(T):
        t = T - 1 - step
        h = tl.load(H + H_offset + _time_offset(t, N, STRIDES, 2), mask, 0)
        scaled = tl.div_rn(h, threshold)
        sg = tl.where((scaled >= lower) & (scaled <= upper), 1.0 / threshold, 0.0)
        dr = tl.full(n.shape, 1.0, tl.float32)
        if not DETACH:
            dr = dr - threshold * sg
        if TRACE:
            iv = cv + tl.load(GV + GV_offset + _time_offset(t, N, STRIDES, 1), mask, 0)
        elif step == 0:
            iv = cv + tl.load(GV + GV_offset, mask, 0)
        else:
            iv = cv
        gh = (
            tl.load(GS + GS_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
                tl.float32
            )
            * sg
            + iv * dr
        )
        dh = 1.0 - 1.0 / tau
        tl.store(GX + GX_offset + _time_offset(t, N, STRIDES, 3), gh, mask)
        cv = gh * dh
    tl.store(GV0 + GV0_offset, cv, mask)


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
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, v, tau, count, lower, upper, threshold, detach_reset, store_v_seq)
    torch._check(x.is_cuda, lambda: "Triton ilif requires CUDA")
    s = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    h = torch.empty_like(x, dtype=torch.float32)
    if x.is_contiguous() and v.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(x, x, v, s, vo, h)
        minor = _block_minor(sizes, strides)
    neurons = v.numel()
    with torch.cuda.device(x.device):
        (
            _forward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_forward_kernel)
        )[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            x,
            v,
            s,
            vo,
            h,
            x.shape[0],
            neurons,
            tau,
            count,
            threshold,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return s, vo, h


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
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, h, tau, count, lower, upper, threshold, detach_reset, store_v_seq
    )
    torch._check(h.is_cuda, lambda: "Triton ilif requires CUDA")
    gx = torch.empty_like(gs)
    v0 = torch.empty_like(h[0])
    if gs.is_contiguous() and gv.is_contiguous() and h.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(h, gs, gv, h, gx, v0)
        minor = _block_minor(sizes, strides)
    neurons = v0.numel()
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            gs,
            gv,
            h,
            gx,
            v0,
            h.shape[0],
            neurons,
            tau,
            lower,
            upper,
            threshold,
            detach_reset,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return gx, v0
