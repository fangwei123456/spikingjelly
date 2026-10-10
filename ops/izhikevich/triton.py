from typing import Optional

import torch
import triton
import triton.language as tl

from ..triton_layout import (
    _block_minor,
    _neuron_grid,
    _neuron_indices,
    _spatial_offsets,
    _time_offset,
    _triton_layout_args,
)
from ..triton_surrogate import _surrogate_gradient
from .autograd import _check, _check_backward


@triton.jit
def _forward_kernel(
    X,
    V,
    W,
    S,
    VO,
    WO,
    H,
    P,
    T,
    N,
    tau,
    rest,
    critical,
    a0,
    a,
    b,
    tau_w,
    threshold,
    reset,
    SOFT: tl.constexpr,
    TRACE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tau = tl.cast(tau, tl.float32)
    rest = tl.cast(rest, tl.float32)
    critical = tl.cast(critical, tl.float32)
    a0 = tl.cast(a0, tl.float32)
    a = tl.cast(a, tl.float32)
    b = tl.cast(b, tl.float32)
    tau_w = tl.cast(tau_w, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
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
    W_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    S_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    VO_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    WO_offset = _spatial_offsets(n, SIZES, STRIDES, 5)
    H_offset = _spatial_offsets(n, SIZES, STRIDES, 6)
    P_offset = _spatial_offsets(n, SIZES, STRIDES, 7)
    v = tl.load(V + V_offset, mask, 0)
    w = tl.load(W + W_offset, mask, 0)
    for t in range(T):
        current = tl.load(X + X_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
            tl.float32
        )
        tl.store(P + P_offset + _time_offset(t, N, STRIDES, 7), v, mask)
        h = v + (current + a0 * (v - rest) * (v - critical) - w) / tau
        tl.store(H + H_offset + _time_offset(t, N, STRIDES, 6), h, mask)
        spike = (h >= threshold).to(tl.float32)
        w = w + (a * (h - rest) - w) / tau_w + b * spike
        if SOFT:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(S + S_offset + _time_offset(t, N, STRIDES, 3), spike, mask)
        if TRACE:
            tl.store(VO + VO_offset + _time_offset(t, N, STRIDES, 4), v, mask)
            tl.store(WO + WO_offset + _time_offset(t, N, STRIDES, 5), w, mask)
    if not TRACE:
        tl.store(VO + VO_offset, v, mask)
        tl.store(WO + WO_offset, w, mask)


@triton.jit
def _backward_kernel(
    GS,
    GV,
    GW,
    H,
    P,
    GX,
    GV0,
    GW0,
    T,
    N,
    tau,
    rest,
    critical,
    a0,
    a,
    b,
    tau_w,
    threshold,
    reset,
    SOFT: tl.constexpr,
    DETACH: tl.constexpr,
    alpha,
    TRACE: tl.constexpr,
    SURROGATE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tau = tl.cast(tau, tl.float32)
    rest = tl.cast(rest, tl.float32)
    critical = tl.cast(critical, tl.float32)
    a0 = tl.cast(a0, tl.float32)
    a = tl.cast(a, tl.float32)
    b = tl.cast(b, tl.float32)
    tau_w = tl.cast(tau_w, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    alpha = tl.cast(alpha, tl.float32)
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
    GW_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    H_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    P_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    GX_offset = _spatial_offsets(n, SIZES, STRIDES, 5)
    GV0_offset = _spatial_offsets(n, SIZES, STRIDES, 6)
    GW0_offset = _spatial_offsets(n, SIZES, STRIDES, 7)
    cv = tl.full(n.shape, 0.0, tl.float32)
    cw = tl.full(n.shape, 0.0, tl.float32)
    for step in range(T):
        t = T - 1 - step
        h = tl.load(H + H_offset + _time_offset(t, N, STRIDES, 3), mask, 0)
        p = tl.load(P + P_offset + _time_offset(t, N, STRIDES, 4), mask, 0)
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
        if SOFT:
            dr = tl.full(n.shape, 1.0, tl.float32)
        else:
            dr = 1.0 - (h >= threshold).to(tl.float32)
        if not DETACH:
            if SOFT:
                dr = dr - threshold * sg
            else:
                dr = dr + (reset - h) * sg
        if TRACE:
            iv = cv + tl.load(GV + GV_offset + _time_offset(t, N, STRIDES, 1), mask, 0)
            iw = cw + tl.load(GW + GW_offset + _time_offset(t, N, STRIDES, 2), mask, 0)
        elif step == 0:
            iv = cv + tl.load(GV + GV_offset, mask, 0)
            iw = cw + tl.load(GW + GW_offset, mask, 0)
        else:
            iv = cv
            iw = cw
        gh = (
            tl.load(GS + GS_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
                tl.float32
            )
            * sg
            + iv * dr
        )
        gh = gh + iw * (b * sg + a / tau_w)
        dh = 1.0 + a0 * (2.0 * p - rest - critical) / tau
        tl.store(GX + GX_offset + _time_offset(t, N, STRIDES, 5), gh / tau, mask)
        cv = gh * dh
        cw = iw * (1.0 - 1.0 / tau_w) - gh / tau
    tl.store(GV0 + GV0_offset, cv, mask)
    tl.store(GW0 + GW0_offset, cw, mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(
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
    )
    torch._check(x.is_cuda, lambda: "Triton izhikevich requires CUDA")
    s = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    wo = torch.empty_like(x if store_v_seq else w, dtype=torch.float32)
    h = torch.empty_like(x, dtype=torch.float32)
    previous = torch.empty_like(h)
    if x.is_contiguous() and v.is_contiguous() and w.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(x, x, v, w, s, vo, wo, h, previous)
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
            w,
            s,
            vo,
            wo,
            h,
            previous,
            x.shape[0],
            neurons,
            tau,
            rest,
            critical,
            a0,
            a,
            b,
            tau_w,
            threshold,
            reset or 0.0,
            reset is None,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return s, vo, wo, h, previous


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gw: torch.Tensor,
    h: torch.Tensor,
    previous: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_backward(
        gs,
        gv,
        gw,
        h,
        previous,
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
    )
    torch._check(h.is_cuda, lambda: "Triton izhikevich requires CUDA")
    gx = torch.empty_like(gs)
    v0 = torch.empty_like(h[0])
    w0 = torch.empty_like(v0)
    if (
        gs.is_contiguous()
        and gv.is_contiguous()
        and gw.is_contiguous()
        and h.is_contiguous()
        and previous.is_contiguous()
    ):
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(h, gs, gv, gw, h, previous, gx, v0, w0)
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
            gw,
            h,
            previous,
            gx,
            v0,
            w0,
            h.shape[0],
            neurons,
            tau,
            rest,
            critical,
            a0,
            a,
            b,
            tau_w,
            threshold,
            reset or 0.0,
            reset is None,
            detach_reset,
            alpha,
            store_v_seq,
            surrogate_id,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return gx, v0, w0
