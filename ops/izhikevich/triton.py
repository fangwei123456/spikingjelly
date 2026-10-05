from typing import Optional

import torch
import triton
import triton.language as tl

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
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    v = tl.load(V + n, mask, 0)
    w = tl.load(W + n, mask, 0)
    for t in range(T):
        i = t.to(tl.int64) * N + n
        current = tl.load(X + i, mask, 0).to(tl.float32)
        tl.store(P + i, v, mask)
        h = v + (current + a0 * (v - rest) * (v - critical) - w) / tau
        tl.store(H + i, h, mask)
        spike = (h >= threshold).to(tl.float32)
        w = w + (a * (h - rest) - w) / tau_w + b * spike
        if SOFT:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(S + i, spike, mask)
        if TRACE:
            tl.store(VO + i, v, mask)
            tl.store(WO + i, w, mask)
    if not TRACE:
        tl.store(VO + n, v, mask)
        tl.store(WO + n, w, mask)


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
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    cv = tl.full((BLOCK,), 0.0, tl.float32)
    cw = tl.full((BLOCK,), 0.0, tl.float32)
    for step in range(T):
        t = T - 1 - step
        i = t.to(tl.int64) * N + n
        h = tl.load(H + i, mask, 0)
        p = tl.load(P + i, mask, 0)
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
        if SOFT:
            dr = tl.full((BLOCK,), 1.0, tl.float32)
        else:
            dr = 1.0 - (h >= threshold).to(tl.float32)
        if not DETACH:
            if SOFT:
                dr = dr - threshold * sg
            else:
                dr = dr + (reset - h) * sg
        elif not SOFT:
            dr = dr + reset * sg
        if TRACE:
            iv = cv + tl.load(GV + i, mask, 0)
            iw = cw + tl.load(GW + i, mask, 0)
        elif step == 0:
            iv = cv + tl.load(GV + n, mask, 0)
            iw = cw + tl.load(GW + n, mask, 0)
        else:
            iv = cv
            iw = cw
        gh = tl.load(GS + i, mask, 0).to(tl.float32) * sg + iv * dr
        gh = gh + iw * (b * sg + a / tau_w)
        dh = 1.0 + a0 * (2.0 * p - rest - critical) / tau
        tl.store(GX + i, gh / tau, mask)
        cv = gh * dh
        cw = iw * (1.0 - 1.0 / tau_w) - gh / tau
    tl.store(GV0 + n, cv, mask)
    tl.store(GW0 + n, cw, mask)


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
    x, v, w = (x.contiguous(), v.contiguous(), w.contiguous())
    s = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    wo = torch.empty_like(vo)
    h = torch.empty_like(x, dtype=torch.float32)
    previous = torch.empty_like(h)
    with torch.cuda.device(x.device):
        (
            _forward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_forward_kernel)
        )[(triton.cdiv(v.numel(), 256),)](
            x,
            v,
            w,
            s,
            vo,
            wo,
            h,
            previous,
            x.shape[0],
            v.numel(),
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
    gs, gv, gw, h, previous = (
        gs.contiguous(),
        gv.contiguous(),
        gw.contiguous(),
        h.contiguous(),
        previous.contiguous(),
    )
    gx = torch.empty_like(gs)
    v0 = torch.empty_like(h[0])
    w0 = torch.empty_like(v0)
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[(triton.cdiv(v0.numel(), 256),)](
            gs,
            gv,
            gw,
            h,
            previous,
            gx,
            v0,
            w0,
            h.shape[0],
            v0.numel(),
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
            enable_fp_fusion=False,
        )
    return gx, v0, w0
