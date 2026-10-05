import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
):
    tau = tl.cast(tau, tl.float32)
    count = tl.cast(count, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    v = tl.load(V + n, mask, 0)
    for t in range(T):
        i = t.to(tl.int64) * N + n
        current = tl.load(X + i, mask, 0).to(tl.float32)
        h = (1.0 - 1.0 / tau) * v + current
        # Keep FP32 workspace stores before low-precision spikes: Triton 3.3's
        # loop layout conversion otherwise fails for FP16 voltage traces.
        tl.store(H + i, h, mask)
        spike = libdevice.rint(
            tl.minimum(tl.maximum(tl.div_rn(h, threshold), 0.0), count)
        )
        v = h - spike * threshold
        tl.store(S + i, spike, mask)
        if TRACE:
            tl.store(VO + i, v, mask)
    if not TRACE:
        tl.store(VO + n, v, mask)


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
):
    tau = tl.cast(tau, tl.float32)
    lower = tl.cast(lower, tl.float32)
    upper = tl.cast(upper, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    cv = tl.full((BLOCK,), 0.0, tl.float32)
    for step in range(T):
        t = T - 1 - step
        i = t.to(tl.int64) * N + n
        h = tl.load(H + i, mask, 0)
        scaled = tl.div_rn(h, threshold)
        sg = tl.where((scaled >= lower) & (scaled <= upper), 1.0 / threshold, 0.0)
        dr = tl.full((BLOCK,), 1.0, tl.float32)
        if not DETACH:
            dr = dr - threshold * sg
        if TRACE:
            iv = cv + tl.load(GV + i, mask, 0)
        elif step == 0:
            iv = cv + tl.load(GV + n, mask, 0)
        else:
            iv = cv
        gh = tl.load(GS + i, mask, 0).to(tl.float32) * sg + iv * dr
        dh = 1.0 - 1.0 / tau
        tl.store(GX + i, gh, mask)
        cv = gh * dh
    tl.store(GV0 + n, cv, mask)


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
    x, v = (x.contiguous(), v.contiguous())
    s = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    h = torch.empty_like(x, dtype=torch.float32)
    with torch.cuda.device(x.device):
        (
            _forward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_forward_kernel)
        )[(triton.cdiv(v.numel(), 256),)](
            x,
            v,
            s,
            vo,
            h,
            x.shape[0],
            v.numel(),
            tau,
            count,
            threshold,
            store_v_seq,
            256,
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
    gs, gv, h = (gs.contiguous(), gv.contiguous(), h.contiguous())
    gx = torch.empty_like(gs)
    v0 = torch.empty_like(h[0])
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[(triton.cdiv(v0.numel(), 256),)](
            gs,
            gv,
            h,
            gx,
            v0,
            h.shape[0],
            v0.numel(),
            tau,
            lower,
            upper,
            threshold,
            detach_reset,
            store_v_seq,
            256,
            enable_fp_fusion=False,
        )
    return gx, v0
