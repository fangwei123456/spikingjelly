import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .autograd import _check


@triton.jit
def _forward_kernel(
    X,
    V,
    LEAK_V,
    M,
    S,
    VO,
    MO,
    H,
    N,
    TAU: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = X.dtype.element_ty
    x = tl.load(X + i, mask, 0).to(tl.float32)
    v = tl.load(V + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    leak_v = tl.load(LEAK_V + i, mask, 0).to(tl.float32)
    leak = tl.div_rn(leak_v, TAU).to(dtype).to(tl.float32)
    h = ((v - leak).to(dtype).to(tl.float32) + x).to(dtype).to(tl.float32)
    q = tl.div_rn(h, TAU).to(dtype).to(tl.float32)
    a = tl.div_rn(1.0, 1.0 + libdevice.exp(-q)).to(dtype).to(tl.float32)
    decayed_m = (m * a).to(dtype).to(tl.float32)
    z = (h - THRESHOLD).to(dtype).to(tl.float32)
    spike = (z >= 0).to(tl.float32)
    m_next = (decayed_m + spike).to(dtype).to(tl.float32)
    b = tl.div_rn(1.0, 1.0 + libdevice.exp(-m_next)).to(dtype).to(tl.float32)
    reset = (spike * (THRESHOLD + b).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(S + i, spike, mask)
    tl.store(VO + i, h - reset, mask)
    tl.store(MO + i, m_next, mask)
    tl.store(H + i, h, mask)


@triton.jit
def _backward_kernel(
    GS,
    GV,
    GM,
    H,
    M,
    MO,
    GX,
    GVI,
    GMI,
    N,
    TAU: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ALPHA: tl.constexpr,
    SURROGATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    gs = tl.load(GS + i, mask, 0).to(tl.float32)
    gv = tl.load(GV + i, mask, 0).to(tl.float32)
    gm = tl.load(GM + i, mask, 0).to(tl.float32)
    h = tl.load(H + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    m_next = tl.load(MO + i, mask, 0).to(tl.float32)
    z = (h - THRESHOLD).to(dtype).to(tl.float32)
    spike = (z >= 0).to(tl.float32)
    b = tl.div_rn(1.0, 1.0 + libdevice.exp(-m_next)).to(dtype).to(tl.float32)
    gs_reset = (-gv * (THRESHOLD + b).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    gm_reset = (-gv * spike).to(dtype).to(tl.float32)
    gm_reset = (gm_reset * (1.0 - b).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    gm_reset = (gm_reset * b).to(dtype).to(tl.float32)
    gm = (gm + gm_reset).to(dtype).to(tl.float32)
    gs = ((gs + gs_reset).to(dtype).to(tl.float32) + gm).to(dtype).to(tl.float32)
    q = tl.div_rn(h, TAU).to(dtype).to(tl.float32)
    a = tl.div_rn(1.0, 1.0 + libdevice.exp(-q)).to(dtype).to(tl.float32)
    gh_decay = (gm * m).to(dtype).to(tl.float32)
    gh_decay = (gh_decay * (1.0 - a).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    gh_decay = (gh_decay * a).to(dtype).to(tl.float32)
    gh_decay = tl.div_rn(gh_decay, TAU).to(dtype).to(tl.float32)
    if SURROGATE == 0:
        u = (z * ALPHA).to(dtype).to(tl.float32)
        s = tl.div_rn(1.0, 1.0 + libdevice.exp(-u)).to(dtype).to(tl.float32)
        gh_spike = (gs * (1.0 - s).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        gh_spike = (gh_spike * s).to(dtype).to(tl.float32)
        gh_spike = (gh_spike * ALPHA).to(dtype).to(tl.float32)
    elif SURROGATE == 1:
        u = (z * (1.5707963267948966 * ALPHA)).to(dtype).to(tl.float32)
        denominator = (1.0 + (u * u).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        coefficient = tl.div_rn(1.0, denominator).to(dtype).to(tl.float32)
        coefficient = (coefficient * (ALPHA / 2.0)).to(dtype).to(tl.float32)
        gh_spike = (coefficient * gs).to(dtype).to(tl.float32)
    elif SURROGATE == 2:
        coefficient = (-ALPHA * ALPHA * tl.abs(z)).to(dtype).to(tl.float32)
        coefficient = (coefficient + ALPHA).to(dtype).to(tl.float32)
        outside = tl.abs(z) > tl.full((), 1.0 / ALPHA, tl.float32).to(dtype).to(
            tl.float32
        )
        gh_spike = tl.where(outside, 0.0, (gs * coefficient).to(dtype).to(tl.float32))
    elif SURROGATE == 3:
        u = (-ALPHA * tl.abs(z)).to(dtype).to(tl.float32)
        coefficient = libdevice.exp(u).to(dtype).to(tl.float32)
        coefficient = ((ALPHA / 2.0) * coefficient).to(dtype).to(tl.float32)
        gh_spike = (coefficient * gs).to(dtype).to(tl.float32)
    elif SURROGATE == 4:
        denominator = (1.0 / ALPHA + tl.abs(z)).to(dtype).to(tl.float32)
        denominator = (denominator * denominator).to(dtype).to(tl.float32)
        denominator = (2.0 * ALPHA * denominator).to(dtype).to(tl.float32)
        gh_spike = tl.div_rn(gs, denominator).to(dtype).to(tl.float32)
    elif SURROGATE == 5:
        numerator = (ALPHA * gs).to(dtype).to(tl.float32)
        denominator = (tl.abs(z) + 1.0).to(dtype).to(tl.float32)
        denominator = (denominator * denominator).to(dtype).to(tl.float32)
        gh_spike = tl.div_rn(numerator, denominator).to(dtype).to(tl.float32)
    elif SURROGATE == 6:
        u = (z * ALPHA).to(dtype).to(tl.float32)
        u = (u * u).to(dtype).to(tl.float32)
        exponential = libdevice.exp(-u).to(dtype).to(tl.float32)
        gh_spike = (gs * exponential).to(dtype).to(tl.float32)
        gh_spike = (gh_spike * (ALPHA / 1.7724538509055159)).to(dtype).to(tl.float32)
    elif SURROGATE == 7:
        width = tl.full((), 0.5 / ALPHA, tl.float32).to(dtype).to(tl.float32)
        coefficient = (ALPHA * (tl.abs(z) < width)).to(dtype).to(tl.float32)
        gh_spike = (coefficient * gs).to(dtype).to(tl.float32)
    else:
        gh_spike = tl.full((BLOCK,), 0.0, tl.float32)
    gh = ((gv + gh_spike).to(dtype).to(tl.float32) + gh_decay).to(dtype).to(tl.float32)
    leak_grad = (gh * (-1.0 / TAU)).to(dtype).to(tl.float32)
    tl.store(GX + i, gh, mask)
    tl.store(GVI + i, leak_grad, mask)
    tl.store(GMI + i, gm * a, mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    leak_v: torch.Tensor,
    m: torch.Tensor,
    tau: float,
    threshold: float,
    alpha: float,
    surrogate_id: int,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, (v, leak_v, m), tau, threshold, alpha, surrogate_id)
    spike, v_next, m_next, h = (
        torch.empty_like(x, memory_format=torch.contiguous_format) for _ in range(4)
    )
    if x.numel():
        with torch.cuda.device(x.device):
            (
                _forward_kernel
                if _kernel_wrapper is None
                else _kernel_wrapper(_forward_kernel)
            )[(triton.cdiv(x.numel(), 256),)](
                x.contiguous(),
                v.contiguous(),
                leak_v.contiguous(),
                m.contiguous(),
                spike,
                v_next,
                m_next,
                h,
                x.numel(),
                tau,
                threshold,
                256,
                enable_fp_fusion=False,
            )
    return spike, v_next, m_next, h


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gm: torch.Tensor,
    h: torch.Tensor,
    m: torch.Tensor,
    m_next: torch.Tensor,
    tau: float,
    threshold: float,
    alpha: float,
    surrogate_id: int,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(h, (gs, gv, gm, m, m_next), tau, threshold, alpha, surrogate_id)
    gx, gv_init, gm_init = (
        torch.empty_like(h, memory_format=torch.contiguous_format) for _ in range(3)
    )
    if h.numel():
        with torch.cuda.device(h.device):
            (
                _backward_kernel
                if _kernel_wrapper is None
                else _kernel_wrapper(_backward_kernel)
            )[(triton.cdiv(h.numel(), 256),)](
                gs.contiguous(),
                gv.contiguous(),
                gm.contiguous(),
                h.contiguous(),
                m.contiguous(),
                m_next.contiguous(),
                gx,
                gv_init,
                gm_init,
                h.numel(),
                tau,
                threshold,
                alpha,
                surrogate_id,
                256,
                enable_fp_fusion=False,
            )
    return gx, gv_init, gm_init
