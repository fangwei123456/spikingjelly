import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..autograd import _higher_order_grad, _save_for_higher_order
from . import cpu


@triton.jit
def _round(x, DTYPE: tl.constexpr):
    return x.to(DTYPE).to(tl.float32)


@triton.jit
def _sigmoid(x):
    return tl.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _charge_kernel(X, V, M, H, D, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = X.dtype.element_ty
    x = tl.load(X + i, mask, 0).to(tl.float32)
    v = tl.load(V + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    h = _round(_round(v - _round(tl.div_rn(v, TAU), dtype), dtype) + x, dtype)
    a = _round(_sigmoid(_round(tl.div_rn(h, TAU), dtype)), dtype)
    tl.store(H + i, h, mask)
    tl.store(D + i, m * a, mask)


@triton.jit
def _reset_kernel(H, D, S, V, M, N, THRESHOLD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    d = tl.load(D + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    m = _round(d + s, dtype)
    b = _round(_sigmoid(m), dtype)
    v = h - _round(s * _round(THRESHOLD + b, dtype), dtype)
    tl.store(V + i, v, mask)
    tl.store(M + i, m, mask)


@triton.jit
def _charge_backward_kernel(
    GH, GD, H, M, GX, GV, GM, N, TAU: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    gh = tl.load(GH + i, mask, 0).to(tl.float32)
    gd = tl.load(GD + i, mask, 0).to(tl.float32)
    a = _round(_sigmoid(_round(tl.div_rn(h, TAU), dtype)), dtype)
    ga = _round(gd * m, dtype)
    gq = _round(ga * (1.0 - a) * a, dtype)
    g = _round(gh + _round(tl.div_rn(gq, TAU), dtype), dtype)
    tl.store(GX + i, g, mask)
    tl.store(GV + i, g + _round(tl.div_rn(-g, TAU), dtype), mask)
    tl.store(GM + i, gd * a, mask)


@triton.jit
def _reset_backward_kernel(
    GV, GM, M, S, GH, GD, GS, N, THRESHOLD: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = M.dtype.element_ty
    m = tl.load(M + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    gv = tl.load(GV + i, mask, 0).to(tl.float32)
    gm = tl.load(GM + i, mask, 0).to(tl.float32)
    b = _round(_sigmoid(m), dtype)
    gb = _round(-gv * s, dtype)
    incoming_m = _round(gm + _round(gb * (1.0 - b) * b, dtype), dtype)
    gs = _round(-gv * _round(THRESHOLD + b, dtype), dtype) + incoming_m
    tl.store(GH + i, gv, mask)
    tl.store(GD + i, incoming_m, mask)
    tl.store(GS + i, gs, mask)


@torch.library.custom_op("sj_clif::charge", mutates_args=(), device_types="cuda")
def _charge(
    x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    h, d = torch.empty_like(x), torch.empty_like(x)
    if x.numel() == 0:
        return h, d
    with torch.cuda.device(x.device):
        _charge_kernel[(triton.cdiv(x.numel(), 256),)](
            x, v, m, h, d, x.numel(), tau, 256, enable_fp_fusion=False
        )
    return h, d


@_charge.register_fake
def _charge_fake(x, v, m, tau):
    return torch.empty_like(x), torch.empty_like(x)


def _charge_setup(ctx, inputs, output):
    _save_for_higher_order(ctx, inputs, (output[0],))
    ctx.set_materialize_grads(False)


def _charge_backward(ctx, gh, gd):
    if torch.is_grad_enabled():
        return _higher_order_grad(ctx, cpu._charge, (gh, gd))
    h, x, v, m = ctx.saved_tensors
    gx, gv, gm = (torch.empty_like(x) for _ in range(3))
    if h.numel() == 0:
        return gx, gv, gm, None
    with torch.cuda.device(h.device):
        _charge_backward_kernel[(triton.cdiv(h.numel(), 256),)](
            torch.zeros_like(h) if gh is None else gh.contiguous(),
            torch.zeros_like(h) if gd is None else gd.contiguous(),
            h,
            m,
            gx,
            gv,
            gm,
            h.numel(),
            ctx.input_template[-1],
            256,
            enable_fp_fusion=False,
        )
    return gx, gv, gm, None


_charge.register_autograd(_charge_backward, setup_context=_charge_setup)


@torch.library.custom_op("sj_clif::reset", mutates_args=(), device_types="cuda")
def _reset(
    h: torch.Tensor, d: torch.Tensor, spike: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    v, m = torch.empty_like(h), torch.empty_like(h)
    if h.numel() == 0:
        return v, m
    with torch.cuda.device(h.device):
        _reset_kernel[(triton.cdiv(h.numel(), 256),)](
            h, d, spike, v, m, h.numel(), threshold, 256, enable_fp_fusion=False
        )
    return v, m


@_reset.register_fake
def _reset_fake(h, d, spike, threshold):
    return torch.empty_like(h), torch.empty_like(h)


def _reset_setup(ctx, inputs, output):
    _save_for_higher_order(ctx, inputs, (output[1],))
    ctx.set_materialize_grads(False)


def _reset_backward(ctx, gv, gm):
    if torch.is_grad_enabled():
        return _higher_order_grad(ctx, cpu._reset, (gv, gm))
    m, h, d, spike = ctx.saved_tensors
    gh, gd, gs = (torch.empty_like(h) for _ in range(3))
    if h.numel() == 0:
        return gh, gd, gs, None
    with torch.cuda.device(h.device):
        _reset_backward_kernel[(triton.cdiv(h.numel(), 256),)](
            torch.zeros_like(m) if gv is None else gv.contiguous(),
            torch.zeros_like(m) if gm is None else gm.contiguous(),
            m,
            spike,
            gh,
            gd,
            gs,
            m.numel(),
            ctx.input_template[-1],
            256,
            enable_fp_fusion=False,
        )
    return gh, gd, gs, None


_reset.register_autograd(_reset_backward, setup_context=_reset_setup)


def _forward_impl(x, v, m, tau, threshold, spike_function, *, _kernel_wrapper=None):
    h, d = _charge(x.contiguous(), v.contiguous(), m.contiguous(), tau)
    spike = spike_function(h - threshold)
    # The callable may return a broadcast view or a different dtype.
    if spike.shape != h.shape or spike.dtype != h.dtype:
        return spike, *cpu._reset(h, d, spike, threshold)
    v, m = _reset(h, d, spike.contiguous(), threshold)
    return spike, v, m
