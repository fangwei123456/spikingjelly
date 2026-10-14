import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import cpu


@triton.jit
def _round(x, DTYPE: tl.constexpr):
    return x.to(DTYPE).to(tl.float32)


@triton.jit
def _sigmoid(x):
    return tl.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _sigmoid_backward(g, s, DTYPE: tl.constexpr):
    return _round(_round(g * _round(1.0 - s, DTYPE), DTYPE) * s, DTYPE)


@triton.jit
def _charge_kernel(X, V, H, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = X.dtype.element_ty
    x = tl.load(X + i, mask, 0).to(tl.float32)
    v = tl.load(V + i, mask, 0).to(tl.float32)
    h = _round(v - _round(tl.div_rn(v, TAU), dtype), dtype) + x
    tl.store(H + i, h, mask)


@triton.jit
def _decay_kernel(H, M, D, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    a = _round(_sigmoid(_round(tl.div_rn(h, TAU), dtype)), dtype)
    tl.store(D + i, m * a, mask)


@triton.jit
def _reset_kernel(H, S, M, V, N, THRESHOLD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    b = _round(_sigmoid(m), dtype)
    tl.store(V + i, h - _round(s * _round(THRESHOLD + b, dtype), dtype), mask)


@triton.jit
def _charge_backward_kernel(G, GV, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = G.dtype.element_ty
    g = tl.load(G + i, mask, 0).to(tl.float32)
    tl.store(GV + i, g + _round(tl.div_rn(-g, TAU), dtype), mask)


@triton.jit
def _decay_backward_kernel(GD, H, M, GH, GM, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    gd = tl.load(GD + i, mask, 0).to(tl.float32)
    a = _round(_sigmoid(_round(tl.div_rn(h, TAU), dtype)), dtype)
    gq = _sigmoid_backward(_round(gd * m, dtype), a, dtype)
    tl.store(GH + i, tl.div_rn(gq, TAU), mask)
    tl.store(GM + i, gd * a, mask)


@triton.jit
def _reset_backward_kernel(
    GV, M, S, GS, GM, N, THRESHOLD: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = M.dtype.element_ty
    m = tl.load(M + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    gv = tl.load(GV + i, mask, 0).to(tl.float32)
    b = _round(_sigmoid(m), dtype)
    tl.store(GS + i, -gv * _round(THRESHOLD + b, dtype), mask)
    tl.store(GM + i, _sigmoid_backward(_round(-gv * s, dtype), b, dtype), mask)


@torch.library.custom_op("sj_clif::charge", mutates_args=(), device_types="cuda")
def _charge(x: torch.Tensor, v: torch.Tensor, tau: float) -> torch.Tensor:
    h = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            _charge_kernel[(triton.cdiv(x.numel(), 256),)](
                x, v, h, x.numel(), tau, 256, enable_fp_fusion=False
            )
    return h


@_charge.register_fake
def _charge_fake(x, v, tau):
    return torch.empty_like(x)


def _charge_setup(ctx, inputs, output):
    ctx.tau = inputs[-1]


def _charge_backward(ctx, g):
    if torch.is_grad_enabled():
        return g, g - g / ctx.tau, None
    gv = torch.empty_like(g)
    if g.numel():
        with torch.cuda.device(g.device):
            _charge_backward_kernel[(triton.cdiv(g.numel(), 256),)](
                g.contiguous(), gv, g.numel(), ctx.tau, 256, enable_fp_fusion=False
            )
    return g, gv, None


_charge.register_autograd(_charge_backward, setup_context=_charge_setup)


@torch.library.custom_op("sj_clif::decay", mutates_args=(), device_types="cuda")
def _decay(h: torch.Tensor, m: torch.Tensor, tau: float) -> torch.Tensor:
    d = torch.empty_like(h)
    if h.numel():
        with torch.cuda.device(h.device):
            _decay_kernel[(triton.cdiv(h.numel(), 256),)](
                h, m, d, h.numel(), tau, 256, enable_fp_fusion=False
            )
    return d


@_decay.register_fake
def _decay_fake(h, m, tau):
    return torch.empty_like(h)


def _decay_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:2])
    ctx.tau = inputs[-1]


def _decay_backward(ctx, gd):
    h, m = ctx.saved_tensors
    if torch.is_grad_enabled():
        a = torch.sigmoid(h / ctx.tau)
        return torch.ops.aten.sigmoid_backward(gd * m, a) / ctx.tau, gd * a, None
    gh, gm = torch.empty_like(h), torch.empty_like(h)
    if h.numel():
        with torch.cuda.device(h.device):
            _decay_backward_kernel[(triton.cdiv(h.numel(), 256),)](
                gd.contiguous(),
                h,
                m,
                gh,
                gm,
                h.numel(),
                ctx.tau,
                256,
                enable_fp_fusion=False,
            )
    return gh, gm, None


_decay.register_autograd(_decay_backward, setup_context=_decay_setup)


@torch.library.custom_op("sj_clif::reset", mutates_args=(), device_types="cuda")
def _reset(
    h: torch.Tensor, spike: torch.Tensor, m: torch.Tensor, threshold: float
) -> torch.Tensor:
    v = torch.empty_like(h)
    if h.numel():
        with torch.cuda.device(h.device):
            _reset_kernel[(triton.cdiv(h.numel(), 256),)](
                h, spike, m, v, h.numel(), threshold, 256, enable_fp_fusion=False
            )
    return v


@_reset.register_fake
def _reset_fake(h, spike, m, threshold):
    return torch.empty_like(h)


def _reset_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[1], inputs[2])
    ctx.threshold = inputs[-1]


def _reset_backward(ctx, gv):
    spike, m = ctx.saved_tensors
    if torch.is_grad_enabled():
        b = torch.sigmoid(m)
        return (
            gv,
            -gv * (ctx.threshold + b),
            torch.ops.aten.sigmoid_backward(-gv * spike, b),
            None,
        )
    gs, gm = torch.empty_like(m), torch.empty_like(m)
    if m.numel():
        with torch.cuda.device(m.device):
            _reset_backward_kernel[(triton.cdiv(m.numel(), 256),)](
                gv.contiguous(),
                m,
                spike,
                gs,
                gm,
                m.numel(),
                ctx.threshold,
                256,
                enable_fp_fusion=False,
            )
    return gv, gs, gm, None


_reset.register_autograd(_reset_backward, setup_context=_reset_setup)


def _forward_impl(x, v, m, tau, threshold, spike_function, *, _kernel_wrapper=None):
    h = _charge(x.contiguous(), v.contiguous(), tau)
    decayed_m = _decay(h, m.contiguous(), tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    # Keep gradient accumulation at the same branch points as the Torch graph.
    if spike.shape != h.shape or spike.dtype != h.dtype:
        return spike, cpu._reset(h, spike, m, threshold), m
    v = _reset(h, spike.contiguous(), m, threshold)
    return spike, v, m
