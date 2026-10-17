import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import reference as _reference


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
def _charge_kernel(X, V, LEAK_V, H, N, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = X.dtype.element_ty
    x = tl.load(X + i, mask, 0).to(tl.float32)
    v = tl.load(V + i, mask, 0).to(tl.float32)
    leak_v = tl.load(LEAK_V + i, mask, 0).to(tl.float32)
    h = _round(v - _round(tl.div_rn(leak_v, TAU), dtype), dtype) + x
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


class _Charge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, leak_v, tau):
        ctx.tau = tau
        h = torch.empty_like(x)
        if x.numel():
            with torch.cuda.device(x.device):
                _charge_kernel[triton.cdiv(x.numel(), 256),](
                    x, v, leak_v, h, x.numel(), tau, 256, enable_fp_fusion=False
                )
        return h

    @staticmethod
    def backward(ctx, g):
        return g, g, g / -ctx.tau, None


class _Decay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, m, tau):
        ctx.save_for_backward(h, m)
        ctx.tau = tau
        d = torch.empty_like(h)
        if h.numel():
            with torch.cuda.device(h.device):
                _decay_kernel[triton.cdiv(h.numel(), 256),](
                    h, m, d, h.numel(), tau, 256, enable_fp_fusion=False
                )
        return d

    @staticmethod
    def backward(ctx, gd):
        h, m = ctx.saved_tensors
        if torch.is_grad_enabled():
            a = torch.sigmoid(h / ctx.tau)
            return (torch.ops.aten.sigmoid_backward(gd * m, a) / ctx.tau, gd * a, None)
        gh, gm = (torch.empty_like(h), torch.empty_like(h))
        if h.numel():
            with torch.cuda.device(h.device):
                _decay_backward_kernel[triton.cdiv(h.numel(), 256),](
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
        return (gh, gm, None)


class _Reset(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, spike, m, threshold):
        ctx.save_for_backward(spike, m)
        ctx.threshold = threshold
        v = torch.empty_like(h)
        if h.numel():
            with torch.cuda.device(h.device):
                _reset_kernel[triton.cdiv(h.numel(), 256),](
                    h, spike, m, v, h.numel(), threshold, 256, enable_fp_fusion=False
                )
        return v

    @staticmethod
    def backward(ctx, gv):
        spike, m = ctx.saved_tensors
        if torch.is_grad_enabled():
            b = torch.sigmoid(m)
            return (
                gv,
                -gv * (ctx.threshold + b),
                torch.ops.aten.sigmoid_backward(-gv * spike, b),
                None,
            )
        gs, gm = (torch.empty_like(m), torch.empty_like(m))
        if m.numel():
            with torch.cuda.device(m.device):
                _reset_backward_kernel[triton.cdiv(m.numel(), 256),](
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
        return (gv, gs, gm, None)


def _forward_impl(x, v, m, tau, threshold, spike_function, *, _kernel_wrapper=None):
    # Separate edges preserve accumulation at v across recurrent time steps.
    v = v.contiguous()
    h = _Charge.apply(x.contiguous(), v, v, tau)
    decayed_m = _Decay.apply(h, m.contiguous(), tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    # Keep gradient accumulation at the same branch points as the Torch graph.
    if spike.shape != h.shape or spike.dtype != h.dtype:
        return spike, _reference._reset(h, spike, m, threshold), m
    v = _Reset.apply(h, spike.contiguous(), m, threshold)
    return spike, v, m
