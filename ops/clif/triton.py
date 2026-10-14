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
def _decay_kernel(
    H, M, D, A, N, TAU: tl.constexpr, CACHE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    a = _round(_sigmoid(_round(tl.div_rn(h, TAU), dtype)), dtype)
    tl.store(D + i, m * a, mask)
    if CACHE:
        tl.store(A + i, a, mask)


@triton.jit
def _reset_kernel(
    H, S, M, V, B, N, THRESHOLD: tl.constexpr, CACHE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = H.dtype.element_ty
    h = tl.load(H + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    b = _round(_sigmoid(m), dtype)
    tl.store(V + i, h - _round(s * _round(THRESHOLD + b, dtype), dtype), mask)
    if CACHE:
        tl.store(B + i, b, mask)


@triton.jit
def _decay_backward_kernel(
    GD,
    GA,
    A,
    M,
    GH,
    GM,
    N,
    TAU: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = A.dtype.element_ty
    a = tl.load(A + i, mask, 0).to(tl.float32)
    m = tl.load(M + i, mask, 0).to(tl.float32)
    gd = tl.load(GD + i, mask, 0).to(tl.float32) if HAS_D else 0.0
    ga = _round(gd * m, dtype)
    if HAS_A:
        extra = tl.load(GA + i, mask, 0).to(tl.float32)
        ga = _round(ga + extra, dtype) if HAS_D else extra
    gq = _sigmoid_backward(ga, a, dtype)
    tl.store(GH + i, tl.div_rn(gq, TAU), mask)
    tl.store(GM + i, gd * a, mask)


@triton.jit
def _reset_backward_kernel(
    GV,
    GB,
    B,
    S,
    GS,
    GM,
    N,
    THRESHOLD: tl.constexpr,
    HAS_V: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dtype: tl.constexpr = B.dtype.element_ty
    b = tl.load(B + i, mask, 0).to(tl.float32)
    s = tl.load(S + i, mask, 0).to(tl.float32)
    gv = tl.load(GV + i, mask, 0).to(tl.float32) if HAS_V else 0.0
    gb = _round(-gv * s, dtype)
    if HAS_B:
        extra = tl.load(GB + i, mask, 0).to(tl.float32)
        gb = _round(gb + extra, dtype) if HAS_V else extra
    tl.store(GS + i, -gv * _round(THRESHOLD + b, dtype), mask)
    tl.store(GM + i, _sigmoid_backward(gb, b, dtype), mask)


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
        cache = any(ctx.needs_input_grad)
        d = torch.empty_like(h)
        a = torch.empty_like(h) if cache else None
        if h.numel():
            with torch.cuda.device(h.device):
                _decay_kernel[(triton.cdiv(h.numel(), 256),)](
                    h, m, d, a, h.numel(), tau, cache, 256, enable_fp_fusion=False
                )
        # Returning the cached sigmoid preserves its graph for double backward.
        ctx.save_for_backward(m, a)
        ctx.tau = tau
        ctx.set_materialize_grads(False)
        return d, a

    @staticmethod
    def backward(ctx, gd, ga):
        m, a = ctx.saved_tensors
        if torch.is_grad_enabled():
            g = ga if gd is None else gd * m if ga is None else gd * m + ga
            gm = torch.zeros_like(m) if gd is None else gd * a
            return torch.ops.aten.sigmoid_backward(g, a) / ctx.tau, gm, None
        gh, gm = torch.empty_like(a), torch.empty_like(a)
        if a.numel():
            with torch.cuda.device(a.device):
                _decay_backward_kernel[(triton.cdiv(a.numel(), 256),)](
                    None if gd is None else gd.contiguous(),
                    None if ga is None else ga.contiguous(),
                    a,
                    m,
                    gh,
                    gm,
                    a.numel(),
                    ctx.tau,
                    gd is not None,
                    ga is not None,
                    256,
                    enable_fp_fusion=False,
                )
        return gh, gm, None


class _Reset(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, spike, m, threshold):
        cache = any(ctx.needs_input_grad)
        v = torch.empty_like(h)
        b = torch.empty_like(h) if cache else None
        if h.numel():
            with torch.cuda.device(h.device):
                _reset_kernel[(triton.cdiv(h.numel(), 256),)](
                    h,
                    spike,
                    m,
                    v,
                    b,
                    h.numel(),
                    threshold,
                    cache,
                    256,
                    enable_fp_fusion=False,
                )
        ctx.save_for_backward(spike, b)
        ctx.threshold = threshold
        ctx.set_materialize_grads(False)
        return v, b

    @staticmethod
    def backward(ctx, gv, gb):
        spike, b = ctx.saved_tensors
        if torch.is_grad_enabled():
            g = gb if gv is None else -gv * spike if gb is None else -gv * spike + gb
            gs = torch.zeros_like(spike) if gv is None else -gv * (ctx.threshold + b)
            return gv, gs, torch.ops.aten.sigmoid_backward(g, b), None
        gs, gm = torch.empty_like(b), torch.empty_like(b)
        if b.numel():
            with torch.cuda.device(b.device):
                _reset_backward_kernel[(triton.cdiv(b.numel(), 256),)](
                    None if gv is None else gv.contiguous(),
                    None if gb is None else gb.contiguous(),
                    b,
                    spike,
                    gs,
                    gm,
                    b.numel(),
                    ctx.threshold,
                    gv is not None,
                    gb is not None,
                    256,
                    enable_fp_fusion=False,
                )
        return gv, gs, gm, None


def _forward_impl(x, v, m, tau, threshold, spike_function, *, _kernel_wrapper=None):
    # Separate edges preserve accumulation at v across recurrent time steps.
    v = v.contiguous()
    h = _Charge.apply(x.contiguous(), v, v, tau)
    decayed_m, _ = _Decay.apply(h, m.contiguous(), tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    # Keep gradient accumulation at the same branch points as the Torch graph.
    if spike.shape != h.shape or spike.dtype != h.dtype:
        return spike, cpu._reset(h, spike, m, threshold), m
    v, _ = _Reset.apply(h, spike.contiguous(), m, threshold)
    return spike, v, m
