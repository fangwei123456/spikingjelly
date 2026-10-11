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
from .validation import _check


@triton.jit
def _kernel(
    X,
    V,
    W,
    TH,
    OFF,
    NEG,
    OUT,
    VO,
    WO,
    CUR,
    T,
    N,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
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
    OUT_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    VO_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    WO_offset = _spatial_offsets(n, SIZES, STRIDES, 5)
    CUR_offset = _spatial_offsets(n, SIZES, STRIDES, 6)
    th = tl.load(TH)
    off = tl.load(OFF)
    neg_bound = tl.load(NEG)
    v = tl.load(V + V_offset, mask, 0)
    w = tl.load(W + W_offset, mask, 0)
    cur = tl.full(n.shape, 0.0, tl.float32)
    for t in range(T):
        x = tl.load(X + X_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
            tl.float32
        )
        v = v + x / th
        w = libdevice.rint(w)
        pos = (v >= 1.0) & (w < off)
        neg = (v < 0.0) & (w > neg_bound)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        w = w + cur
        v = v - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(OUT + OUT_offset + _time_offset(t, N, STRIDES, 3), cur * th, mask)
    tl.store(VO + VO_offset, v, mask)
    tl.store(WO + WO_offset, w, mask)
    tl.store(CUR + CUR_offset, cur, mask)


def _forward_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    torch._check(x.is_cuda, lambda: "Triton stbif requires CUDA")
    out = torch.empty_like(x)
    vo = torch.empty_like(q)
    wo = torch.empty_like(q)
    cur = torch.empty_like(q)
    if x.is_contiguous() and q.is_contiguous() and acc_q.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(x, x, q, acc_q, out, vo, wo, cur)
        minor = _block_minor(sizes, strides)
    neurons = q.numel()
    with torch.cuda.device(x.device):
        (_kernel if _kernel_wrapper is None else _kernel_wrapper(_kernel))[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            x,
            q,
            acc_q,
            q_threshold,
            pos_max,
            neg_min,
            out,
            vo,
            wo,
            cur,
            x.shape[0],
            neurons,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return out, vo, wo, cur
