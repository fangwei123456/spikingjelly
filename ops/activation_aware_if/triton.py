from typing import Optional

import torch
import triton
import triton.language as tl

from .validation import _check


@triton.jit
def _kernel(
    X,
    V,
    TH,
    OFF,
    OUT,
    VO,
    T,
    N,
    CHANNELS: tl.constexpr,
    INNER: tl.constexpr,
    SCALAR_TH: tl.constexpr,
    SCALAR_OFF: tl.constexpr,
    reset,
    SOFT: tl.constexpr,
    TRACE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    c = n // INNER % CHANNELS
    if SCALAR_TH:
        th = tl.load(TH)
    else:
        th = tl.load(TH + c, mask, 1.0)
    if SCALAR_OFF:
        off = tl.load(OFF)
    else:
        off = tl.load(OFF + c, mask, 0.0)
    v = tl.load(V + n, mask, 0)
    cur = tl.full((BLOCK,), 0.0, tl.float32)
    reset = tl.cast(reset, tl.float32)
    for t in range(T):
        i = t.to(tl.int64) * N + n
        x = tl.load(X + i, mask, 0).to(tl.float32)
        h = v + x
        cur = (h + off >= th).to(tl.float32)
        if SOFT:
            v = h - cur * th
        else:
            v = cur * reset + (1.0 - cur) * h
        tl.store(OUT + i, cur, mask)
        if TRACE:
            tl.store(VO + i, v, mask)
    if not TRACE:
        tl.store(VO + n, v, mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channels: int,
    inner: int,
    reset: Optional[float],
    store_v_seq: bool,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, v, threshold, offset, channels, inner, reset, store_v_seq)
    torch._check(x.is_cuda, lambda: "Triton activation_aware_if requires CUDA")
    x, v, threshold, offset = (
        x.contiguous(),
        v.contiguous(),
        threshold.contiguous(),
        offset.contiguous(),
    )
    out = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    with torch.cuda.device(x.device):
        (_kernel if _kernel_wrapper is None else _kernel_wrapper(_kernel))[
            (triton.cdiv(v.numel(), 256),)
        ](
            x,
            v,
            threshold,
            offset,
            out,
            vo,
            x.shape[0],
            v.numel(),
            channels,
            inner,
            threshold.numel() == 1,
            offset.numel() == 1,
            reset or 0.0,
            reset is None,
            store_v_seq,
            256,
            enable_fp_fusion=False,
        )
    return out, vo
