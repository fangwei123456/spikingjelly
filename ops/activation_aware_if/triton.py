import math
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
    LAYOUT: tl.constexpr,
):
    PARAMETERS: tl.constexpr = None if LAYOUT is None else tl.constexpr(LAYOUT).value[5]
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
    OUT_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    VO_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    LOGICAL: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[4]
    logical = _spatial_offsets(n, SIZES, LOGICAL)
    c = logical // INNER % CHANNELS
    if SCALAR_TH:
        th = tl.load(TH)
    elif PARAMETERS is None:
        th = tl.load(TH + c, mask, 1.0)
    else:
        TH_SIZES: tl.constexpr = tl.constexpr(PARAMETERS).value[0]
        TH_STRIDES: tl.constexpr = tl.constexpr(PARAMETERS).value[1]
        th = tl.load(
            TH + _spatial_offsets(c, TH_SIZES, TH_STRIDES),
            mask,
            1.0,
        )
    if SCALAR_OFF:
        off = tl.load(OFF)
    elif PARAMETERS is None:
        off = tl.load(OFF + c, mask, 0.0)
    else:
        OFF_SIZES: tl.constexpr = tl.constexpr(PARAMETERS).value[2]
        OFF_STRIDES: tl.constexpr = tl.constexpr(PARAMETERS).value[3]
        off = tl.load(
            OFF + _spatial_offsets(c, OFF_SIZES, OFF_STRIDES),
            mask,
            0.0,
        )
    v = tl.load(V + V_offset, mask, 0)
    reset = tl.cast(reset, tl.float32)
    for t in range(T):
        x = tl.load(X + X_offset + _time_offset(t, N, STRIDES, 0), mask, 0).to(
            tl.float32
        )
        h = v + x
        cur = (h + off >= th).to(tl.float32)
        if SOFT:
            v = h - cur * th
        else:
            v = cur * reset + (1.0 - cur) * h
        tl.store(OUT + OUT_offset + _time_offset(t, N, STRIDES, 2), cur, mask)
        if TRACE:
            tl.store(VO + VO_offset + _time_offset(t, N, STRIDES, 3), v, mask)
    if not TRACE:
        tl.store(VO + VO_offset, v, mask)


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
    out = torch.empty_like(x)
    vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    if x.is_contiguous() and v.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(x, x, v, out, vo)
        minor = _block_minor(sizes, strides)
    if sizes:
        order = sorted(range(1, x.ndim), key=x.stride().__getitem__)
        logical = ((0, *(math.prod(x.shape[d + 1 :]) for d in order)),)
    else:
        logical = ()
    parameters = None
    if not (threshold.is_contiguous() and offset.is_contiguous()):
        parameters = (
            tuple(reversed(threshold.shape)),
            ((0, *reversed(threshold.stride())),),
            tuple(reversed(offset.shape)),
            ((0, *reversed(offset.stride())),),
        )
    neurons = v.numel()
    with torch.cuda.device(x.get_device()):
        (_kernel if _kernel_wrapper is None else _kernel_wrapper(_kernel))[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            x,
            v,
            threshold,
            offset,
            out,
            vo,
            x.shape[0],
            neurons,
            channels,
            inner,
            threshold.numel() == 1,
            offset.numel() == 1,
            reset or 0.0,
            reset is None,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons), logical, parameters)
            if strides or parameters is not None
            else None,
            enable_fp_fusion=False,
        )
    return out, vo
