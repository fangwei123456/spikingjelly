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
from ..triton_surrogate import _surrogate_gradient
from .autograd import _check_backward, _check_forward


@triton.jit
def _forward_kernel(
    x_ptr,
    v_ptr,
    spikes_ptr,
    voltages_ptr,
    charged_ptr,
    T,
    N,
    q_ptr,
    threshold,
    reset,
    DECAY_INPUT: tl.constexpr,
    SOFT_RESET: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    # Inductor may pass Python floats as fp64; this operator's arithmetic is fp32.
    q = tl.load(q_ptr)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    SIZES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[0]
    STRIDES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[1]
    MINOR: tl.constexpr = 1 if LAYOUT is None else tl.constexpr(LAYOUT).value[2]
    if LAYOUT is None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = n < N
    else:
        n, mask = _neuron_indices(tl.constexpr(LAYOUT).value[3], BLOCK, SIZES, MINOR)
    x_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 0)
    v_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 1)
    spikes_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    voltages_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    charged_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    v = tl.load(v_ptr + v_ptr_offset, mask=mask, other=0.0)
    for t in range(T):
        x = tl.load(
            x_ptr + x_ptr_offset + _time_offset(t, N, STRIDES, 0), mask=mask, other=0.0
        ).to(tl.float32)
        if DECAY_INPUT:
            h = v + (x - (v - reset)) * q
        else:
            h = v - (v - reset) * q + x
        spike = (h >= threshold).to(tl.float32)
        if SOFT_RESET:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(
            spikes_ptr + spikes_ptr_offset + _time_offset(t, N, STRIDES, 2),
            spike,
            mask=mask,
        )
        if STORE_V_SEQ:
            tl.store(
                voltages_ptr + voltages_ptr_offset + _time_offset(t, N, STRIDES, 3),
                v,
                mask=mask,
            )
        tl.store(
            charged_ptr + charged_ptr_offset + _time_offset(t, N, STRIDES, 4),
            h,
            mask=mask,
        )

    if not STORE_V_SEQ:
        tl.store(voltages_ptr + voltages_ptr_offset, v, mask=mask)


@triton.jit
def _backward_kernel(
    gs_ptr,
    gv_ptr,
    x_ptr,
    v_ptr,
    h_ptr,
    gx_ptr,
    gv_init_ptr,
    gq_ptr,
    T,
    N,
    q_ptr,
    threshold,
    reset,
    alpha,
    DECAY_INPUT: tl.constexpr,
    SOFT_RESET: tl.constexpr,
    DETACH_RESET: tl.constexpr,
    SURROGATE: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    q = tl.load(q_ptr)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    alpha = tl.cast(alpha, tl.float32)
    SIZES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[0]
    STRIDES: tl.constexpr = () if LAYOUT is None else tl.constexpr(LAYOUT).value[1]
    MINOR: tl.constexpr = 1 if LAYOUT is None else tl.constexpr(LAYOUT).value[2]
    if LAYOUT is None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = n < N
    else:
        n, mask = _neuron_indices(tl.constexpr(LAYOUT).value[3], BLOCK, SIZES, MINOR)
    gs_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 0)
    gv_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 1)
    x_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    v_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    h_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    gx_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 5)
    gv_init_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 6)
    gq_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 7)
    carry = tl.full(n.shape, 0.0, tl.float32)
    gq = tl.full(n.shape, 0.0, tl.float32)
    previous_step = -N if LAYOUT is None else _time_offset(-1, N, STRIDES, 4)
    for step in range(T):
        t = T - 1 - step
        h_offset = _time_offset(t, N, STRIDES, 4) + h_ptr_offset
        h = tl.load(h_ptr + h_offset, mask=mask, other=0.0)
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
        if SOFT_RESET:
            reset_grad = tl.full(n.shape, 1.0, tl.float32)
        else:
            reset_grad = 1.0 - (h >= threshold).to(tl.float32)
        if not DETACH_RESET:
            if SOFT_RESET:
                reset_grad = reset_grad - threshold * sg
            else:
                reset_grad = reset_grad + (reset - h) * sg
        gs = tl.load(
            gs_ptr + gs_ptr_offset + _time_offset(t, N, STRIDES, 0),
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if STORE_V_SEQ:
            gv = tl.load(
                gv_ptr + gv_ptr_offset + _time_offset(t, N, STRIDES, 1),
                mask=mask,
                other=0.0,
            )
        elif step == 0:
            gv = tl.load(gv_ptr + gv_ptr_offset, mask=mask, other=0.0)
        else:
            gv = tl.full(n.shape, 0.0, tl.float32)
        gh = gs * sg + (gv + carry) * reset_grad
        gx = gh * q if DECAY_INPUT else gh
        tl.store(gx_ptr + gx_ptr_offset + _time_offset(t, N, STRIDES, 5), gx, mask=mask)
        carry = gh - gh * q
        previous = tl.load(v_ptr + v_ptr_offset, mask=mask, other=0.0)
        if T - 1 - step > 0:
            # Reuse the current address so the loop advances pointers directly.
            previous_h = tl.load(
                h_ptr + h_offset + previous_step,
                mask=mask,
                other=0.0,
            )
            spike = (previous_h >= threshold).to(tl.float32)
            if SOFT_RESET:
                previous = previous_h - spike * threshold
            else:
                previous = spike * reset + (1.0 - spike) * previous_h
        if DECAY_INPUT:
            x = tl.load(
                x_ptr + x_ptr_offset + _time_offset(t, N, STRIDES, 2),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            factor = x - (previous - reset)
        else:
            factor = -(previous - reset)
        gq = gq + gh * factor
    tl.store(gv_init_ptr + gv_init_ptr_offset, carry, mask=mask)
    tl.store(gq_ptr + gq_ptr_offset, gq, mask=mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    torch._check(x.device.type == "cuda", lambda: "Triton PLIF requires CUDA")
    q = w.float().sigmoid()
    spikes = torch.empty_like(x)
    voltages = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    charged = torch.empty_like(x, dtype=torch.float32)
    if x.is_contiguous() and v.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(x, x, v, spikes, voltages, charged)
        minor = _block_minor(sizes, strides)
    neurons = v.numel()
    with torch.cuda.device(x.device):
        (
            _forward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_forward_kernel)
        )[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            x,
            v,
            spikes,
            voltages,
            charged,
            x.shape[0],
            neurons,
            q,
            threshold,
            0.0 if reset is None else reset,
            decay_input,
            reset is None,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return spikes, voltages, charged


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, x, v, w, h, threshold, reset, alpha, store_v_seq, surrogate_id
    )
    torch._check(h.device.type == "cuda", lambda: "Triton PLIF requires CUDA")
    q = w.float().sigmoid()
    gx = torch.empty_like(h, dtype=gs.dtype)
    gv_init = torch.empty_like(v)
    gq = torch.empty_like(v)
    if (
        gs.is_contiguous()
        and gv.is_contiguous()
        and x.is_contiguous()
        and v.is_contiguous()
        and h.is_contiguous()
    ):
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(h, gs, gv, x, v, h, gx, gv_init, gq)
        minor = _block_minor(sizes, strides)
    neurons = gv_init.numel()
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[
            (triton.cdiv(neurons, 256),)
            if minor == 1
            else _neuron_grid(neurons, sizes, 256, minor)
        ](
            gs,
            gv,
            x,
            v,
            h,
            gx,
            gv_init,
            gq,
            h.shape[0],
            neurons,
            q,
            threshold,
            0.0 if reset is None else reset,
            alpha,
            decay_input,
            reset is None,
            detach_reset,
            surrogate_id,
            store_v_seq,
            256,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return gx, gv_init, (gq.sum() * q * (1 - q)).to(w.dtype)
