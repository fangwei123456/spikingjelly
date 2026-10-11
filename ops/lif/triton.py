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
from ..triton_runtime import use_static_range_for_triton_neuron_kernel
from ..triton_surrogate import _surrogate_gradient
from .autograd import _check_backward, _check_forward


@triton.jit
def _forward_kernel(
    x_ptr,
    v_ptr,
    spikes_ptr,
    voltages_ptr,
    charged_ptr,
    T: tl.constexpr,
    N,
    tau,
    threshold,
    reset,
    DECAY_INPUT: tl.constexpr,
    SOFT_RESET: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr = None,
):
    # Inductor may pass Python floats as fp64; this operator's arithmetic is fp32.
    tau = tl.cast(tau, tl.float32)
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
            h = v + tl.div_rn(x - (v - reset), tau)
        else:
            h = v - tl.div_rn(v - reset, tau) + x
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
    h_ptr,
    gx_ptr,
    gv_init_ptr,
    T: tl.constexpr,
    N,
    tau,
    threshold,
    reset,
    alpha,
    DECAY_INPUT: tl.constexpr,
    SOFT_RESET: tl.constexpr,
    DETACH_RESET: tl.constexpr,
    SURROGATE: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    UNROLL: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYOUT: tl.constexpr = None,
):
    tau = tl.cast(tau, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    alpha = tl.cast(alpha, tl.float32)
    r_tau = 1.0 / tau
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
    h_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 2)
    gx_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 3)
    gv_init_ptr_offset = _spatial_offsets(n, SIZES, STRIDES, 4)
    if STORE_V_SEQ:
        carry = tl.full(n.shape, 0.0, tl.float32)
    else:
        # Seed the recurrence before the loop; its first-step load breaks
        # Triton 3.6 coalescing when lowered inside the temporal branch.
        carry = tl.load(gv_ptr + gv_ptr_offset, mask=mask, other=0.0)
    for step in tl.range(T, loop_unroll_factor=UNROLL):
        t = T - 1 - step
        h = tl.load(
            h_ptr + h_ptr_offset + _time_offset(t, N, STRIDES, 2), mask=mask, other=0.0
        )
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
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
            grad_v = gv + carry
        else:
            grad_v = carry
        if SOFT_RESET:
            if DETACH_RESET:
                gh = tl.fma(gs, sg, grad_v)
            else:
                gh = tl.fma(gs - threshold * grad_v, sg, grad_v)
        else:
            spike = (h >= threshold).to(tl.float32)
            if DETACH_RESET:
                gh = tl.fma(gs, sg, grad_v * (1.0 - spike))
            else:
                gh = tl.fma(
                    tl.fma(grad_v, reset - h, gs),
                    sg,
                    grad_v * (1.0 - spike),
                )
        gx = gh * r_tau if DECAY_INPUT else gh
        tl.store(gx_ptr + gx_ptr_offset + _time_offset(t, N, STRIDES, 3), gx, mask=mask)
        carry = gh * (1.0 - r_tau)
    tl.store(gv_init_ptr + gv_init_ptr_offset, carry, mask=mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
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
    _check_forward(x, v, tau, threshold, reset, alpha, surrogate_id)
    torch._check(x.device.type == "cuda", lambda: "Triton LIF requires CUDA")
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
            tau,
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
    h: torch.Tensor,
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(gs, gv, h, tau, threshold, reset, alpha, store_v_seq, surrogate_id)
    torch._check(h.device.type == "cuda", lambda: "Triton LIF requires CUDA")
    gx = torch.empty_like(h, dtype=gs.dtype)
    gv_init = torch.empty_like(h[0])
    neurons = gv_init.numel()
    block = (
        512
        if (
            surrogate_id == 1
            and h.shape[0] > 1
            and neurons >= 2097152
            and not store_v_seq
        )
        else 256
    )
    unroll = h.shape[0] if use_static_range_for_triton_neuron_kernel(h.shape[0]) else 1
    if gs.is_contiguous() and gv.is_contiguous() and h.is_contiguous():
        sizes, strides, minor = (), (), 1
    else:
        sizes, strides = _triton_layout_args(h, gs, gv, h, gx, gv_init)
        minor = _block_minor(sizes, strides)
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[
            (triton.cdiv(neurons, block),)
            if minor == 1
            else _neuron_grid(neurons, sizes, block, minor)
        ](
            gs,
            gv,
            h,
            gx,
            gv_init,
            h.shape[0],
            neurons,
            tau,
            threshold,
            0.0 if reset is None else reset,
            alpha,
            decay_input,
            reset is None,
            detach_reset,
            surrogate_id,
            store_v_seq,
            unroll,
            block,
            (sizes, strides, minor, int(neurons)) if strides else None,
            enable_fp_fusion=False,
        )
    return gx, gv_init
