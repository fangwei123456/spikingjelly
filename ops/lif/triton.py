from typing import Optional

import torch
import triton
import triton.language as tl

from ..triton_surrogate import _surrogate_gradient
from ..triton_runtime import use_static_range_for_triton_neuron_kernel
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
):
    # Inductor may pass Python floats as fp64; this operator's arithmetic is fp32.
    tau = tl.cast(tau, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    v = tl.load(v_ptr + n, mask=mask, other=0.0)
    for t in range(T):
        offset = t.to(tl.int64) * N + n
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if DECAY_INPUT:
            h = v + tl.div_rn(x - (v - reset), tau)
        else:
            h = v - tl.div_rn(v - reset, tau) + x
        spike = (h >= threshold).to(tl.float32)
        if SOFT_RESET:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(spikes_ptr + offset, spike, mask=mask)
        if STORE_V_SEQ:
            tl.store(voltages_ptr + offset, v, mask=mask)
        tl.store(charged_ptr + offset, h, mask=mask)

    if not STORE_V_SEQ:
        tl.store(voltages_ptr + n, v, mask=mask)


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
):
    tau = tl.cast(tau, tl.float32)
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    alpha = tl.cast(alpha, tl.float32)
    r_tau = 1.0 / tau
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    if STORE_V_SEQ:
        carry = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        # Seed the recurrence before the loop; its first-step load breaks
        # Triton 3.6 coalescing when lowered inside the temporal branch.
        carry = tl.load(gv_ptr + n, mask=mask, other=0.0)
    for step in tl.range(T, loop_unroll_factor=UNROLL):
        offset = (T - 1 - step).to(tl.int64) * N + n
        h = tl.load(h_ptr + offset, mask=mask, other=0.0)
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
        gs = tl.load(gs_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if STORE_V_SEQ:
            gv = tl.load(gv_ptr + offset, mask=mask, other=0.0)
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
        tl.store(gx_ptr + offset, gx, mask=mask)
        carry = gh * (1.0 - r_tau)
    tl.store(gv_init_ptr + n, carry, mask=mask)


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
    x, v = x.contiguous(), v.contiguous()
    spikes = torch.empty_like(x)
    voltages = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
    charged = torch.empty_like(x, dtype=torch.float32)
    with torch.cuda.device(x.device):
        (
            _forward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_forward_kernel)
        )[(triton.cdiv(v.numel(), 256),)](
            x,
            v,
            spikes,
            voltages,
            charged,
            x.shape[0],
            v.numel(),
            tau,
            threshold,
            0.0 if reset is None else reset,
            decay_input,
            reset is None,
            store_v_seq,
            256,
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
    gs, gv, h = gs.contiguous(), gv.contiguous(), h.contiguous()
    gx = torch.empty_like(h, dtype=gs.dtype)
    gv_init = torch.empty_like(h[0])
    block = (
        512
        if (
            surrogate_id == 1
            and h.shape[0] > 1
            and gv_init.numel() >= 2097152
            and not store_v_seq
        )
        else 256
    )
    unroll = h.shape[0] if use_static_range_for_triton_neuron_kernel(h.shape[0]) else 1
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[(triton.cdiv(gv_init.numel(), block),)](
            gs,
            gv,
            h,
            gx,
            gv_init,
            h.shape[0],
            gv_init.numel(),
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
            enable_fp_fusion=False,
        )
    return gx, gv_init
