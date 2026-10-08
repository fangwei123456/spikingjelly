from typing import Optional

import torch
import triton
import triton.language as tl

from ..triton_surrogate import _surrogate_gradient
from ..validation import _check_gradients, _check_inputs


@triton.jit
def _forward_kernel(
    x_ptr,
    v_ptr,
    spikes_ptr,
    voltages_ptr,
    charged_ptr,
    T,
    N,
    threshold,
    reset,
    SOFT_RESET: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inductor may pass Python floats as fp64; this operator's arithmetic is fp32.
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    v = tl.load(v_ptr + n, mask=mask, other=0.0)
    for t in range(T):
        offset = t.to(tl.int64) * N + n
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        h = v + x
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
    T,
    N,
    threshold,
    reset,
    alpha,
    SOFT_RESET: tl.constexpr,
    DETACH_RESET: tl.constexpr,
    SURROGATE: tl.constexpr,
    STORE_V_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    threshold = tl.cast(threshold, tl.float32)
    reset = tl.cast(reset, tl.float32)
    alpha = tl.cast(alpha, tl.float32)
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    carry = tl.full((BLOCK,), 0.0, tl.float32)
    for step in range(T):
        offset = (T - 1 - step).to(tl.int64) * N + n
        h = tl.load(h_ptr + offset, mask=mask, other=0.0)
        sg = _surrogate_gradient(h - threshold, alpha, SURROGATE)
        if SOFT_RESET:
            reset_grad = tl.full((BLOCK,), 1.0, tl.float32)
        else:
            reset_grad = 1.0 - (h >= threshold).to(tl.float32)
        if not DETACH_RESET:
            if SOFT_RESET:
                reset_grad = reset_grad - threshold * sg
            else:
                reset_grad = reset_grad + (reset - h) * sg
        gs = tl.load(gs_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if STORE_V_SEQ:
            gv = tl.load(gv_ptr + offset, mask=mask, other=0.0)
        elif step == 0:
            gv = tl.load(gv_ptr + n, mask=mask, other=0.0)
        else:
            gv = tl.full((BLOCK,), 0.0, tl.float32)
        gh = gs * sg + (gv + carry) * reset_grad
        tl.store(gx_ptr + offset, gh, mask=mask)
        carry = gh
    tl.store(gv_init_ptr + n, carry, mask=mask)


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_inputs(x, v, threshold, reset, alpha, surrogate_id)
    torch._check(x.device.type == "cuda", lambda: "Triton IF requires CUDA")
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
            threshold,
            0.0 if reset is None else reset,
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
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_gradients(gs, gv, h, threshold, reset, alpha, store_v_seq, surrogate_id)
    torch._check(h.device.type == "cuda", lambda: "Triton IF requires CUDA")
    gs, gv, h = gs.contiguous(), gv.contiguous(), h.contiguous()
    gx = torch.empty_like(h, dtype=gs.dtype)
    gv_init = torch.empty_like(h[0])
    with torch.cuda.device(h.device):
        (
            _backward_kernel
            if _kernel_wrapper is None
            else _kernel_wrapper(_backward_kernel)
        )[(triton.cdiv(gv_init.numel(), 256),)](
            gs,
            gv,
            h,
            gx,
            gv_init,
            h.shape[0],
            gv_init.numel(),
            threshold,
            0.0 if reset is None else reset,
            alpha,
            reset is None,
            detach_reset,
            surrogate_id,
            store_v_seq,
            256,
            enable_fp_fusion=False,
        )
    return gx, gv_init
