from typing import Callable

import torch
from torch import autograd

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging
    logging.info(f'spikingjelly.activation_based.triton_kernel.neuron_kernel.plif: {e}')
    triton = None
    tl = None

from ..triton_utils import type_dict, contiguous_and_device_guard
from ..triton_utils import amp_custom_fwd, amp_custom_bwd


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "dtype", "soft_reset", "save_intermediates"],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_plif_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    r_tau,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        if decay_input:
            h = v + r_tau * (v_reset-v+x)
        else:
            h = v + r_tau * (v_reset-v) + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s*v_threshold
        else:
            v = s*v_reset + (1.-s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0)
            )
            tl.store(h_ptrs, h, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr", "grad_r_tau_ptr"],
)
@triton.jit
def _multistep_plif_backward_kernel(
    grad_s_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    v_init_v_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    grad_r_tau_ptr,
    r_tau,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,  # grad_s_seq.dtype; might != h_seq or s_seq.dtype
    sg_fn: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=dtype)
    grad_r_tau_acc = tl.zeros([1, BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        grad_s = tl.load(
            grad_s_ptrs, boundary_check=(1,), padding_option="zero"
        )
        grad_v_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        grad_v = tl.load(
            grad_v_ptrs, boundary_check=(1,), padding_option="zero"
        )
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero")
        v_last_ptrs = tl.make_block_ptr(
            v_init_v_seq_ptr,
            shape=(T+1, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        v_last = tl.load(
            v_last_ptrs, boundary_check=(0, 1), padding_option="zero"
        )

        sg = sg_fn(h - v_threshold)
        grad_v_acc = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc)
            else:
                grad_h = tl.fma(
                    grad_s - v_threshold*grad_v_acc, sg, grad_v_acc
                )
        else:
            s = (h >= v_threshold).to(dtype)
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc * (1.-s))
            else:
                grad_h = tl.fma(
                    tl.fma(grad_v_acc, v_reset - h, grad_s),
                    sg,
                    grad_v_acc * (1.-s),
                )
        grad_v_acc = grad_h * (1.-r_tau)
        if decay_input:
            grad_x = grad_h * r_tau
            grad_r_tau = grad_h * (h - v_last) / r_tau
        else:
            grad_x = grad_h
            grad_r_tau = grad_h * (v_reset - v_last)

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_x.to(dtype), boundary_check=(1,))
        grad_r_tau_acc = grad_r_tau_acc + grad_r_tau

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(grad_v_init_ptrs, grad_v_acc.to(dtype), boundary_check=(1,))
    #! atomic add is not supported on some devices / triton versions
    #! so we use a workaround here, summing the gradient outside the kernel
    grad_r_tau_ptrs = tl.make_block_ptr(
        grad_r_tau_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(grad_r_tau_ptrs, grad_r_tau_acc.to(dtype), boundary_check=(1,))


def multistep_plif_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq, v_seq = torch.empty_like(x_seq), torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_forward_kernel[grid](
        x_seq,
        v_init,
        s_seq,
        None,
        v_seq,
        r_tau.item(),
        v_threshold,
        v_reset,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        decay_input=decay_input,
        soft_reset=soft_reset,
        save_intermediates=False,
    )
    return s_seq, v_seq


def multistep_plif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq, v_seq = torch.empty_like(x_seq), torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_forward_kernel[grid](
        x_seq,
        v_init,
        s_seq,
        h_seq,
        v_seq,
        r_tau.item(),
        v_threshold,
        v_reset,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        decay_input=decay_input,
        soft_reset=soft_reset,
        save_intermediates=True,
    )
    return s_seq, v_seq, h_seq


def multistep_plif_backward(
    grad_s_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_init_v_seq: torch.Tensor,
    r_tau: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    sg_fn: Callable,
    decay_input: bool,
    soft_reset: bool,
    detach_reset: bool,
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v_init = torch.empty_like(grad_v_seq[0])
    grad_r_tau = torch.empty_like(grad_v_seq[0])
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_backward_kernel[grid](
        grad_s_seq,
        grad_v_seq,
        h_seq,
        v_init_v_seq,
        grad_x_seq,
        grad_v_init,
        grad_r_tau,
        r_tau.item(),
        v_threshold,
        v_reset,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        sg_fn=sg_fn,
        decay_input=decay_input,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
    )
    return grad_x_seq, grad_v_init, grad_r_tau.sum()


class MultiStepParametricLIFNodePTT(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, v_init: torch.Tensor, r_tau: torch.Tensor,
        decay_input: bool, v_threshold: float, v_reset: float,
        detach_reset: bool, sg_fn: Callable
    ):
        soft_reset = v_reset is None
        v_reset = v_reset if v_reset is not None else 0.
        if any(ctx.needs_input_grad):
            s_seq, v_seq, h_seq = multistep_plif_forward(
                x_seq, v_init, r_tau, decay_input, v_threshold, v_reset,
                soft_reset
            )
            v_init_v_seq = torch.cat([v_init.unsqueeze(0), v_seq], dim=0)
            ctx.save_for_backward(h_seq, v_init_v_seq, r_tau)
            ctx.decay_input = decay_input
            ctx.v_threshold = v_threshold
            ctx.v_reset = v_reset
            ctx.soft_reset = soft_reset
            ctx.detach_reset = detach_reset
            ctx.sg_fn = sg_fn
        else:
            s_seq, v_seq = multistep_plif_inference(
                x_seq, v_init, r_tau, decay_input, v_threshold, v_reset,
                soft_reset
            )
        return s_seq, v_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        h_seq = ctx.saved_tensors[0]
        v_init_v_seq = ctx.saved_tensors[1]
        r_tau = ctx.saved_tensors[2]
        grad_x_seq, grad_v_init, grad_r_tau = multistep_plif_backward(
            grad_s_seq, grad_v_seq, h_seq, v_init_v_seq, r_tau, ctx.v_threshold,
            ctx.v_reset, ctx.sg_fn, ctx.decay_input, ctx.soft_reset,
            ctx.detach_reset
        )
        return (
            grad_x_seq, grad_v_init, grad_r_tau, None, None, None, None, None
        )
