from typing import Optional

import torch

from ..surrogate_kernel import resolve_sg_triton_id_and_alpha, sg_triton
from ..triton_utils import convert_and_store, register_op, type_dict, wrap_triton

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging

    from .. import dummy

    logging.info(
        f"spikingjelly.activation_based.triton_kernel.neuron_kernel.integrate_and_file: {e}"
    )
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()

__all__ = ["multistep_if"]


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
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
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
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr"],
)
@triton.jit
def _multistep_if_backward_kernel(
    grad_s_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    v_threshold,
    v_reset,
    sg_alpha,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,  # grad_s_seq.dtype; might != h_seq or s_seq.dtype
    sg_triton_id: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s = tl.load(grad_s_ptrs, boundary_check=(1,), padding_option="zero")
        grad_v_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v = tl.load(grad_v_ptrs, boundary_check=(1,), padding_option="zero")
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero")

        sg = sg_triton(h - v_threshold, sg_alpha, sg_triton_id)
        grad_v_combined = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_combined)
            else:
                grad_h = tl.fma(
                    grad_s - v_threshold * grad_v_combined, sg, grad_v_combined
                )
        else:
            s = (h >= v_threshold).to(dtype)
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_combined * (1.0 - s))
            else:
                grad_h = tl.fma(
                    tl.fma(grad_v_combined, v_reset - h, grad_s),
                    sg,
                    grad_v_combined * (1.0 - s),
                )
        grad_v_acc = grad_h
        grad_x = grad_h

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(grad_x_ptrs, grad_x, boundary_check=(1,))

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v_init_ptrs, grad_v_acc, boundary_check=(1,))


@register_op("sj::multistep_if_inference")
def multistep_if_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    with torch.cuda.device(x_seq.device.index):
        wrap_triton(_multistep_if_forward_kernel)[grid](
            x_seq,
            v_init,
            s_seq,
            v_seq,  # dummy
            v_seq,
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            dtype=type_dict[dtype],
            soft_reset=soft_reset,
            save_intermediates=False,
        )
    return s_seq, v_seq


@torch.library.register_fake("sj::multistep_if_inference")
def _multistep_if_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
    )


@register_op("sj::multistep_if_forward")
def multistep_if_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    with torch.cuda.device(x_seq.device.index):
        wrap_triton(_multistep_if_forward_kernel)[grid](
            x_seq,
            v_init,
            s_seq,
            h_seq,
            v_seq,
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            dtype=type_dict[dtype],
            soft_reset=soft_reset,
            save_intermediates=True,
        )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_if_forward")
def _multistep_if_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
    )


def _setup_context(ctx, inputs, output):
    v_threshold, v_reset, soft_reset, detach_reset, sg_triton_id, sg_alpha = inputs[2:]
    h_seq = output[2]
    ctx.save_for_backward(h_seq)
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha


def _multistep_if_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    if h_seq.numel() == 0:
        raise RuntimeError("backward called without saved intermediates")

    grad_s_seq = grad_s_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v_init = torch.empty_like(grad_v_seq[0])
    dtype = grad_s_seq.dtype
    if dtype not in type_dict:
        raise NotImplementedError(dtype)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    with torch.cuda.device(grad_s_seq.device.index):
        wrap_triton(_multistep_if_backward_kernel)[grid](
            grad_s_seq,
            grad_v_seq,
            h_seq,
            grad_x_seq,
            grad_v_init,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.sg_alpha,
            T=T,
            NCL=NCL,
            dtype=type_dict[dtype],
            sg_triton_id=ctx.sg_triton_id,
            soft_reset=ctx.soft_reset,
            detach_reset=ctx.detach_reset,
        )
    return grad_x_seq, grad_v_init, None, None, None, None, None, None


torch.library.register_autograd(
    "sj::multistep_if_forward", _multistep_if_backward, setup_context=_setup_context
)


def multistep_if(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function,
) -> tuple[torch.Tensor, torch.Tensor]:
    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v_init.requires_grad
    )
    if need_grad:
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, _ = multistep_if_forward(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
        )
    else:
        s_seq, v_seq = multistep_if_inference(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            soft_reset,
        )
    return s_seq, v_seq
