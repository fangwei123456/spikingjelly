from typing import Optional

import torch

from ..surrogate_kernel import resolve_sg_triton_id_and_alpha, sg_triton
from ..triton_utils import (
    convert_and_store,
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

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
def _multistep_if_forward_kernel_static(
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
    key=["NCL", "dtype", "soft_reset", "save_intermediates"],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_if_forward_kernel_dynamic(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T,
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

    for t in tl.range(0, T, 1):
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
def _multistep_if_backward_kernel_static(
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["NCL", "dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr"],
)
@triton.jit
def _multistep_if_backward_kernel_dynamic(
    grad_s_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    v_threshold,
    v_reset,
    sg_alpha,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=dtype)

    for t in tl.range(T - 1, -1, -1):
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


LAST_FORWARD_LOOP_MODE = None
LAST_BACKWARD_LOOP_MODE = None


def _select_forward_kernel(T: int):
    global LAST_FORWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_FORWARD_LOOP_MODE = "static"
        return _multistep_if_forward_kernel_static
    LAST_FORWARD_LOOP_MODE = "dynamic"
    return _multistep_if_forward_kernel_dynamic


def _select_backward_kernel(T: int):
    global LAST_BACKWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_BACKWARD_LOOP_MODE = "static"
        return _multistep_if_backward_kernel_static
    LAST_BACKWARD_LOOP_MODE = "dynamic"
    return _multistep_if_backward_kernel_dynamic


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
        wrap_triton(_select_forward_kernel(T))[grid](
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
        wrap_triton(_select_forward_kernel(T))[grid](
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
        wrap_triton(_select_backward_kernel(T))[grid](
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
    """Multi-step IF neuron forward pass via Triton kernel.
    **API Language:**
    :ref:`中文 <multistep_if-cn>` | :ref:`English <multistep_if-en>`

    ----

    .. _multistep_if-cn:

    * **中文**

    多步IF神经元Triton kernel前向传播

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :type x_seq: ``torch.Tensor``
    :param v_init: Initial membrane potential
    :type v_init: ``torch.Tensor``
    :param v_threshold: Threshold voltage
    :type v_threshold: float
    :param v_reset: Reset voltage (``None`` for soft reset)
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach the reset term in backward
    :type detach_reset: bool
    :param surrogate_function: Surrogate gradient function
    :type surrogate_function: ``surrogate.SurrogateFunctionBase``
    :return: Tuple of (spike_seq, v_seq)
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _multistep_if-en:

    * **English**

    Multi-step IF neuron Triton kernel forward

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :param v_init: Initial membrane potential
    :param v_threshold: Threshold voltage
    :param v_reset: Reset voltage (``None`` for soft reset)
    :param detach_reset: Whether to detach the reset term in backward
    :param surrogate_function: Surrogate gradient function
    :type x_seq: ``torch.Tensor``
    :type v_init: ``torch.Tensor``
    :type v_threshold: float
    :type v_reset: Optional[float]
    :type detach_reset: bool
    :type surrogate_function: ``surrogate.SurrogateFunctionBase``
    :return: Tuple of (spike_seq, v_seq)
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
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
