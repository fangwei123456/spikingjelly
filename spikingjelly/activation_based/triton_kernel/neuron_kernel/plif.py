from typing import Optional

import torch

from ... import surrogate
from ..surrogate_kernel import resolve_sg_triton_id_and_alpha, sg_triton
from ..triton_utils import (
    convert_and_store,
    register_op,
    torch_dtype_for_triton_compute_dtype,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)
from .utils import (
    TritonNeuronForwardPlan,
    _check_plan_inputs,
    prepare_triton_neuron_forward_plan,
)

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging

    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.neuron_kernel.plif: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()


__all__ = ["multistep_plif"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "compute_dtype", "soft_reset", "save_intermediates"],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_plif_forward_kernel_static(
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
    compute_dtype: tl.constexpr,
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
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero").to(
        compute_dtype
    )
    r_tau = tl.full([1], r_tau, dtype=compute_dtype)

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )

        if decay_input:
            h = v + r_tau * (v_reset - v + x)
        else:
            h = v + r_tau * (v_reset - v) + x
        s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
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
    key=["NCL", "compute_dtype", "soft_reset", "save_intermediates"],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_plif_forward_kernel_dynamic(
    x_seq_ptr,
    v_init_ptr,
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    r_tau,
    v_threshold,
    v_reset,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
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
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero").to(
        compute_dtype
    )
    r_tau = tl.full([1], r_tau, dtype=compute_dtype)

    for t in tl.range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )

        if decay_input:
            h = v + r_tau * (v_reset - v + x)
        else:
            h = v + r_tau * (v_reset - v) + x
        s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
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
    key=["T", "NCL", "compute_dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr", "grad_r_tau_ptr"],
)
@triton.jit
def _multistep_plif_backward_kernel_static(
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
    alpha,  # for surrogate gradient
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    r_tau = tl.full([1], r_tau, dtype=compute_dtype)
    grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    grad_r_tau_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s = tl.load(
            grad_s_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        grad_v_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v = tl.load(
            grad_v_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        v_last_ptrs = tl.make_block_ptr(
            v_init_v_seq_ptr,
            shape=(T + 1, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        v_last = tl.load(
            v_last_ptrs, boundary_check=(0, 1), padding_option="zero"
        ).to(compute_dtype)

        sg = sg_triton(h - v_threshold, alpha, sg_triton_id)
        grad_v_acc = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc)
            else:
                grad_h = tl.fma(grad_s - v_threshold * grad_v_acc, sg, grad_v_acc)
        else:
            s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc * (1.0 - s))
            else:
                grad_h = tl.fma(
                    tl.fma(grad_v_acc, v_reset - h, grad_s),
                    sg,
                    grad_v_acc * (1.0 - s),
                )
        grad_v_acc = grad_h * (1.0 - r_tau)
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
            order=(1, 0),
        )
        convert_and_store(grad_x_ptrs, grad_x, boundary_check=(1,))
        grad_r_tau_acc = grad_r_tau_acc + grad_r_tau

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v_init_ptrs, grad_v_acc, boundary_check=(1,))
    #! atomic add is not supported on some devices / triton versions
    #! so we use a workaround here, summing the gradient outside the kernel
    grad_r_tau_ptrs = tl.make_block_ptr(
        grad_r_tau_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_r_tau_ptrs, grad_r_tau_acc, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["NCL", "compute_dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr", "grad_r_tau_ptr"],
)
@triton.jit
def _multistep_plif_backward_kernel_dynamic(
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
    alpha,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    r_tau = tl.full([1], r_tau, dtype=compute_dtype)
    grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    grad_r_tau_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)

    for t in tl.range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s = tl.load(
            grad_s_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        grad_v_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v = tl.load(
            grad_v_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        v_last_ptrs = tl.make_block_ptr(
            v_init_v_seq_ptr,
            shape=(T + 1, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        v_last = tl.load(
            v_last_ptrs, boundary_check=(0, 1), padding_option="zero"
        ).to(compute_dtype)

        sg = sg_triton(h - v_threshold, alpha, sg_triton_id)
        grad_v_acc = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc)
            else:
                grad_h = tl.fma(grad_s - v_threshold * grad_v_acc, sg, grad_v_acc)
        else:
            s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc * (1.0 - s))
            else:
                grad_h = tl.fma(
                    tl.fma(grad_v_acc, v_reset - h, grad_s),
                    sg,
                    grad_v_acc * (1.0 - s),
                )
        grad_v_acc = grad_h * (1.0 - r_tau)
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
            order=(1, 0),
        )
        convert_and_store(grad_x_ptrs, grad_x, boundary_check=(1,))
        grad_r_tau_acc = grad_r_tau_acc + grad_r_tau

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v_init_ptrs, grad_v_acc, boundary_check=(1,))
    grad_r_tau_ptrs = tl.make_block_ptr(
        grad_r_tau_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_r_tau_ptrs, grad_r_tau_acc, boundary_check=(1,))


# Test instrumentation only; not thread-safe.
LAST_FORWARD_LOOP_MODE = None
LAST_BACKWARD_LOOP_MODE = None


def _select_forward_kernel(T: int):
    global LAST_FORWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_FORWARD_LOOP_MODE = "static"
        return _multistep_plif_forward_kernel_static
    LAST_FORWARD_LOOP_MODE = "dynamic"
    return _multistep_plif_forward_kernel_dynamic


def _select_backward_kernel(T: int):
    global LAST_BACKWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_BACKWARD_LOOP_MODE = "static"
        return _multistep_plif_backward_kernel_static
    LAST_BACKWARD_LOOP_MODE = "dynamic"
    return _multistep_plif_backward_kernel_dynamic


def _launch_plif_forward_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    s_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    *,
    r_tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    compute_dtype,
    save_intermediates: bool,
    use_torch_wrap: bool,
) -> None:
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    kernel = _select_forward_kernel(T)
    if use_torch_wrap:
        kernel = wrap_triton(kernel)

    with torch.cuda.device(x_seq.device.index):
        kernel[grid](
            x_seq,
            v_init,
            s_seq,
            h_seq,
            v_seq,
            r_tau,
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            compute_dtype=compute_dtype,
            decay_input=decay_input,
            soft_reset=soft_reset,
            save_intermediates=save_intermediates,
        )


def _launch_plif_backward_kernel(
    grad_s_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_init_v_seq: torch.Tensor,
    grad_x_seq: torch.Tensor,
    grad_v_init: torch.Tensor,
    grad_r_tau: torch.Tensor,
    *,
    r_tau: float,
    v_threshold: float,
    v_reset: float,
    sg_alpha: float,
    compute_dtype,
    sg_triton_id: int,
    decay_input: bool,
    soft_reset: bool,
    detach_reset: bool,
    use_torch_wrap: bool,
) -> None:
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    kernel = _select_backward_kernel(T)
    if use_torch_wrap:
        kernel = wrap_triton(kernel)

    with torch.cuda.device(grad_s_seq.device.index):
        kernel[grid](
            grad_s_seq,
            grad_v_seq,
            h_seq,
            v_init_v_seq,
            grad_x_seq,
            grad_v_init,
            grad_r_tau,
            r_tau,
            v_threshold,
            v_reset,
            sg_alpha,
            T=T,
            NCL=NCL,
            compute_dtype=compute_dtype,
            sg_triton_id=sg_triton_id,
            decay_input=decay_input,
            soft_reset=soft_reset,
            detach_reset=detach_reset,
        )


@register_op("sj::multistep_plif_inference")
def multistep_plif_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    _launch_plif_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        v_seq,  # dummy
        v_seq,
        r_tau=r_tau.item(),
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=False,
        use_torch_wrap=True,
    )
    return s_seq, v_seq


@torch.library.register_fake("sj::multistep_plif_inference")
def _multistep_plif_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
    )


@register_op("sj::multistep_plif_forward")
def multistep_plif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    _launch_plif_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        h_seq,
        v_seq,
        r_tau=r_tau.item(),
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=True,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_plif_forward")
def _multistep_plif_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
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


def multistep_plif_mixed_precision_forward_with_plan(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    plan: TritonNeuronForwardPlan,
    *,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool = False,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if plan.neuron_type != "plif":
        raise ValueError(
            f"PLIF forward requires a PLIF plan, got {plan.neuron_type!r}."
        )
    if torch.is_grad_enabled() and (
        x_seq.requires_grad or v_init.requires_grad or r_tau.requires_grad
    ):
        if surrogate_function is None:
            surrogate_function = surrogate.Sigmoid()
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, h_seq = _MixedPrecisionPLIF.apply(
            x_seq,
            v_init,
            r_tau,
            plan,
            decay_input,
            v_threshold,
            v_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
        )
        return s_seq, v_seq, (h_seq if plan.save_intermediates else None)
    _check_plan_inputs(x_seq, v_init, plan, "PLIF")

    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    x_storage = x_seq.detach().to(dtype=plan.storage_dtype).contiguous()
    v_storage = v_init.detach().to(dtype=plan.storage_dtype).contiguous()
    s_seq = torch.empty(x_seq.shape, dtype=plan.spike_dtype, device=x_seq.device)
    v_seq = torch.empty(x_seq.shape, dtype=plan.storage_dtype, device=x_seq.device)
    if plan.save_intermediates:
        h_seq = torch.empty(x_seq.shape, dtype=plan.storage_dtype, device=x_seq.device)
    else:
        h_seq = v_seq

    _launch_plif_forward_kernel(
        x_storage,
        v_storage,
        s_seq,
        h_seq,
        v_seq,
        r_tau=r_tau.detach().item(),
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=plan.compute_tl_dtype,
        save_intermediates=plan.save_intermediates,
        use_torch_wrap=False,
    )
    return s_seq, v_seq, (h_seq if plan.save_intermediates else None)


def multistep_plif_mixed_precision_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    *,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    storage_dtype,
    compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
    detach_reset: bool = False,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Experimental mixed-precision multi-step PLIF forward path using the same
    Triton forward kernel source as :func:`multistep_plif`.

    This path is intended for FP8 storage experiments where storage dtype,
    forward compute dtype, and backward compute dtype must be controlled
    independently.

    .. warning::
        When ``compute_dtype='fp8'``, the PLIF recurrence and threshold comparison
        are performed in FP8 precision. This mode has limited dynamic range and
        mantissa bits, and may produce incorrect spike patterns. Use it only for
        experiments, not for accuracy-critical inference.
    """
    plan = prepare_triton_neuron_forward_plan(
        neuron_type="plif",
        device=x_seq.device,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=spike_dtype,
        save_intermediates=save_intermediates,
    )
    return multistep_plif_mixed_precision_forward_with_plan(
        x_seq,
        v_init,
        r_tau,
        plan,
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )


class _MixedPrecisionPLIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        r_tau: torch.Tensor,
        plan: TritonNeuronForwardPlan,
        decay_input: bool,
        v_threshold: float,
        v_reset: Optional[float],
        detach_reset: bool,
        sg_triton_id: int,
        sg_alpha: float,
    ):
        _check_plan_inputs(x_seq, v_init, plan, "PLIF")
        soft_reset = v_reset is None
        v_reset = v_reset if v_reset is not None else 0.0
        x_storage = x_seq.to(dtype=plan.storage_dtype).contiguous()
        v_storage = v_init.to(dtype=plan.storage_dtype).contiguous()
        s_seq = torch.empty(x_seq.shape, dtype=plan.spike_dtype, device=x_seq.device)
        v_seq = torch.empty(x_seq.shape, dtype=plan.storage_dtype, device=x_seq.device)
        h_seq = torch.empty(x_seq.shape, dtype=plan.storage_dtype, device=x_seq.device)

        _launch_plif_forward_kernel(
            x_storage,
            v_storage,
            s_seq,
            h_seq,
            v_seq,
            r_tau=r_tau.detach().item(),
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            soft_reset=soft_reset,
            compute_dtype=plan.forward_compute_tl_dtype,
            save_intermediates=True,
            use_torch_wrap=False,
        )
        v_init_v_seq = torch.cat([v_storage.unsqueeze(0), v_seq], dim=0)
        ctx.save_for_backward(h_seq, v_init_v_seq, r_tau)
        ctx.plan = plan
        ctx.x_dtype = x_seq.dtype
        ctx.v_init_dtype = v_init.dtype
        ctx.r_tau_dtype = r_tau.dtype
        ctx.decay_input = decay_input
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.soft_reset = soft_reset
        ctx.detach_reset = detach_reset
        ctx.sg_triton_id = sg_triton_id
        ctx.sg_alpha = sg_alpha
        return s_seq, v_seq, h_seq

    @staticmethod
    def backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
        h_seq, v_init_v_seq, r_tau = ctx.saved_tensors
        plan = ctx.plan
        if grad_s_seq is None:
            grad_s_seq = torch.zeros(
                h_seq.shape, dtype=plan.spike_dtype, device=h_seq.device
            )
        if grad_v_seq is None:
            grad_v_seq = torch.zeros(
                h_seq.shape, dtype=plan.storage_dtype, device=h_seq.device
            )
        grad_s_seq = grad_s_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()
        h_seq = h_seq.contiguous()
        v_init_v_seq = v_init_v_seq.contiguous()
        grad_x_seq = torch.empty(h_seq.shape, dtype=ctx.x_dtype, device=h_seq.device)
        grad_v_init = torch.empty(
            h_seq[0].shape, dtype=ctx.v_init_dtype, device=h_seq.device
        )
        grad_r_tau_dtype = torch_dtype_for_triton_compute_dtype(
            plan.backward_compute_dtype_name
        )
        grad_r_tau_seq = torch.empty(
            h_seq[0].shape, dtype=grad_r_tau_dtype, device=h_seq.device
        )

        _launch_plif_backward_kernel(
            grad_s_seq,
            grad_v_seq,
            h_seq,
            v_init_v_seq,
            grad_x_seq,
            grad_v_init,
            grad_r_tau_seq,
            r_tau=r_tau.item(),
            v_threshold=ctx.v_threshold,
            v_reset=ctx.v_reset,
            sg_alpha=ctx.sg_alpha,
            compute_dtype=plan.backward_compute_tl_dtype,
            sg_triton_id=ctx.sg_triton_id,
            decay_input=ctx.decay_input,
            soft_reset=ctx.soft_reset,
            detach_reset=ctx.detach_reset,
            use_torch_wrap=False,
        )
        grad_r_tau = grad_r_tau_seq.sum().to(dtype=ctx.r_tau_dtype)
        return (
            grad_x_seq,
            grad_v_init,
            grad_r_tau,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _setup_context(ctx, inputs, output):
    (
        v_init,
        r_tau,
        decay_input,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        sg_triton_id,
        sg_alpha,
    ) = inputs[1:]
    _, v_seq, h_seq = output
    v_init_v_seq = torch.cat([v_init.unsqueeze(0), v_seq], dim=0)
    ctx.save_for_backward(h_seq, v_init_v_seq, r_tau)
    ctx.decay_input = decay_input
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha


def _multistep_plif_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    h_seq, v_init_v_seq, r_tau = ctx.saved_tensors
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v_init = torch.empty_like(grad_v_seq[0])
    grad_r_tau = torch.empty_like(grad_v_seq[0])
    dtype = grad_s_seq.dtype
    _launch_plif_backward_kernel(
        grad_s_seq.contiguous(),
        grad_v_seq.contiguous(),
        h_seq.contiguous(),
        v_init_v_seq.contiguous(),
        grad_x_seq,
        grad_v_init,
        grad_r_tau,
        r_tau=r_tau.item(),
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        compute_dtype=type_dict[dtype],
        sg_triton_id=ctx.sg_triton_id,
        decay_input=ctx.decay_input,
        soft_reset=ctx.soft_reset,
        detach_reset=ctx.detach_reset,
        use_torch_wrap=True,
    )
    grad_r_tau = grad_r_tau.sum()
    return grad_x_seq, grad_v_init, grad_r_tau, None, None, None, None, None, None, None


torch.library.register_autograd(
    "sj::multistep_plif_forward", _multistep_plif_backward, setup_context=_setup_context
)


def multistep_plif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    r_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-step Parametric LIF neuron forward pass via Triton kernel.

    **API Language** - :ref:`中文 <multistep_plif-cn>` | :ref:`English <multistep_plif-en>`

    ----

    .. _multistep_plif-cn:

    * **中文**

    多步PLIF神经元Triton kernel前向传播

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :type x_seq: ``torch.Tensor``
    :param v_init: Initial membrane potential
    :type v_init: ``torch.Tensor``
    :param r_tau: Reciprocal of the learnable membrane time constant
    :type r_tau: ``torch.Tensor``
    :param decay_input: Whether input participates in decay
    :type decay_input: bool
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

    .. _multistep_plif-en:

    * **English**

    Multi-step PLIF neuron Triton kernel forward

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :param v_init: Initial membrane potential
    :param r_tau: Reciprocal of the learnable membrane time constant
    :param decay_input: Whether input participates in decay
    :param v_threshold: Threshold voltage
    :param v_reset: Reset voltage (``None`` for soft reset)
    :param detach_reset: Whether to detach the reset term in backward
    :param surrogate_function: Surrogate gradient function
    :type x_seq: ``torch.Tensor``
    :type v_init: ``torch.Tensor``
    :type r_tau: ``torch.Tensor``
    :type decay_input: bool
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
        x_seq.requires_grad or v_init.requires_grad or r_tau.requires_grad
    )
    if need_grad:
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, _ = multistep_plif_forward(
            x_seq,
            v_init,
            r_tau,
            decay_input,
            v_threshold,
            v_reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
        )
    else:
        s_seq, v_seq = multistep_plif_inference(
            x_seq,
            v_init,
            r_tau,
            decay_input,
            v_threshold,
            v_reset,
            soft_reset,
        )
    return s_seq, v_seq
