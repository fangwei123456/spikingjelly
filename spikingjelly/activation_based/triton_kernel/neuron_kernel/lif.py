from typing import Optional

import torch

from ... import surrogate
from ..surrogate_kernel import resolve_sg_triton_id_and_alpha, sg_triton
from ..triton_utils import (
    convert_and_store,
    register_op,
    triton_neuron_compute_dtype_id_to_tl_dtype,
    triton_neuron_dtype_id_to_torch_dtype,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)
from .utils import (
    TritonNeuronExecutionPlan,
    _check_mp_cuda_inputs,
    _check_plan_inputs,
    prepare_triton_neuron_execution_plan,
)

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging

    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.neuron_kernel.lif: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()


__all__ = ["single_step_lif", "multistep_lif"]


@triton.jit
def _single_step_lif_forward_kernel(
    x_ptr,
    v_init_ptr,
    spike_ptr,
    h_ptr,
    v_ptr,
    tau,
    v_threshold,
    v_reset,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = offsets < NCL
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    v_init = tl.load(v_init_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)
    threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    reset = tl.full([1], v_reset, dtype=compute_dtype)

    if decay_input:
        h = v_init + r_tau * (reset - v_init + x)
    else:
        h = v_init + r_tau * (reset - v_init) + x
    spike = tl.where(h >= threshold, 1.0, 0.0).to(compute_dtype)
    if soft_reset:
        v = h - spike * threshold
    else:
        v = spike * reset + (1.0 - spike) * h

    tl.store(spike_ptr + offsets, spike, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)
    if save_intermediates:
        tl.store(h_ptr + offsets, h, mask=mask)


@triton.jit
def _single_step_lif_backward_kernel(
    grad_spike_ptr,
    grad_v_ptr,
    h_ptr,
    grad_x_ptr,
    grad_v_init_ptr,
    tau,
    v_threshold,
    v_reset,
    sg_alpha,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = offsets < NCL
    grad_spike = tl.load(grad_spike_ptr + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    grad_v = tl.load(grad_v_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)
    threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    reset = tl.full([1], v_reset, dtype=compute_dtype)

    sg = sg_triton(h - threshold, sg_alpha, sg_triton_id)
    if soft_reset:
        if detach_reset:
            grad_h = tl.fma(grad_spike, sg, grad_v)
        else:
            grad_h = tl.fma(grad_spike - threshold * grad_v, sg, grad_v)
    else:
        spike = tl.where(h >= threshold, 1.0, 0.0).to(compute_dtype)
        if detach_reset:
            grad_h = tl.fma(grad_spike, sg, grad_v * (1.0 - spike))
        else:
            grad_h = tl.fma(
                tl.fma(grad_v, reset - h, grad_spike),
                sg,
                grad_v * (1.0 - spike),
            )
    if decay_input:
        grad_x = grad_h * r_tau
    else:
        grad_x = grad_h
    grad_v_init = grad_h * (1.0 - r_tau)

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_v_init_ptr + offsets, grad_v_init, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "T",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "save_intermediates",
        "store_v_seq",
    ],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_lif_forward_kernel_static(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    tau,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)

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
        if store_v_seq:
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

    if not store_v_seq:
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_last_ptrs, v, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "NCL",
        "compute_dtype",
        "soft_reset",
        "save_intermediates",
        "store_v_seq",
    ],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_lif_forward_kernel_dynamic(
    x_seq_ptr,
    v_init_ptr,
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    tau,
    v_threshold,
    v_reset,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)

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
        if store_v_seq:
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

    if not store_v_seq:
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_last_ptrs, v, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "T",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "detach_reset",
        "store_v_seq",
    ],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr"],
)
@triton.jit
def _multistep_lif_backward_kernel_static(
    grad_s_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    tau,
    v_threshold,
    v_reset,
    sg_alpha,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    else:
        grad_v_last_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v_acc = tl.load(
            grad_v_last_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s = tl.load(grad_s_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        if store_v_seq:
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

        sg = sg_triton(h - v_threshold, sg_alpha, sg_triton_id)
        if store_v_seq:
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
        else:
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
    key=[
        "NCL",
        "compute_dtype",
        "soft_reset",
        "detach_reset",
        "store_v_seq",
    ],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr"],
)
@triton.jit
def _multistep_lif_backward_kernel_dynamic(
    grad_s_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    tau,
    v_threshold,
    v_reset,
    sg_alpha,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    else:
        grad_v_last_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v_acc = tl.load(
            grad_v_last_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)

    for t in tl.range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s = tl.load(grad_s_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        if store_v_seq:
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

        sg = sg_triton(h - v_threshold, sg_alpha, sg_triton_id)
        if store_v_seq:
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
        else:
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


# Test instrumentation only; not thread-safe.
LAST_FORWARD_LOOP_MODE = None
LAST_BACKWARD_LOOP_MODE = None


def _select_forward_kernel(T: int):
    global LAST_FORWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_FORWARD_LOOP_MODE = "static"
        return _multistep_lif_forward_kernel_static
    LAST_FORWARD_LOOP_MODE = "dynamic"
    return _multistep_lif_forward_kernel_dynamic


def _select_backward_kernel(T: int):
    global LAST_BACKWARD_LOOP_MODE
    if use_static_range_for_triton_neuron_kernel(T):
        LAST_BACKWARD_LOOP_MODE = "static"
        return _multistep_lif_backward_kernel_static
    LAST_BACKWARD_LOOP_MODE = "dynamic"
    return _multistep_lif_backward_kernel_dynamic


def _launch_lif_forward_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    s_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    *,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    compute_dtype,
    save_intermediates: bool,
    store_v_seq: bool,
    use_torch_wrap: bool,
) -> None:
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()

    def grid(meta):
        return (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    kernel = _select_forward_kernel(T)
    if use_torch_wrap:
        kernel = wrap_triton(kernel)

    with torch.cuda.device(x_seq.device):
        kernel[grid](
            x_seq,
            v_init,
            s_seq,
            h_seq,
            v_seq,
            tau,
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            compute_dtype=compute_dtype,
            decay_input=decay_input,
            soft_reset=soft_reset,
            save_intermediates=save_intermediates,
            store_v_seq=store_v_seq,
        )


def _launch_lif_backward_kernel(
    grad_s_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    grad_x_seq: torch.Tensor,
    grad_v_init: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
    v_reset: float,
    sg_alpha: float,
    compute_dtype,
    sg_triton_id: int,
    decay_input: bool,
    soft_reset: bool,
    detach_reset: bool,
    store_v_seq: bool,
    use_torch_wrap: bool,
) -> None:
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()

    def grid(meta):
        return (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    kernel = _select_backward_kernel(T)
    if use_torch_wrap:
        kernel = wrap_triton(kernel)

    with torch.cuda.device(grad_s_seq.device):
        kernel[grid](
            grad_s_seq,
            grad_v_seq,
            h_seq,
            grad_x_seq,
            grad_v_init,
            tau,
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
            store_v_seq=store_v_seq,
        )


@register_op("sj::multistep_lif_inference")
def multistep_lif_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq) if store_v_seq else torch.empty_like(v_init)
    dtype = x_seq.dtype
    _launch_lif_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        v_seq,  # dummy
        v_seq,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=False,
        store_v_seq=store_v_seq,
        use_torch_wrap=True,
    )
    return s_seq, v_seq


@torch.library.register_fake("sj::multistep_lif_inference")
def _multistep_lif_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape if store_v_seq else v_init.shape),
    )


@register_op("sj::multistep_lif_forward")
def multistep_lif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq) if store_v_seq else torch.empty_like(v_init)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    _launch_lif_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        h_seq,
        v_seq,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=True,
        store_v_seq=store_v_seq,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_lif_forward")
def _multistep_lif_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
    store_v_seq: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape if store_v_seq else v_init.shape),
        x_seq.new_empty(x_seq.shape),
    )


@register_op("sj::multistep_lif_mp_inference")
def multistep_lif_mp_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    spike_dtype_id: int,
    save_intermediates: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_mp_cuda_inputs(x_seq, v_init, "LIF")
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    compute_tl_dtype = triton_neuron_compute_dtype_id_to_tl_dtype(
        forward_compute_dtype_id, storage_dtype_id
    )
    x_storage = x_seq.detach().to(dtype=storage_dtype).contiguous()
    v_storage = v_init.detach().to(dtype=storage_dtype).contiguous()
    s_seq = torch.empty(x_seq.shape, dtype=spike_dtype, device=x_seq.device)
    v_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)
    if save_intermediates:
        h_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)
        h_buffer = h_seq
    else:
        h_seq = torch.empty((0,), dtype=storage_dtype, device=x_seq.device)
        h_buffer = v_seq

    _launch_lif_forward_kernel(
        x_storage,
        v_storage,
        s_seq,
        h_buffer,
        v_seq,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=compute_tl_dtype,
        save_intermediates=save_intermediates,
        store_v_seq=True,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_lif_mp_inference")
def _multistep_lif_mp_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    spike_dtype_id: int,
    save_intermediates: bool,
):
    del (
        v_init,
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        forward_compute_dtype_id,
    )
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    h_shape = x_seq.shape if save_intermediates else (0,)
    return (
        torch.empty(x_seq.shape, dtype=spike_dtype, device=x_seq.device),
        torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device),
        torch.empty(h_shape, dtype=storage_dtype, device=x_seq.device),
    )


@register_op("sj::multistep_lif_mp_forward")
def multistep_lif_mp_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    backward_compute_dtype_id: int,
    spike_dtype_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del detach_reset, backward_compute_dtype_id
    _check_mp_cuda_inputs(x_seq, v_init, "LIF")
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    compute_tl_dtype = triton_neuron_compute_dtype_id_to_tl_dtype(
        forward_compute_dtype_id, storage_dtype_id
    )
    x_storage = x_seq.to(dtype=storage_dtype).contiguous()
    v_storage = v_init.to(dtype=storage_dtype).contiguous()
    s_seq = torch.empty(x_seq.shape, dtype=spike_dtype, device=x_seq.device)
    v_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)
    h_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)

    _launch_lif_forward_kernel(
        x_storage,
        v_storage,
        s_seq,
        h_seq,
        v_seq,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=compute_tl_dtype,
        save_intermediates=True,
        store_v_seq=True,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_lif_mp_forward")
def _multistep_lif_mp_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    backward_compute_dtype_id: int,
    spike_dtype_id: int,
):
    del (
        v_init,
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        sg_triton_id,
        sg_alpha,
        forward_compute_dtype_id,
        backward_compute_dtype_id,
    )
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    return (
        torch.empty(x_seq.shape, dtype=spike_dtype, device=x_seq.device),
        torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device),
        torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device),
    )


def multistep_lif_mp_with_plan(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    plan: TritonNeuronExecutionPlan,
    *,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool = False,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if plan.neuron_type != "lif":
        raise ValueError(f"LIF forward requires a LIF plan, got {plan.neuron_type!r}.")
    _check_plan_inputs(x_seq, v_init, plan, "LIF")
    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    if torch.is_grad_enabled() and (x_seq.requires_grad or v_init.requires_grad):
        if surrogate_function is None:
            surrogate_function = surrogate.Sigmoid()
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, h_seq = multistep_lif_mp_forward(
            x_seq,
            v_init,
            decay_input,
            tau,
            v_threshold,
            v_reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
            plan.storage_dtype_id,
            plan.forward_compute_dtype_id,
            plan.backward_compute_dtype_id,
            plan.spike_dtype_id,
        )
        return s_seq, v_seq, (h_seq if plan.save_intermediates else None)
    s_seq, v_seq, h_seq = multistep_lif_mp_inference(
        x_seq,
        v_init,
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        plan.storage_dtype_id,
        plan.forward_compute_dtype_id,
        plan.spike_dtype_id,
        plan.save_intermediates,
    )
    return s_seq, v_seq, (h_seq if plan.save_intermediates else None)


def multistep_lif_mp(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
    decay_input: bool,
    tau: float,
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
    Experimental mixed-precision multi-step LIF forward path using the same
    Triton forward kernel source as :func:`multistep_lif`.

    This path is intended for FP8 storage experiments where storage dtype,
    forward compute dtype, and backward compute dtype must be controlled
    independently.

    .. warning::
        When ``compute_dtype='fp8'``, the LIF recurrence and threshold comparison
        are performed in FP8 precision. This mode has limited dynamic range and
        mantissa bits, and may produce incorrect spike patterns. Use it only for
        experiments, not for accuracy-critical inference.
    """
    plan = prepare_triton_neuron_execution_plan(
        neuron_type="lif",
        device=x_seq.device,
        storage_dtype=storage_dtype,
        forward_compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=spike_dtype,
        save_intermediates=save_intermediates,
    )
    return multistep_lif_mp_with_plan(
        x_seq,
        v_init,
        plan,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )


multistep_lif_mixed_precision_forward = multistep_lif_mp
multistep_lif_mixed_precision_forward_with_plan = multistep_lif_mp_with_plan


def _setup_mp_lif_context(ctx, inputs, output):
    (
        x_seq,
        v_init,
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        sg_triton_id,
        sg_alpha,
        storage_dtype_id,
        forward_compute_dtype_id,
        backward_compute_dtype_id,
        spike_dtype_id,
    ) = inputs
    del forward_compute_dtype_id
    h_seq = output[2]
    ctx.save_for_backward(h_seq)
    ctx.x_dtype = x_seq.dtype
    ctx.v_init_dtype = v_init.dtype
    ctx.decay_input = decay_input
    ctx.tau = tau
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha
    ctx.storage_dtype_id = storage_dtype_id
    ctx.backward_compute_dtype_id = backward_compute_dtype_id
    ctx.spike_dtype_id = spike_dtype_id


def _multistep_lif_mp_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    del grad_h_seq
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(ctx.storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(ctx.spike_dtype_id)
    if grad_s_seq is None:
        grad_s_seq = torch.zeros(h_seq.shape, dtype=spike_dtype, device=h_seq.device)
    if grad_v_seq is None:
        grad_v_seq = torch.zeros(h_seq.shape, dtype=storage_dtype, device=h_seq.device)
    grad_s_seq = grad_s_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    grad_x_seq = torch.empty(h_seq.shape, dtype=ctx.x_dtype, device=h_seq.device)
    grad_v_init = torch.empty(
        h_seq[0].shape, dtype=ctx.v_init_dtype, device=h_seq.device
    )

    _launch_lif_backward_kernel(
        grad_s_seq,
        grad_v_seq,
        h_seq,
        grad_x_seq,
        grad_v_init,
        tau=ctx.tau,
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        compute_dtype=triton_neuron_compute_dtype_id_to_tl_dtype(
            ctx.backward_compute_dtype_id, ctx.storage_dtype_id
        ),
        sg_triton_id=ctx.sg_triton_id,
        decay_input=ctx.decay_input,
        soft_reset=ctx.soft_reset,
        detach_reset=ctx.detach_reset,
        store_v_seq=True,
        use_torch_wrap=True,
    )
    return (
        grad_x_seq,
        grad_v_init,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "sj::multistep_lif_mp_forward",
    _multistep_lif_mp_backward,
    setup_context=_setup_mp_lif_context,
)


def _launch_single_step_lif_forward(
    x: torch.Tensor,
    v_init: torch.Tensor,
    spike: torch.Tensor,
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    save_intermediates: bool,
) -> None:
    NCL = x.numel()
    grid = (triton.cdiv(NCL, 256),)
    with torch.cuda.device(x.device):
        wrap_triton(_single_step_lif_forward_kernel)[grid](
            x,
            v_init,
            spike,
            h,
            v,
            tau,
            v_threshold,
            v_reset,
            NCL=NCL,
            BLOCK_NCL=256,
            compute_dtype=type_dict[x.dtype],
            decay_input=decay_input,
            soft_reset=soft_reset,
            save_intermediates=save_intermediates,
        )


def _launch_single_step_lif_backward(
    grad_spike: torch.Tensor,
    grad_v: torch.Tensor,
    h: torch.Tensor,
    grad_x: torch.Tensor,
    grad_v_init: torch.Tensor,
    *,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    sg_alpha: float,
    sg_triton_id: int,
    soft_reset: bool,
    detach_reset: bool,
) -> None:
    NCL = grad_spike.numel()
    grid = (triton.cdiv(NCL, 256),)
    with torch.cuda.device(grad_spike.device):
        wrap_triton(_single_step_lif_backward_kernel)[grid](
            grad_spike,
            grad_v,
            h,
            grad_x,
            grad_v_init,
            tau,
            v_threshold,
            v_reset,
            sg_alpha,
            NCL=NCL,
            BLOCK_NCL=256,
            compute_dtype=type_dict[grad_spike.dtype],
            sg_triton_id=sg_triton_id,
            decay_input=decay_input,
            soft_reset=soft_reset,
            detach_reset=detach_reset,
        )


@register_op("sj::single_step_lif_inference")
def single_step_lif_inference(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    v_init = v_init.contiguous()
    spike = torch.empty_like(x)
    v = torch.empty_like(v_init)
    _launch_single_step_lif_forward(
        x,
        v_init,
        spike,
        v,
        v,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        save_intermediates=False,
    )
    return spike, v


@torch.library.register_fake("sj::single_step_lif_inference")
def _single_step_lif_inference_fake(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
):
    return x.new_empty(x.shape), v_init.new_empty(v_init.shape)


@register_op("sj::single_step_lif_forward")
def single_step_lif_forward(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    v_init = v_init.contiguous()
    spike = torch.empty_like(x)
    v = torch.empty_like(v_init)
    h = torch.empty_like(x)
    _launch_single_step_lif_forward(
        x,
        v_init,
        spike,
        h,
        v,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        save_intermediates=True,
    )
    return spike, v, h


@torch.library.register_fake("sj::single_step_lif_forward")
def _single_step_lif_forward_fake(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_triton_id: int,
    sg_alpha: float,
):
    return (
        x.new_empty(x.shape),
        v_init.new_empty(v_init.shape),
        x.new_empty(x.shape),
    )


def _setup_single_step_context(ctx, inputs, output):
    (
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        sg_triton_id,
        sg_alpha,
    ) = inputs[2:]
    ctx.save_for_backward(output[2])
    ctx.decay_input = decay_input
    ctx.tau = tau
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha


def _single_step_lif_backward(ctx, grad_spike, grad_v, grad_h):
    (h,) = ctx.saved_tensors
    grad_spike = grad_spike.contiguous()
    grad_v = grad_v.contiguous()
    h = h.contiguous()
    grad_x = torch.empty_like(grad_spike)
    grad_v_init = torch.empty_like(grad_v)
    _launch_single_step_lif_backward(
        grad_spike,
        grad_v,
        h,
        grad_x,
        grad_v_init,
        decay_input=ctx.decay_input,
        tau=ctx.tau,
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        sg_triton_id=ctx.sg_triton_id,
        soft_reset=ctx.soft_reset,
        detach_reset=ctx.detach_reset,
    )
    return (
        grad_x,
        grad_v_init,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "sj::single_step_lif_forward",
    _single_step_lif_backward,
    setup_context=_setup_single_step_context,
)


def single_step_lif(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <single_step_lif-cn>` | :ref:`English <single_step_lif-en>`

    ----

    .. _single_step_lif-cn:

    * **中文**

    使用专用 Triton kernel 执行单步 LIF 状态转移。``x`` 和 ``v_init`` 必须是
    shape、dtype 和 device 相同的 CUDA FP32、FP16 或 BF16 张量。

    :param x: 单步 CUDA 输入，形状为 ``[N, *]``
    :type x: torch.Tensor
    :param v_init: 与 ``x`` 同形状的初始膜电位
    :type v_init: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param tau: 膜时间常数
    :type tau: float
    :param v_threshold: 发放阈值
    :type v_threshold: float
    :param v_reset: 硬复位电压；``None`` 表示软复位
    :type v_reset: Optional[float]
    :param detach_reset: 是否在反向传播中分离复位项
    :type detach_reset: bool
    :param surrogate_function: 反向传播使用的替代函数
    :type surrogate_function: surrogate.SurrogateFunctionBase
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: 当 ``x`` 和 ``v_init`` 的 shape、dtype 或 CUDA device
        不一致，或输入不在 CUDA 上时
    :raises NotImplementedError: 当 dtype 或替代函数不受 Triton 后端支持时

    ----

    .. _single_step_lif-en:

    * **English**

    Run one LIF state transition with the dedicated Triton kernel. ``x`` and
    ``v_init`` must be CUDA FP32, FP16, or BF16 tensors with matching shape,
    dtype, and device.

    :param x: Single-step CUDA input shaped ``[N, *]``
    :type x: torch.Tensor
    :param v_init: Initial membrane voltage with the same shape as ``x``
    :type v_init: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Hard-reset voltage; ``None`` selects soft reset
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach the reset term in backward
    :type detach_reset: bool
    :param surrogate_function: Surrogate function used in backward
    :type surrogate_function: surrogate.SurrogateFunctionBase
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If ``x`` and ``v_init`` differ in shape, dtype, or CUDA
        device, or if the input is not on CUDA
    :raises NotImplementedError: If the dtype or surrogate function is not
        supported by the Triton backend
    """
    if x.device.type != "cuda":
        raise ValueError("single_step_lif requires CUDA tensors.")
    if v_init.shape != x.shape or v_init.dtype != x.dtype or v_init.device != x.device:
        raise ValueError("x and v_init must have the same shape, dtype, and device.")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(x.dtype)
    soft_reset = v_reset is None
    reset = 0.0 if soft_reset else v_reset
    need_grad = torch.is_grad_enabled() and (x.requires_grad or v_init.requires_grad)
    if need_grad:
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        spike, v, _ = single_step_lif_forward(
            x,
            v_init,
            decay_input,
            tau,
            v_threshold,
            reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
        )
        return spike, v
    return single_step_lif_inference(
        x,
        v_init,
        decay_input,
        tau,
        v_threshold,
        reset,
        soft_reset,
    )


def _setup_context(ctx, inputs, output):
    (
        decay_input,
        tau,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        sg_triton_id,
        sg_alpha,
        store_v_seq,
    ) = inputs[2:]
    h_seq = output[2]
    ctx.save_for_backward(h_seq)
    ctx.decay_input = decay_input
    ctx.tau = tau
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha
    ctx.store_v_seq = store_v_seq


def _multistep_lif_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    grad_s_seq = grad_s_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v_init = torch.empty_like(h_seq[0])
    dtype = grad_s_seq.dtype
    _launch_lif_backward_kernel(
        grad_s_seq,
        grad_v_seq,
        h_seq,
        grad_x_seq,
        grad_v_init,
        tau=ctx.tau,
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        compute_dtype=type_dict[dtype],
        sg_triton_id=ctx.sg_triton_id,
        decay_input=ctx.decay_input,
        soft_reset=ctx.soft_reset,
        detach_reset=ctx.detach_reset,
        store_v_seq=ctx.store_v_seq,
        use_torch_wrap=True,
    )
    return (
        grad_x_seq,
        grad_v_init,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "sj::multistep_lif_forward",
    _multistep_lif_backward,
    setup_context=_setup_context,
)


def multistep_lif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function,
    store_v_seq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-step LIF neuron forward pass via Triton kernel.

    **API Language** - :ref:`中文 <multistep_lif-cn>` | :ref:`English <multistep_lif-en>`

    ----

    .. _multistep_lif-cn:

    * **中文**

    多步LIF神经元Triton kernel前向传播

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :type x_seq: ``torch.Tensor``
    :param v_init: Initial membrane potential
    :type v_init: ``torch.Tensor``
    :param decay_input: Whether input participates in decay
    :type decay_input: bool
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Threshold voltage
    :type v_threshold: float
    :param v_reset: Reset voltage (``None`` for soft reset)
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach the reset term in backward
    :type detach_reset: bool
    :param surrogate_function: Surrogate gradient function
    :type surrogate_function: ``surrogate.SurrogateFunctionBase``
    :param store_v_seq: 是否返回完整的膜电位序列，默认为 ``True``。设置为 ``False`` 时，
        第二个输出仅包含最终膜电位，其形状与 ``v_init`` 相同。
    :type store_v_seq: bool
    :return: 当 ``store_v_seq=True`` 时返回 ``(spike_seq, v_seq)``，否则返回
        ``(spike_seq, v_last)``，其中 ``v_last`` 的形状与 ``v_init`` 相同。
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _multistep_lif-en:

    * **English**

    Multi-step LIF neuron Triton kernel forward

    :param x_seq: Input sequence, shape ``[T, N, *]``
    :param v_init: Initial membrane potential
    :param decay_input: Whether input participates in decay
    :param tau: Membrane time constant
    :param v_threshold: Threshold voltage
    :param v_reset: Reset voltage (``None`` for soft reset)
    :param detach_reset: Whether to detach the reset term in backward
    :param surrogate_function: Surrogate gradient function
    :type x_seq: ``torch.Tensor``
    :type v_init: ``torch.Tensor``
    :type decay_input: bool
    :type tau: float
    :type v_threshold: float
    :type v_reset: Optional[float]
    :type detach_reset: bool
    :type surrogate_function: ``surrogate.SurrogateFunctionBase``
    :param store_v_seq: Whether to return the full membrane-potential sequence.
        Defaults to ``True``. If ``False``, the second output contains only the
        final membrane potential and has the same shape as ``v_init``.
    :type store_v_seq: bool
    :return: Tuple of ``(spike_seq, v_seq)`` when ``store_v_seq=True`` or
        ``(spike_seq, v_last)`` otherwise, where ``v_last`` has the same shape as
        ``v_init``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v_init.requires_grad
    )
    if need_grad:
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, _ = multistep_lif_forward(
            x_seq,
            v_init,
            decay_input,
            tau,
            v_threshold,
            v_reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
            store_v_seq,
        )
    else:
        s_seq, v_seq = multistep_lif_inference(
            x_seq,
            v_init,
            decay_input,
            tau,
            v_threshold,
            v_reset,
            soft_reset,
            store_v_seq,
        )
    return s_seq, v_seq
