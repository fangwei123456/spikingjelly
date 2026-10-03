from spikingjelly.logger import logger
from typing import Optional

import torch

from ..._neuron_layout import _empty_like
from .utils import (
    _spatial_offsets,
    _time_offset,
    _triton_layout_args,
    _neuron_grid,
    _neuron_indices,
    _block_minor,
)

from ... import surrogate
from ..surrogate_kernel import resolve_sg_triton_id_and_alpha, sg_triton
from ..triton_utils import (
    do_bench_cudagraph,
    register_op,
    triton_neuron_compute_dtype_id_to_tl_dtype,
    triton_neuron_dtype_id_to_torch_dtype,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)
from .utils import (
    _TritonNeuronExecutionPlan,
    _check_mp_cuda_inputs,
    _check_plan_inputs,
    _prepare_triton_neuron_execution_plan,
)

try:
    import triton
    import triton.language as tl
except (ImportError, OSError) as e:
    from .. import dummy

    logger.debug("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()

__all__ = ["multistep_if"]


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "T",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "save_intermediates",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    s_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    v_init_ptrs = v_init_ptr + v_init_ptr_offsets
    v = tl.load(v_init_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.static_range(0, T, 1):
        x_ptrs = x_seq_ptr + x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(compute_dtype)

        h = v + x
        s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = s_seq_ptr + s_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        tl.store(s_ptrs, s, mask=mask)
        if store_v_seq:
            v_ptrs = v_seq_ptr + v_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 4)
            tl.store(v_ptrs, v, mask=mask)
        if save_intermediates:
            h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
            tl.store(h_ptrs, h, mask=mask)

    if not store_v_seq:
        v_last_ptrs = v_seq_ptr + v_seq_ptr_offsets
        tl.store(v_last_ptrs, v, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "save_intermediates",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    s_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    v_init_ptrs = v_init_ptr + v_init_ptr_offsets
    v = tl.load(v_init_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.range(0, T, 1):
        x_ptrs = x_seq_ptr + x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(compute_dtype)

        h = v + x
        s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = s_seq_ptr + s_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        tl.store(s_ptrs, s, mask=mask)
        if store_v_seq:
            v_ptrs = v_seq_ptr + v_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 4)
            tl.store(v_ptrs, v, mask=mask)
        if save_intermediates:
            h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
            tl.store(h_ptrs, h, mask=mask)

    if not store_v_seq:
        v_last_ptrs = v_seq_ptr + v_seq_ptr_offsets
        tl.store(v_last_ptrs, v, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "T",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "detach_reset",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    grad_s_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    grad_v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    grad_x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    grad_v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    if store_v_seq:
        grad_v_acc = tl.zeros(indices.shape, dtype=compute_dtype)
    else:
        grad_v_last_ptrs = grad_v_seq_ptr + grad_v_seq_ptr_offsets
        grad_v_acc = tl.load(grad_v_last_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = (
            grad_s_seq_ptr + grad_s_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        )
        grad_s = tl.load(grad_s_ptrs, mask=mask, other=0.0).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = (
                grad_v_seq_ptr
                + grad_v_seq_ptr_offsets
                + _time_offset(t, NCL, STRIDES, 1)
            )
            grad_v = tl.load(grad_v_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        h = tl.load(h_ptrs, mask=mask, other=0.0).to(compute_dtype)

        sg = sg_triton(h.to(tl.float32) - v_threshold, sg_alpha, sg_triton_id).to(
            compute_dtype
        )
        if store_v_seq:
            grad_v_combined = grad_v + grad_v_acc
        else:
            grad_v_combined = grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_combined)
            else:
                grad_h = tl.fma(
                    grad_s - v_threshold * grad_v_combined, sg, grad_v_combined
                )
        else:
            s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
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

        grad_x_ptrs = (
            grad_x_seq_ptr + grad_x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
        )
        tl.store(grad_x_ptrs, grad_x, mask=mask)

    grad_v_init_ptrs = grad_v_init_ptr + grad_v_init_ptr_offsets
    tl.store(grad_v_init_ptrs, grad_v_acc, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "NCL",
        "compute_dtype",
        "soft_reset",
        "detach_reset",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    compute_dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    grad_s_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    grad_v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    grad_x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    grad_v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)

    if store_v_seq:
        grad_v_acc = tl.zeros(indices.shape, dtype=compute_dtype)
    else:
        grad_v_last_ptrs = grad_v_seq_ptr + grad_v_seq_ptr_offsets
        grad_v_acc = tl.load(grad_v_last_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.range(T - 1, -1, -1):
        grad_s_ptrs = (
            grad_s_seq_ptr + grad_s_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        )
        grad_s = tl.load(grad_s_ptrs, mask=mask, other=0.0).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = (
                grad_v_seq_ptr
                + grad_v_seq_ptr_offsets
                + _time_offset(t, NCL, STRIDES, 1)
            )
            grad_v = tl.load(grad_v_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        h = tl.load(h_ptrs, mask=mask, other=0.0).to(compute_dtype)

        sg = sg_triton(h.to(tl.float32) - v_threshold, sg_alpha, sg_triton_id).to(
            compute_dtype
        )
        if store_v_seq:
            grad_v_combined = grad_v + grad_v_acc
        else:
            grad_v_combined = grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_combined)
            else:
                grad_h = tl.fma(
                    grad_s - v_threshold * grad_v_combined, sg, grad_v_combined
                )
        else:
            s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
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

        grad_x_ptrs = (
            grad_x_seq_ptr + grad_x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
        )
        tl.store(grad_x_ptrs, grad_x, mask=mask)

    grad_v_init_ptrs = grad_v_init_ptr + grad_v_init_ptr_offsets
    tl.store(grad_v_init_ptrs, grad_v_acc, mask=mask)


def _select_forward_kernel(T: int):
    if use_static_range_for_triton_neuron_kernel(T):
        return _multistep_if_forward_kernel_static
    return _multistep_if_forward_kernel_dynamic


def _select_backward_kernel(T: int):
    if use_static_range_for_triton_neuron_kernel(T):
        return _multistep_if_backward_kernel_static
    return _multistep_if_backward_kernel_dynamic


def _launch_if_forward_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    s_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    compute_dtype,
    save_intermediates: bool,
    store_v_seq: bool,
    use_torch_wrap: bool,
) -> None:
    sizes, strides = _triton_layout_args(x_seq, x_seq, v_init, s_seq, h_seq, v_seq)
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()

    def grid(meta):
        return _neuron_grid(NCL, sizes, meta["BLOCK_NCL"], meta["BLOCK_MINOR"])

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
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            BLOCK_MINOR=_block_minor(sizes, strides),
            SIZES=sizes,
            STRIDES=strides,
            compute_dtype=compute_dtype,
            soft_reset=soft_reset,
            save_intermediates=save_intermediates,
            store_v_seq=store_v_seq,
        )


def _launch_if_backward_kernel(
    grad_s_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    grad_x_seq: torch.Tensor,
    grad_v_init: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float,
    sg_alpha: float,
    compute_dtype,
    sg_triton_id: int,
    soft_reset: bool,
    detach_reset: bool,
    store_v_seq: bool,
    use_torch_wrap: bool,
) -> None:
    sizes, strides = _triton_layout_args(
        h_seq, grad_s_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init
    )
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()

    def grid(meta):
        return _neuron_grid(NCL, sizes, meta["BLOCK_NCL"], meta["BLOCK_MINOR"])

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
            v_threshold,
            v_reset,
            sg_alpha,
            T=T,
            NCL=NCL,
            BLOCK_MINOR=_block_minor(sizes, strides),
            SIZES=sizes,
            STRIDES=strides,
            compute_dtype=compute_dtype,
            sg_triton_id=sg_triton_id,
            soft_reset=soft_reset,
            detach_reset=detach_reset,
            store_v_seq=store_v_seq,
        )


@register_op("sj::multistep_if_inference")
def multistep_if_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_seq = _empty_like(x_seq)
    v_seq = _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False)
    dtype = x_seq.dtype
    _launch_if_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        v_seq,  # dummy
        v_seq,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=False,
        store_v_seq=store_v_seq,
        use_torch_wrap=True,
    )
    return s_seq, v_seq


@torch.library.register_fake("sj::multistep_if_inference")
def _multistep_if_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
):
    return (
        _empty_like(x_seq),
        _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False),
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
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_seq = _empty_like(x_seq)
    v_seq = _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False)
    h_seq = _empty_like(x_seq)
    dtype = x_seq.dtype
    _launch_if_forward_kernel(
        x_seq,
        v_init,
        s_seq,
        h_seq,
        v_seq,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=type_dict[dtype],
        save_intermediates=True,
        store_v_seq=store_v_seq,
        use_torch_wrap=True,
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
    store_v_seq: bool,
):
    return (
        _empty_like(x_seq),
        _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False),
        _empty_like(x_seq),
    )


@register_op("sj::multistep_if_mp_inference")
def multistep_if_mp_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    spike_dtype_id: int,
    save_intermediates: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_mp_cuda_inputs(x_seq, v_init, "IF")
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    compute_tl_dtype = triton_neuron_compute_dtype_id_to_tl_dtype(
        forward_compute_dtype_id, storage_dtype_id
    )
    x_storage = x_seq.detach().to(dtype=storage_dtype)
    v_storage = v_init.detach().to(dtype=storage_dtype)
    s_seq = _empty_like(x_seq, dtype=spike_dtype)
    v_seq = _empty_like(x_seq, dtype=storage_dtype)
    if save_intermediates:
        h_seq = _empty_like(x_seq, dtype=storage_dtype)
        h_buffer = h_seq
    else:
        h_seq = torch.empty((0,), dtype=storage_dtype, device=x_seq.device)
        h_buffer = v_seq

    _launch_if_forward_kernel(
        x_storage,
        v_storage,
        s_seq,
        h_buffer,
        v_seq,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=compute_tl_dtype,
        save_intermediates=save_intermediates,
        store_v_seq=True,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_if_mp_inference")
def _multistep_if_mp_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    storage_dtype_id: int,
    forward_compute_dtype_id: int,
    spike_dtype_id: int,
    save_intermediates: bool,
):
    del v_init, v_threshold, v_reset, soft_reset, forward_compute_dtype_id
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    return (
        _empty_like(x_seq, dtype=spike_dtype),
        _empty_like(x_seq, dtype=storage_dtype),
        _empty_like(x_seq, dtype=storage_dtype)
        if save_intermediates
        else torch.empty((0,), dtype=storage_dtype, device=x_seq.device),
    )


@register_op("sj::multistep_if_mp_forward")
def multistep_if_mp_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
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
    _check_mp_cuda_inputs(x_seq, v_init, "IF")
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(spike_dtype_id)
    compute_tl_dtype = triton_neuron_compute_dtype_id_to_tl_dtype(
        forward_compute_dtype_id, storage_dtype_id
    )
    x_storage = x_seq.to(dtype=storage_dtype)
    v_storage = v_init.to(dtype=storage_dtype)
    s_seq = _empty_like(x_seq, dtype=spike_dtype)
    v_seq = _empty_like(x_seq, dtype=storage_dtype)
    h_seq = _empty_like(x_seq, dtype=storage_dtype)

    _launch_if_forward_kernel(
        x_storage,
        v_storage,
        s_seq,
        h_seq,
        v_seq,
        v_threshold=v_threshold,
        v_reset=v_reset,
        soft_reset=soft_reset,
        compute_dtype=compute_tl_dtype,
        save_intermediates=True,
        store_v_seq=True,
        use_torch_wrap=True,
    )
    return s_seq, v_seq, h_seq


@torch.library.register_fake("sj::multistep_if_mp_forward")
def _multistep_if_mp_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
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
        _empty_like(x_seq, dtype=spike_dtype),
        _empty_like(x_seq, dtype=storage_dtype),
        _empty_like(x_seq, dtype=storage_dtype),
    )


def _multistep_if_mp_with_plan(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    plan: _TritonNeuronExecutionPlan,
    *,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool = False,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if plan.neuron_type != "if":
        raise ValueError(f"IF forward requires an IF plan, got {plan.neuron_type!r}.")
    _check_plan_inputs(x_seq, v_init, plan, "IF")
    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    if torch.is_grad_enabled() and (x_seq.requires_grad or v_init.requires_grad):
        if surrogate_function is None:
            surrogate_function = surrogate.Sigmoid()
        sg_triton_id, sg_alpha = resolve_sg_triton_id_and_alpha(surrogate_function)
        s_seq, v_seq, h_seq = multistep_if_mp_forward(
            x_seq,
            v_init,
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
    s_seq, v_seq, h_seq = multistep_if_mp_inference(
        x_seq,
        v_init,
        v_threshold,
        v_reset,
        soft_reset,
        plan.storage_dtype_id,
        plan.forward_compute_dtype_id,
        plan.spike_dtype_id,
        plan.save_intermediates,
    )
    return s_seq, v_seq, (h_seq if plan.save_intermediates else None)


def _multistep_if_mp(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
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
    Experimental mixed-precision multi-step IF forward path using the same
    Triton forward kernel source as :func:`multistep_if`.

    This path is intended for FP8 storage experiments where storage dtype,
    forward compute dtype, and backward compute dtype must be controlled
    independently.

    .. warning::
        When ``compute_dtype='fp8'``, the IF recurrence and threshold comparison
        are performed in FP8 precision. This mode has limited dynamic range and
        mantissa bits, and may produce incorrect spike patterns. Use it only for
        experiments, not for accuracy-critical inference.
    """
    plan = _prepare_triton_neuron_execution_plan(
        neuron_type="if",
        device=x_seq.device,
        storage_dtype=storage_dtype,
        forward_compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=spike_dtype,
        save_intermediates=save_intermediates,
    )
    return _multistep_if_mp_with_plan(
        x_seq,
        v_init,
        plan,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )


def _setup_mp_if_context(ctx, inputs, output):
    (
        x_seq,
        v_init,
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
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha
    ctx.storage_dtype_id = storage_dtype_id
    ctx.backward_compute_dtype_id = backward_compute_dtype_id
    ctx.spike_dtype_id = spike_dtype_id


def _multistep_if_mp_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    del grad_h_seq
    storage_dtype = triton_neuron_dtype_id_to_torch_dtype(ctx.storage_dtype_id)
    spike_dtype = triton_neuron_dtype_id_to_torch_dtype(ctx.spike_dtype_id)
    if grad_s_seq is None:
        grad_s_seq = torch.zeros(h_seq.shape, dtype=spike_dtype, device=h_seq.device)
    if grad_v_seq is None:
        grad_v_seq = torch.zeros(h_seq.shape, dtype=storage_dtype, device=h_seq.device)
    grad_x_seq = _empty_like(h_seq, dtype=ctx.x_dtype)
    grad_v_init = _empty_like(h_seq[0], dtype=ctx.v_init_dtype, sequence=False)

    _launch_if_backward_kernel(
        grad_s_seq,
        grad_v_seq,
        h_seq,
        grad_x_seq,
        grad_v_init,
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        compute_dtype=triton_neuron_compute_dtype_id_to_tl_dtype(
            ctx.backward_compute_dtype_id, ctx.storage_dtype_id
        ),
        sg_triton_id=ctx.sg_triton_id,
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
    )


torch.library.register_autograd(
    "sj::multistep_if_mp_forward",
    _multistep_if_mp_backward,
    setup_context=_setup_mp_if_context,
)


def _setup_context(ctx, inputs, output):
    (
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
    ctx.v_threshold = v_threshold
    ctx.v_reset = v_reset
    ctx.soft_reset = soft_reset
    ctx.detach_reset = detach_reset
    ctx.sg_triton_id = sg_triton_id
    ctx.sg_alpha = sg_alpha
    ctx.store_v_seq = store_v_seq


def _multistep_if_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    if h_seq.numel() == 0:
        raise RuntimeError("backward called without saved intermediates")

    grad_x_seq = _empty_like(h_seq)
    grad_v_init = _empty_like(h_seq[0], sequence=False)
    dtype = grad_s_seq.dtype
    if dtype not in type_dict:
        raise NotImplementedError(dtype)
    _launch_if_backward_kernel(
        grad_s_seq,
        grad_v_seq,
        h_seq,
        grad_x_seq,
        grad_v_init,
        v_threshold=ctx.v_threshold,
        v_reset=ctx.v_reset,
        sg_alpha=ctx.sg_alpha,
        compute_dtype=type_dict[dtype],
        sg_triton_id=ctx.sg_triton_id,
        soft_reset=ctx.soft_reset,
        detach_reset=ctx.detach_reset,
        store_v_seq=ctx.store_v_seq,
        use_torch_wrap=True,
    )
    return grad_x_seq, grad_v_init, None, None, None, None, None, None, None


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
    store_v_seq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-step IF neuron forward pass via Triton kernel.

    **API Language** - :ref:`中文 <multistep_if-cn>` | :ref:`English <multistep_if-en>`

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
    :param store_v_seq: 是否返回完整的膜电位序列，默认为 ``True``。设置为 ``False`` 时，
        第二个输出仅包含最终膜电位，其形状与 ``v_init`` 相同。
    :type store_v_seq: bool
    :return: 当 ``store_v_seq=True`` 时返回 ``(spike_seq, v_seq)``，否则返回
        ``(spike_seq, v_last)``，其中 ``v_last`` 的形状与 ``v_init`` 相同。
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
        s_seq, v_seq, _ = multistep_if_forward(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            soft_reset,
            detach_reset,
            sg_triton_id,
            sg_alpha,
            store_v_seq,
        )
    else:
        s_seq, v_seq = multistep_if_inference(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            soft_reset,
            store_v_seq,
        )
    return s_seq, v_seq
