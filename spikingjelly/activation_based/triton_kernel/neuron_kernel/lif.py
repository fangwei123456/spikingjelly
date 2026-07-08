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

    logging.info(f"spikingjelly.activation_based.triton_kernel.neuron_kernel.lif: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()


__all__ = ["multistep_lif"]


_COMPUTE_DTYPE_ALIASES = {
    "float32": "fp32",
    "torch.float32": "fp32",
    "float": "fp32",
    "fp32": "fp32",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "bf16": "bf16",
    "float8": "fp8",
    "fp8": "fp8",
}


def _normalize_compute_dtype_name(compute_dtype) -> str:
    if isinstance(compute_dtype, torch.dtype):
        if compute_dtype == torch.float32:
            return "fp32"
        if compute_dtype == torch.float16:
            return "fp16"
        if hasattr(torch, "bfloat16") and compute_dtype == torch.bfloat16:
            return "bf16"
        raise ValueError(f"Unsupported LIF Triton compute dtype: {compute_dtype}.")
    if not isinstance(compute_dtype, str):
        raise ValueError(
            "compute_dtype must be a string or torch.dtype, "
            f"but got {type(compute_dtype).__name__}."
        )
    key = compute_dtype.lower()
    try:
        return _COMPUTE_DTYPE_ALIASES[key]
    except KeyError as e:
        raise ValueError(
            "compute_dtype must be one of 'fp8', 'fp16', 'bf16', or 'fp32', "
            f"but got {compute_dtype!r}."
        ) from e


def _normalize_storage_dtype(storage_dtype) -> torch.dtype:
    if isinstance(storage_dtype, torch.dtype):
        dtype = storage_dtype
    elif isinstance(storage_dtype, str):
        key = storage_dtype.lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if hasattr(torch, "bfloat16"):
            mapping.update({"bfloat16": torch.bfloat16, "bf16": torch.bfloat16})
        if hasattr(torch, "float8_e4m3fn"):
            mapping.update(
                {
                    "float8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "e4m3fn": torch.float8_e4m3fn,
                }
            )
        if hasattr(torch, "float8_e5m2"):
            mapping.update(
                {
                    "float8_e5m2": torch.float8_e5m2,
                    "fp8_e5m2": torch.float8_e5m2,
                    "e5m2": torch.float8_e5m2,
                }
            )
        try:
            dtype = mapping[key]
        except KeyError as e:
            raise ValueError(
                f"Unsupported LIF Triton storage dtype: {storage_dtype!r}."
            ) from e
    else:
        raise ValueError(
            "storage_dtype must be a string or torch.dtype, "
            f"but got {type(storage_dtype).__name__}."
        )

    allowed = {torch.float32, torch.float16}
    if hasattr(torch, "bfloat16"):
        allowed.add(torch.bfloat16)
    if hasattr(torch, "float8_e4m3fn"):
        allowed.add(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        allowed.add(torch.float8_e5m2)
    if dtype not in allowed:
        raise ValueError(f"Unsupported LIF Triton storage dtype: {dtype}.")
    return dtype


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return (
        (hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn)
        or (hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2)
    )


def _resolve_compute_tl_dtype(
    compute_dtype, storage_dtype: Optional[torch.dtype] = None
):
    name = _normalize_compute_dtype_name(compute_dtype)
    if name == "fp32":
        return type_dict[torch.float32]
    if name == "fp16":
        return type_dict[torch.float16]
    if name == "bf16":
        if not hasattr(torch, "bfloat16") or torch.bfloat16 not in type_dict:
            raise ValueError("Triton bfloat16 compute dtype is unavailable.")
        return type_dict[torch.bfloat16]
    if name == "fp8":
        if storage_dtype is None:
            raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
        storage_dtype = _normalize_storage_dtype(storage_dtype)
        if not _is_fp8_dtype(storage_dtype):
            raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
        if hasattr(torch, "float8_e4m3fn") and storage_dtype == torch.float8_e4m3fn:
            tl_dtype = getattr(tl, "float8e4nv", None)
            if tl_dtype is None:
                raise ValueError("Triton float8e4nv dtype is unavailable.")
            return tl_dtype
        if hasattr(torch, "float8_e5m2") and storage_dtype == torch.float8_e5m2:
            tl_dtype = getattr(tl, "float8e5", None)
            if tl_dtype is None:
                raise ValueError("Triton float8e5 dtype is unavailable.")
            return tl_dtype
    raise ValueError(f"Unsupported LIF Triton compute dtype: {compute_dtype!r}.")


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
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

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
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

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
    dtype: tl.constexpr,  # grad_s_seq.dtype; might != h_seq or s_seq.dtype
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    r_tau = tl.full([1], 1.0 / tau, dtype=dtype)
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
        grad_v_acc = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc)
            else:
                grad_h = tl.fma(grad_s - v_threshold * grad_v_acc, sg, grad_v_acc)
        else:
            s = (h >= v_threshold).to(dtype)
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
    key=["NCL", "dtype", "soft_reset", "detach_reset"],
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
    dtype: tl.constexpr,
    sg_triton_id: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    r_tau = tl.full([1], 1.0 / tau, dtype=dtype)
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
        grad_v_acc = grad_v + grad_v_acc
        if soft_reset:
            if detach_reset:
                grad_h = tl.fma(grad_s, sg, grad_v_acc)
            else:
                grad_h = tl.fma(grad_s - v_threshold * grad_v_acc, sg, grad_v_acc)
        else:
            s = (h >= v_threshold).to(dtype)
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
            tau,
            v_threshold,
            v_reset,
            T=T,
            NCL=NCL,
            compute_dtype=compute_dtype,
            decay_input=decay_input,
            soft_reset=soft_reset,
            save_intermediates=save_intermediates,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
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
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()

    s_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
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
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
    )


def multistep_lif_mixed_precision_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    storage_dtype,
    compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Experimental mixed-precision multi-step LIF forward path using the same
    Triton forward kernel source as :func:`multistep_lif`.

    This path is forward-only. It is intended for FP8 storage experiments where
    storage dtype and recurrence compute dtype must be controlled independently.
    """
    storage_dtype = _normalize_storage_dtype(storage_dtype)
    compute_dtype_name = _normalize_compute_dtype_name(compute_dtype)
    if compute_dtype_name == "fp8" and not _is_fp8_dtype(storage_dtype):
        raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
    if spike_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "spike_dtype must be torch.float32, torch.float16, or torch.bfloat16, "
            f"but got {spike_dtype}."
        )
    if not isinstance(save_intermediates, bool):
        raise ValueError("save_intermediates must be bool.")
    if torch.is_grad_enabled() and (x_seq.requires_grad or v_init.requires_grad):
        raise NotImplementedError("FP8 Triton LIF backward is not implemented yet.")
    if _is_fp8_dtype(storage_dtype):
        from ..fp8_capability import triton_fp8_neuron_capability_report

        report = triton_fp8_neuron_capability_report(x_seq.device)
        dtype_report = report.get("dtypes", {}).get(str(storage_dtype), {})
        if not dtype_report.get("available", False):
            reason = dtype_report.get("reason", "unknown reason")
            raise RuntimeError(
                f"Triton FP8 LIF forward is unavailable for {storage_dtype}: {reason}"
            )
    if x_seq.device.type != "cuda" or v_init.device.type != "cuda":
        raise RuntimeError("Mixed-precision Triton LIF forward requires CUDA tensors.")
    if x_seq.device != v_init.device:
        raise RuntimeError("x_seq and v_init must be on the same CUDA device.")
    compute_tl_dtype = _resolve_compute_tl_dtype(compute_dtype_name, storage_dtype)

    soft_reset = v_reset is None
    v_reset = v_reset if v_reset is not None else 0.0
    x_storage = x_seq.detach().to(dtype=storage_dtype).contiguous()
    v_storage = v_init.detach().to(dtype=storage_dtype).contiguous()
    s_seq = torch.empty(x_seq.shape, dtype=spike_dtype, device=x_seq.device)
    v_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)
    if save_intermediates:
        h_seq = torch.empty(x_seq.shape, dtype=storage_dtype, device=x_seq.device)
    else:
        h_seq = v_seq

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
        save_intermediates=save_intermediates,
        use_torch_wrap=False,
    )
    return s_seq, v_seq, h_seq if save_intermediates else None


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


def _multistep_lif_backward(ctx, grad_s_seq, grad_v_seq, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    grad_s_seq = grad_s_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v_init = torch.empty_like(grad_v_seq[0])
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    with torch.cuda.device(grad_s_seq.device.index):
        wrap_triton(_select_backward_kernel(T))[grid](
            grad_s_seq,
            grad_v_seq,
            h_seq,
            grad_x_seq,
            grad_v_init,
            ctx.tau,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.sg_alpha,
            T=T,
            NCL=NCL,
            dtype=type_dict[dtype],
            sg_triton_id=ctx.sg_triton_id,
            decay_input=ctx.decay_input,
            soft_reset=ctx.soft_reset,
            detach_reset=ctx.detach_reset,
        )
    return grad_x_seq, grad_v_init, None, None, None, None, None, None, None, None


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
    :return: Tuple of (spike_seq, v_seq)
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
        )
    return s_seq, v_seq
