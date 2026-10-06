from dataclasses import dataclass

import torch

from spikingjelly.logger import logger

from .layout import _layout_args
from .triton_runtime import (
    is_fp8_dtype,
    normalize_cuda_device,
    normalize_triton_compute_dtype_name,
    normalize_triton_storage_dtype,
    resolve_triton_compute_dtype,
    tl,
    torch_dtype_to_triton_neuron_dtype_id,
    triton,
    triton_compute_dtype_name_to_neuron_dtype_id,
)


def _triton_layout_args(reference, *tensors):
    if all(x.is_contiguous() for x in tensors):
        return (), ()
    return _layout_args(reference, *tensors)


def _block_minor(sizes, strides):
    # Matching spatial strides need no transpose tile, including channels-last.
    if not sizes or all(s[1:] == strides[0][1:] for s in strides[1:]):
        return 1
    return min(64, triton.next_power_of_2(sizes[0])) if len(sizes) > 1 else 1


def _neuron_grid(n, sizes, block, minor):
    if minor == 1:
        return (triton.cdiv(n, block),)
    return (triton.cdiv(sizes[0], minor) * triton.cdiv(n // sizes[0], block // minor),)


@triton.jit
def _neuron_indices(
    NCL: tl.constexpr, BLOCK: tl.constexpr, SIZES: tl.constexpr, MINOR: tl.constexpr
):
    pid = tl.program_id(0)
    if MINOR == 1:
        indices = (pid * BLOCK + tl.arange(0, BLOCK))[None, :]
        mask = indices < NCL
    else:
        inner_size: tl.constexpr = tl.constexpr(SIZES).value[0]
        minor_blocks: tl.constexpr = tl.cdiv(inner_size, MINOR)
        inner = (pid % minor_blocks) * MINOR + tl.arange(0, MINOR)[None, :]
        outer = (pid // minor_blocks) * (BLOCK // MINOR) + tl.arange(0, BLOCK // MINOR)[
            :, None
        ]
        indices = outer * inner_size + inner
        mask = (inner < inner_size) & (indices < NCL)
    return indices, mask


@triton.jit
def _time_offset(t, n: tl.constexpr, layouts: tl.constexpr, slot: tl.constexpr):
    stride: tl.constexpr = (
        n if len(layouts) == 0 else tl.constexpr(layouts).value[slot][0]
    )
    # Widen the stride before multiplying: the time offset can exceed int32.
    return t * tl.full((), stride, tl.int64)


@triton.jit
def _spatial_offsets(
    index, sizes: tl.constexpr, layouts: tl.constexpr, slot: tl.constexpr = 0
):
    if len(layouts) == 0:
        return index.to(tl.int64)
    else:
        # Unwrap constexpr tuples for Triton 3.3. The metadata-only predicate
        # uses scalar IR constants, which fold away before GPU code generation.
        strides: tl.constexpr = tl.constexpr(layouts).value[slot]
        dense = tl.full((), True, tl.int1)
        span = tl.full((), 1, tl.int64)
        for d in tl.static_range(len(sizes)):
            size = tl.full((), tl.constexpr(sizes).value[d], tl.int64)
            stride = tl.full((), tl.constexpr(strides).value[d + 1], tl.int64)
            dense = dense & ((size == 1) | (stride == span))
            span = span * size
        if dense:
            offset = index.to(tl.int64)
        else:
            offset = tl.full(index.shape, 0, tl.int64)
            for d in tl.static_range(len(sizes)):
                offset += (index % tl.constexpr(sizes).value[d]).to(
                    tl.int64
                ) * tl.constexpr(strides).value[d + 1]
                index = index // tl.constexpr(sizes).value[d]
        return offset


_SUPPORTED_PLAN_NEURON_TYPES = frozenset({"if", "lif", "plif"})
_TRITON_NEURON_EXECUTION_PLANS = {}


@dataclass(frozen=True)
class _TritonNeuronExecutionPlan:
    neuron_type: str
    device: torch.device
    storage_dtype: torch.dtype
    forward_compute_dtype_name: str
    forward_compute_tl_dtype: object
    backward_compute_dtype_name: str
    backward_compute_tl_dtype: object
    spike_dtype: torch.dtype
    storage_dtype_id: int
    forward_compute_dtype_id: int
    backward_compute_dtype_id: int
    spike_dtype_id: int
    save_intermediates: bool


def _validate_mp_options(
    storage_dtype,
    compute_dtype,
    spike_dtype: torch.dtype,
    save_intermediates: bool,
    *,
    compute_label: str = "compute_dtype",
) -> tuple[torch.dtype, str]:
    storage_dtype = normalize_triton_storage_dtype(storage_dtype)
    compute_dtype_name = normalize_triton_compute_dtype_name(compute_dtype)
    _require_fp8_storage_dtype(compute_dtype_name, storage_dtype, compute_label)
    if spike_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "spike_dtype must be torch.float32, torch.float16, or torch.bfloat16, "
            f"but got {spike_dtype}."
        )
    if not isinstance(save_intermediates, bool):
        raise ValueError("save_intermediates must be bool.")
    return storage_dtype, compute_dtype_name


def _require_fp8_storage_dtype(
    compute_dtype_name: str,
    storage_dtype: torch.dtype,
    label: str,
) -> None:
    if compute_dtype_name == "fp8" and not is_fp8_dtype(storage_dtype):
        raise ValueError(f"{label}='fp8' requires an FP8 storage_dtype.")


def _check_fp8_capability(
    storage_dtype: torch.dtype,
    device: torch.device,
    compute_dtype_name: str,
    neuron_name: str,
    pass_name: str,
) -> None:
    if not is_fp8_dtype(storage_dtype):
        return
    from .fp8_capability import (
        triton_fp8_neuron_backward_capability,
        triton_fp8_neuron_capability,
    )

    if pass_name == "forward":
        dtype_report = triton_fp8_neuron_capability(
            storage_dtype, device, compute_dtype=compute_dtype_name
        )
        kw = "compute_dtype"
    elif pass_name == "backward":
        dtype_report = triton_fp8_neuron_backward_capability(
            storage_dtype, device, compute_dtype=compute_dtype_name
        )
        kw = "backward_compute_dtype"
    else:
        raise ValueError(f"Unsupported Triton FP8 pass name: {pass_name!r}.")
    if not dtype_report.get("available", False):
        reason = dtype_report.get("reason", "unknown reason")
        raise RuntimeError(
            f"Triton FP8 {neuron_name} {pass_name} is unavailable for "
            f"{storage_dtype} with {kw}={compute_dtype_name!r}: {reason}"
        )


def _prepare_triton_neuron_execution_plan(
    *,
    neuron_type: str,
    device,
    storage_dtype,
    forward_compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
) -> _TritonNeuronExecutionPlan:
    device = normalize_cuda_device(device)
    key = (
        neuron_type,
        device,
        storage_dtype,
        forward_compute_dtype,
        backward_compute_dtype,
        spike_dtype,
        save_intermediates,
    )
    # Dynamo unwraps lru_cache and would repeat device checks and logging.
    if key not in _TRITON_NEURON_EXECUTION_PLANS:
        _TRITON_NEURON_EXECUTION_PLANS[key] = _make_triton_neuron_execution_plan(
            neuron_type=neuron_type,
            device=device,
            storage_dtype=storage_dtype,
            forward_compute_dtype=forward_compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=spike_dtype,
            save_intermediates=save_intermediates,
        )
    return _TRITON_NEURON_EXECUTION_PLANS[key]


def _make_triton_neuron_execution_plan(
    *,
    neuron_type: str,
    device: torch.device,
    storage_dtype,
    forward_compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
) -> _TritonNeuronExecutionPlan:
    if neuron_type not in _SUPPORTED_PLAN_NEURON_TYPES:
        raise ValueError(
            "neuron_type must be one of 'if', 'lif', or 'plif', "
            f"but got {neuron_type!r}."
        )
    storage_dtype, forward_compute_dtype_name = _validate_mp_options(
        storage_dtype, forward_compute_dtype, spike_dtype, save_intermediates
    )
    try:
        backward_compute_dtype_name = normalize_triton_compute_dtype_name(
            backward_compute_dtype
        )
    except ValueError as e:
        raise ValueError(f"Invalid backward_compute_dtype: {e}") from e
    _require_fp8_storage_dtype(
        backward_compute_dtype_name, storage_dtype, "backward_compute_dtype"
    )
    if device.type != "cuda":
        raise RuntimeError(
            "Triton neuron execution plan is unavailable: requires a CUDA device."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Triton neuron execution plan is unavailable: CUDA is absent."
        )

    forward_compute_tl_dtype = resolve_triton_compute_dtype(
        forward_compute_dtype_name, storage_dtype
    )
    backward_compute_tl_dtype = resolve_triton_compute_dtype(
        backward_compute_dtype_name, storage_dtype
    )
    _check_fp8_capability(
        storage_dtype,
        device,
        forward_compute_dtype_name,
        neuron_type.upper(),
        "forward",
    )
    _check_fp8_capability(
        storage_dtype,
        device,
        backward_compute_dtype_name,
        neuron_type.upper(),
        "backward",
    )
    plan = _TritonNeuronExecutionPlan(
        neuron_type=neuron_type,
        device=device,
        storage_dtype=storage_dtype,
        forward_compute_dtype_name=forward_compute_dtype_name,
        forward_compute_tl_dtype=forward_compute_tl_dtype,
        backward_compute_dtype_name=backward_compute_dtype_name,
        backward_compute_tl_dtype=backward_compute_tl_dtype,
        spike_dtype=spike_dtype,
        storage_dtype_id=torch_dtype_to_triton_neuron_dtype_id(storage_dtype),
        forward_compute_dtype_id=triton_compute_dtype_name_to_neuron_dtype_id(
            forward_compute_dtype_name, storage_dtype
        ),
        backward_compute_dtype_id=triton_compute_dtype_name_to_neuron_dtype_id(
            backward_compute_dtype_name, storage_dtype
        ),
        spike_dtype_id=torch_dtype_to_triton_neuron_dtype_id(spike_dtype),
        save_intermediates=save_intermediates,
    )
    logger.info(
        "ops selection operator=sj_{} device=cuda:{} ({}) implementation=triton "
        "profile=storage:{} forward:{} backward:{} spikes:{}",
        neuron_type,
        device.index,
        torch.cuda.get_device_name(device),
        storage_dtype,
        forward_compute_dtype_name,
        backward_compute_dtype_name,
        spike_dtype,
    )
    return plan


def _check_mp_cuda_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    neuron_name: str,
) -> None:
    if x_seq.device.type != "cuda" or v_init.device.type != "cuda":
        raise RuntimeError(
            f"Mixed-precision Triton {neuron_name} forward requires CUDA tensors."
        )
    if normalize_cuda_device(x_seq.device) != normalize_cuda_device(v_init.device):
        raise RuntimeError("x_seq and v_init must be on the same CUDA device.")
    expected_shape = x_seq.shape[1:]
    if v_init.shape != expected_shape:
        raise RuntimeError(
            f"v_init shape {v_init.shape} must match x_seq[0] shape "
            f"{torch.Size(expected_shape)}."
        )


def _check_plan_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    plan: _TritonNeuronExecutionPlan,
    neuron_name: str,
) -> None:
    _check_mp_cuda_inputs(x_seq, v_init, neuron_name)
    if normalize_cuda_device(x_seq.device) != plan.device:
        raise RuntimeError(
            f"Mixed-precision Triton {neuron_name} forward input device "
            f"{x_seq.device} does not match plan device {plan.device}."
        )
