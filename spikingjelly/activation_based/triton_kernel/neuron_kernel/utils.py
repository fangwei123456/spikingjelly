from dataclasses import dataclass
from typing import Any

import torch

from ..triton_utils import (
    normalize_cuda_device,
    is_fp8_dtype,
    normalize_triton_compute_dtype_name,
    normalize_triton_storage_dtype,
    resolve_triton_compute_dtype,
    torch_dtype_to_triton_neuron_dtype_id,
    triton_compute_dtype_name_to_neuron_dtype_id,
)


_SUPPORTED_PLAN_NEURON_TYPES = frozenset({"if", "lif", "plif"})


@dataclass(frozen=True)
class TritonNeuronExecutionPlan:
    neuron_type: str
    device: torch.device
    storage_dtype: torch.dtype
    forward_compute_dtype_name: str
    forward_compute_tl_dtype: Any
    backward_compute_dtype_name: str
    backward_compute_tl_dtype: Any
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
    from ..fp8_capability import (
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


def prepare_triton_neuron_execution_plan(
    *,
    neuron_type: str,
    device,
    storage_dtype,
    forward_compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
) -> TritonNeuronExecutionPlan:
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
    device = normalize_cuda_device(device)
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
    return TritonNeuronExecutionPlan(
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
    plan: TritonNeuronExecutionPlan,
    neuron_name: str,
) -> None:
    _check_mp_cuda_inputs(x_seq, v_init, neuron_name)
    if normalize_cuda_device(x_seq.device) != plan.device:
        raise RuntimeError(
            f"Mixed-precision Triton {neuron_name} forward input device "
            f"{x_seq.device} does not match plan device {plan.device}."
        )
