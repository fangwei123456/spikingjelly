from dataclasses import dataclass
from typing import Any

import torch

from ..triton_utils import (
    is_fp8_dtype,
    normalize_triton_compute_dtype_name,
    normalize_triton_storage_dtype,
    resolve_triton_compute_dtype,
)


_SUPPORTED_PLAN_NEURON_TYPES = {"if", "lif", "plif"}


def _normalize_plan_device(device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        return device
    if device.index is None:
        if not torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda", torch.cuda.current_device())
    return device


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
    save_intermediates: bool

    @property
    def compute_dtype_name(self) -> str:
        return self.forward_compute_dtype_name

    @property
    def compute_tl_dtype(self):
        return self.forward_compute_tl_dtype

    def matches(
        self,
        *,
        neuron_type: str,
        device,
        storage_dtype,
        compute_dtype,
        backward_compute_dtype="fp32",
        spike_dtype: torch.dtype,
        save_intermediates: bool,
    ) -> bool:
        try:
            device = _normalize_plan_device(device)
            storage_dtype = normalize_triton_storage_dtype(storage_dtype)
            compute_dtype_name = normalize_triton_compute_dtype_name(compute_dtype)
            backward_compute_dtype_name = normalize_triton_compute_dtype_name(
                backward_compute_dtype
            )
        except ValueError:
            return False
        return (
            self.neuron_type == neuron_type
            and self.device == device
            and self.storage_dtype == storage_dtype
            and self.forward_compute_dtype_name == compute_dtype_name
            and self.backward_compute_dtype_name == backward_compute_dtype_name
            and self.spike_dtype == spike_dtype
            and self.save_intermediates == save_intermediates
        )


TritonNeuronForwardPlan = TritonNeuronExecutionPlan


def _validate_mixed_precision_options(
    storage_dtype,
    compute_dtype,
    spike_dtype: torch.dtype,
    save_intermediates: bool,
) -> tuple[torch.dtype, str]:
    storage_dtype = normalize_triton_storage_dtype(storage_dtype)
    compute_dtype_name = normalize_triton_compute_dtype_name(compute_dtype)
    if compute_dtype_name == "fp8" and not is_fp8_dtype(storage_dtype):
        raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
    if spike_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "spike_dtype must be torch.float32, torch.float16, or torch.bfloat16, "
            f"but got {spike_dtype}."
        )
    if not isinstance(save_intermediates, bool):
        raise ValueError("save_intermediates must be bool.")
    return storage_dtype, compute_dtype_name


def _check_fp8_forward_capability(
    storage_dtype: torch.dtype,
    device: torch.device,
    compute_dtype_name: str,
    neuron_name: str,
) -> None:
    if not is_fp8_dtype(storage_dtype):
        return
    from ..fp8_capability import triton_fp8_neuron_capability

    dtype_report = triton_fp8_neuron_capability(
        storage_dtype, device, compute_dtype=compute_dtype_name
    )
    if not dtype_report.get("available", False):
        reason = dtype_report.get("reason", "unknown reason")
        raise RuntimeError(
            f"Triton FP8 {neuron_name} forward is unavailable for "
            f"{storage_dtype} with compute_dtype={compute_dtype_name!r}: {reason}"
        )


def _check_fp8_backward_capability(
    storage_dtype: torch.dtype,
    device: torch.device,
    compute_dtype_name: str,
    neuron_name: str,
) -> None:
    if not is_fp8_dtype(storage_dtype):
        return
    from ..fp8_capability import triton_fp8_neuron_capability

    dtype_report = triton_fp8_neuron_capability(
        storage_dtype, device, compute_dtype=compute_dtype_name
    )
    if not dtype_report.get("available", False):
        reason = dtype_report.get("reason", "unknown reason")
        raise RuntimeError(
            f"Triton FP8 {neuron_name} backward is unavailable for "
            f"{storage_dtype} with backward_compute_dtype={compute_dtype_name!r}: "
            f"{reason}"
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
    storage_dtype, forward_compute_dtype_name = _validate_mixed_precision_options(
        storage_dtype, forward_compute_dtype, spike_dtype, save_intermediates
    )
    try:
        backward_compute_dtype_name = normalize_triton_compute_dtype_name(
            backward_compute_dtype
        )
    except ValueError as e:
        raise ValueError(f"Invalid backward_compute_dtype: {e}") from e
    if backward_compute_dtype_name == "fp8" and not is_fp8_dtype(storage_dtype):
        raise ValueError("backward_compute_dtype='fp8' requires an FP8 storage_dtype.")
    device = _normalize_plan_device(device)
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
    _check_fp8_forward_capability(
        storage_dtype,
        device,
        forward_compute_dtype_name,
        neuron_type.upper(),
    )
    _check_fp8_backward_capability(
        storage_dtype,
        device,
        backward_compute_dtype_name,
        neuron_type.upper(),
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
        save_intermediates=save_intermediates,
    )


def prepare_triton_neuron_forward_plan(
    *,
    neuron_type: str,
    device,
    storage_dtype,
    compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype: torch.dtype = torch.float32,
    save_intermediates: bool = True,
) -> TritonNeuronForwardPlan:
    return prepare_triton_neuron_execution_plan(
        neuron_type=neuron_type,
        device=device,
        storage_dtype=storage_dtype,
        forward_compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=spike_dtype,
        save_intermediates=save_intermediates,
    )


def _check_mixed_precision_cuda_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    neuron_name: str,
) -> None:
    if x_seq.device.type != "cuda" or v_init.device.type != "cuda":
        raise RuntimeError(
            f"Mixed-precision Triton {neuron_name} forward requires CUDA tensors."
        )
    if x_seq.device != v_init.device:
        raise RuntimeError("x_seq and v_init must be on the same CUDA device.")


def _check_plan_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    plan: TritonNeuronExecutionPlan,
    neuron_name: str,
) -> None:
    _check_mixed_precision_cuda_inputs(x_seq, v_init, neuron_name)
    if _normalize_plan_device(x_seq.device) != plan.device:
        raise RuntimeError(
            f"Mixed-precision Triton {neuron_name} forward input device "
            f"{x_seq.device} does not match plan device {plan.device}."
        )
