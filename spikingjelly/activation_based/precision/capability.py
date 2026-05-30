from __future__ import annotations

import importlib.util
from typing import Any

import torch


def build_capability_report(model, device, mode: str) -> dict[str, Any]:
    device = torch.device(device)
    device_str = str(device)
    if device_str.startswith("cuda"):
        device_type = "cuda"
    elif device_str.startswith("mps"):
        device_type = "mps"
    else:
        device_type = "cpu"
    is_cuda = device_type == "cuda"
    capability = None
    if is_cuda and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(device)
    torchao_installed = importlib.util.find_spec("torchao") is not None
    cpu_bf16_autocast = hasattr(getattr(torch, "amp", None), "autocast")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_bf16_supported = bool(
        mps_backend is not None
        and mps_available
        and getattr(mps_backend, "is_bf16_supported", lambda: False)()
    )
    can_convert = True
    can_execute = True
    runtime_validation_required = False
    execution_note = None

    if mode == "fp8-torchao":
        can_convert = torchao_installed
        can_execute = (
            is_cuda
            and torch.cuda.is_available()
            and capability is not None
            and capability >= (8, 9)
            and torchao_installed
        )
        runtime_validation_required = can_execute
        if not torchao_installed:
            execution_note = "torchao is not installed"
        elif not is_cuda:
            execution_note = "fp8-torchao requires a CUDA device"
        elif capability is None or capability < (8, 9):
            execution_note = "torch._scaled_mm requires compute capability >= 8.9"
        else:
            execution_note = (
                "runtime execution still requires validation because cuBLASLt / torch._scaled_mm "
                "support may fail on some environments"
            )

    return {
        "requested_mode": mode,
        "device": device_str,
        "device_type": device_type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_capability": capability,
        "torchao_installed": torchao_installed,
        "bf16_supported": (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if is_cuda
            else mps_bf16_supported
        ),
        "cpu_bf16_autocast": cpu_bf16_autocast if device_type == "cpu" else False,
        "mps_available": mps_available,
        "model_class": type(model).__name__,
        "can_convert": can_convert,
        "can_execute": can_execute,
        "runtime_validation_required": runtime_validation_required,
        "execution_note": execution_note,
    }


def validate_capability(report: dict[str, Any]) -> None:
    mode = report["requested_mode"]
    device_type = report["device_type"]

    if mode == "fp32":
        return

    if mode == "fp16":
        if device_type == "cpu":
            raise RuntimeError("precision='fp16' is not supported on cpu in the current stage.")
        if device_type == "mps":
            return
        if not report["cuda_available"]:
            raise RuntimeError("precision='fp16' requires CUDA, but CUDA is not available.")
        return

    if mode == "bf16":
        if device_type == "cpu":
            return
        if device_type == "mps":
            if not report["bf16_supported"]:
                raise RuntimeError(
                    "precision='bf16' was requested on MPS, but this MPS device does not support bf16."
                )
            return
        if not report["cuda_available"]:
            raise RuntimeError("precision='bf16' requires CUDA or CPU bf16 autocast support.")
        if not report["bf16_supported"]:
            raise RuntimeError(
                "precision='bf16' was requested on CUDA, but this CUDA device does not report bf16 support."
            )
        return

    if mode == "fp8-torchao":
        if not report.get("torchao_installed", False):
            raise RuntimeError(
                "precision='fp8-torchao' requires torchao, but torchao is not installed."
            )
        if device_type != "cuda":
            raise RuntimeError(
                "precision='fp8-torchao' is only supported on CUDA in the current stage."
            )
        if not report["cuda_available"]:
            raise RuntimeError(
                "precision='fp8-torchao' requires CUDA, but CUDA is not available."
            )
        capability = report.get("cuda_device_capability")
        if capability is not None:
            capability = tuple(capability)
        if capability is None or capability < (8, 9):
            raise RuntimeError(
                "precision='fp8-torchao' requires compute capability >= 8.9; "
                f"got {capability}."
            )
        return

    raise RuntimeError(
        f"Unsupported precision mode {mode!r}. Current stage supports: fp32, fp16, bf16, fp8-torchao."
    )
