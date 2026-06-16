from __future__ import annotations

from typing import Any

import torch


def _torchao_available() -> bool:
    try:
        import torchao  # noqa: F401

        return True
    except ImportError:
        return False


def _detect_cpu_bf16() -> bool:
    """Check whether the CPU backend actually supports bfloat16 tensors."""
    try:
        _ = torch.tensor(1.0, dtype=torch.bfloat16)
        return True
    except Exception:
        return False


def _detect_mps_bf16(mps_backend) -> bool:
    """Check whether the MPS backend reports bfloat16 support."""
    fn = getattr(mps_backend, "is_bf16_supported", None)
    if fn is not None:
        return bool(fn())
    try:
        _ = torch.tensor(1.0, dtype=torch.bfloat16, device="mps")
        return True
    except Exception:
        return False


def _resolve_device_type(device: torch.device | str) -> str:
    s = str(device)
    if s.startswith("cuda"):
        return "cuda"
    if s.startswith("mps"):
        return "mps"
    return "cpu"


def _assess_fp8_torchao(
    torchao_installed: bool,
    is_cuda: bool,
    capability: tuple | None,
) -> tuple[bool, bool, str | None]:
    """Determine fp8-torchao viability from environment probes.

    Returns ``(can_convert, can_execute, execution_note)``.
    """
    can_convert = torchao_installed
    can_execute = (
        is_cuda
        and torch.cuda.is_available()
        and capability is not None
        and capability >= (8, 9)
        and torchao_installed
    )

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
    return can_convert, can_execute, execution_note


def build_capability_report(model, device, mode: str) -> dict[str, Any]:
    device = torch.device(device)
    device_str = str(device)
    device_type = _resolve_device_type(device)
    is_cuda = device_type == "cuda"
    capability = None
    if is_cuda and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(device)
    torchao_installed = _torchao_available()

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_bf16_supported = (
        _detect_mps_bf16(mps_backend)
        if mps_backend is not None and mps_available
        else False
    )
    cpu_bf16 = _detect_cpu_bf16()
    if is_cuda:
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    elif device_type == "mps":
        bf16_supported = mps_bf16_supported
    else:
        bf16_supported = cpu_bf16

    can_convert = True
    can_execute = True
    execution_note = None

    if mode == "fp8-torchao":
        can_convert, can_execute, execution_note = _assess_fp8_torchao(
            torchao_installed, is_cuda, capability
        )

    return {
        "requested_mode": mode,
        "device": device_str,
        "device_type": device_type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_device_capability": capability,
        "torchao_installed": torchao_installed,
        "bf16_supported": bf16_supported,
        "cpu_bf16_autocast": cpu_bf16 if device_type == "cpu" else False,
        "mps_available": mps_available,
        "model_class": type(model).__name__,
        "can_convert": can_convert,
        "can_execute": can_execute,
        "runtime_validation_required": can_execute,
        "execution_note": execution_note,
    }


def _validate_fp16(report: dict[str, Any]) -> None:
    device_type = report["device_type"]
    if device_type == "cpu":
        raise RuntimeError(
            "precision='fp16' is not supported on cpu in the current stage."
        )
    if device_type == "mps":
        return
    if not report["cuda_available"]:
        raise RuntimeError("precision='fp16' requires CUDA, but CUDA is not available.")


def _validate_bf16(report: dict[str, Any]) -> None:
    device_type = report["device_type"]
    if device_type == "cpu":
        return
    if device_type == "mps":
        if not report["bf16_supported"]:
            raise RuntimeError(
                "precision='bf16' was requested on MPS, but this MPS "
                "device does not support bf16."
            )
        return
    if not report["cuda_available"]:
        raise RuntimeError(
            "precision='bf16' requires CUDA or CPU bf16 autocast support."
        )
    if not report["bf16_supported"]:
        raise RuntimeError(
            "precision='bf16' was requested on CUDA, but this CUDA device "
            "does not report bf16 support."
        )


def _validate_fp8(report: dict[str, Any]) -> None:
    if not report.get("torchao_installed", False):
        raise RuntimeError(
            "precision='fp8-torchao' requires torchao, but torchao is not installed."
        )
    if report["device_type"] != "cuda":
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


_VALIDATORS = {
    "fp16": _validate_fp16,
    "bf16": _validate_bf16,
    "fp8-torchao": _validate_fp8,
}


def validate_capability(report: dict[str, Any]) -> None:
    mode = report["requested_mode"]
    if mode == "fp32":
        return
    validator = _VALIDATORS.get(mode)
    if validator is None:
        raise RuntimeError(
            f"Unsupported precision mode {mode!r}. "
            "Current stage supports: fp32, fp16, bf16, fp8-torchao."
        )
    validator(report)
