from __future__ import annotations

import functools
import logging
import re
from typing import Any

import torch

from .triton_utils import (
    is_fp8_dtype,
    normalize_triton_compute_dtype_name,
    normalize_triton_storage_dtype,
    resolve_triton_compute_dtype,
)

try:
    import triton
    import triton.language as tl
except Exception as e:
    logging.getLogger(__name__).info(
        "Failed to import Triton for FP8 capability probe: %s", e
    )
    triton = None
    tl = None


__all__ = [
    "supports_triton_fp8_e4m3fn",
    "supports_triton_fp8_e5m2",
    "supports_triton_fp8_neuron_forward",
    "triton_fp8_neuron_capability",
    "triton_fp8_neuron_backward_capability",
    "triton_fp8_neuron_capability_report",
]


if triton is not None:

    @triton.jit
    def _fp8_neuron_forward_probe_kernel(
        x_ptr,
        y_ptr,
        N: tl.constexpr,
        compute_dtype: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
        v = tl.full([BLOCK], 0.125, dtype=compute_dtype)
        r_tau = tl.full([1], 0.5, dtype=compute_dtype)
        h = v + r_tau * (0.0 - v + x)
        s = tl.where(h >= 1.0, 1.0, 0.0).to(compute_dtype)
        y = h - s * 1.0
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def _fp8_neuron_backward_probe_kernel(
        grad_s_ptr,
        grad_v_ptr,
        h_ptr,
        v_ptr,
        grad_x_ptr,
        grad_v_init_ptr,
        grad_r_tau_ptr,
        N: tl.constexpr,
        compute_dtype: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        grad_v_acc = tl.zeros([BLOCK], dtype=compute_dtype)
        grad_r_tau_acc = tl.zeros([BLOCK], dtype=compute_dtype)
        r_tau = tl.full([1], 0.5, dtype=compute_dtype)
        alpha = tl.full([1], 4.0, dtype=compute_dtype)

        for t in tl.static_range(1, -1, -1):
            base = t * N + offsets
            grad_s = tl.load(grad_s_ptr + base, mask=mask, other=0.0).to(
                compute_dtype
            )
            grad_v = tl.load(grad_v_ptr + base, mask=mask, other=0.0).to(
                compute_dtype
            )
            h = tl.load(h_ptr + base, mask=mask, other=0.0).to(compute_dtype)
            v_last = tl.load(v_ptr + base, mask=mask, other=0.0).to(compute_dtype)
            s = tl.where(h >= 1.0, 1.0, 0.0).to(compute_dtype)
            over_th = alpha * (h - 1.0)
            sg = alpha * tl.sigmoid(over_th) * (1.0 - tl.sigmoid(over_th))
            grad_v_combined = grad_v + grad_v_acc
            grad_h = tl.fma(grad_s - grad_v_combined * (h - 0.0), sg, grad_v_combined)
            grad_x = grad_h * r_tau
            grad_v_acc = grad_h * (1.0 - r_tau) * (1.0 - s)
            grad_r_tau_acc += grad_h * (h - v_last) / r_tau
            tl.store(grad_x_ptr + base, grad_x, mask=mask)

        tl.store(grad_v_init_ptr + offsets, grad_v_acc, mask=mask)
        tl.store(grad_r_tau_ptr + offsets, grad_r_tau_acc, mask=mask)

else:
    _fp8_neuron_forward_probe_kernel = None
    _fp8_neuron_backward_probe_kernel = None


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _fp8_dtype_candidates() -> dict[str, torch.dtype | None]:
    return {
        "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    }


def _normalize_fp8_dtype(dtype) -> torch.dtype:
    try:
        normalized = normalize_triton_storage_dtype(dtype)
    except ValueError as e:
        raise ValueError(f"Unsupported Triton FP8 dtype: {dtype!r}.") from e
    if not is_fp8_dtype(normalized):
        raise ValueError(f"Unsupported Triton FP8 dtype: {normalized}.")
    return normalized


def _normalize_cuda_device(device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        return device
    if device.index is None:
        if not torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _failure(available: bool, reason: str | None = None) -> dict[str, Any]:
    return {"available": available, "reason": reason}


def _probe_exception_reason(e: Exception) -> str:
    msg = str(e) if str(e) else repr(e)
    value_error = re.search(r'ValueError\("([^"]+)"\)', msg)
    if value_error is not None:
        msg = value_error.group(1)
    else:
        lines = [line.strip() for line in msg.splitlines() if line.strip()]
        for line in reversed(lines):
            if line != "^" and not line.startswith("def "):
                msg = line
                break
    return f"{type(e).__name__}: {msg}"


@functools.lru_cache(maxsize=None)
def _probe_cached(
    dtype_name: str,
    device_str: str,
    compute_dtype_name: str,
    capability: str,
    torch_version: str,
    triton_version: str,
):
    # These parameters are lru_cache keys that invalidate stale probe results when
    # the GPU capability or library versions change.
    del capability, torch_version, triton_version
    try:
        dtype = _normalize_fp8_dtype(dtype_name)
    except ValueError as e:
        return _failure(False, str(e))

    if triton is None or tl is None:
        return _failure(False, "triton is not installed")
    if not torch.cuda.is_available():
        return _failure(False, "CUDA is not available")
    device = _normalize_cuda_device(device_str)
    if device.type != "cuda":
        return _failure(False, "Triton FP8 neuron forward requires a CUDA device")
    if _fp8_neuron_forward_probe_kernel is None:
        return _failure(False, "Triton FP8 probe kernel is unavailable")
    try:
        compute_tl_dtype = resolve_triton_compute_dtype(compute_dtype_name, dtype)
    except ValueError as e:
        return _failure(False, str(e))

    try:
        with torch.cuda.device(device):
            x = torch.tensor(
                [0.0, 0.25, -0.5, 1.0, 2.0, -2.0, 0.75, -0.75],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            y = torch.empty_like(x)
            _fp8_neuron_forward_probe_kernel[(1,)](
                x,
                y,
                N=x.numel(),
                compute_dtype=compute_tl_dtype,
                BLOCK=8,
            )
            torch.cuda.synchronize(device)
            yf = y.to(dtype=torch.float32)
            if not torch.isfinite(yf).all():
                return _failure(False, "probe produced non-finite values")
    except Exception as e:
        return _failure(False, _probe_exception_reason(e))
    return _failure(True)


@functools.lru_cache(maxsize=None)
def _backward_probe_cached(
    dtype_name: str,
    device_str: str,
    compute_dtype_name: str,
    capability: str,
    torch_version: str,
    triton_version: str,
):
    # These parameters are lru_cache keys that invalidate stale probe results when
    # the GPU capability or library versions change.
    del capability, torch_version, triton_version
    try:
        dtype = _normalize_fp8_dtype(dtype_name)
    except ValueError as e:
        return _failure(False, str(e))

    if triton is None or tl is None:
        return _failure(False, "triton is not installed")
    if not torch.cuda.is_available():
        return _failure(False, "CUDA is not available")
    device = _normalize_cuda_device(device_str)
    if device.type != "cuda":
        return _failure(False, "Triton FP8 neuron backward requires a CUDA device")
    if _fp8_neuron_backward_probe_kernel is None:
        return _failure(False, "Triton FP8 backward probe kernel is unavailable")
    try:
        compute_tl_dtype = resolve_triton_compute_dtype(compute_dtype_name, dtype)
    except ValueError as e:
        return _failure(False, str(e))

    try:
        with torch.cuda.device(device):
            grad_s = torch.tensor(
                [0.0, 0.25, -0.5, 1.0, 0.5, -0.25, 0.75, -0.75],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            grad_v = torch.tensor(
                [0.5, -0.25, 0.0, 0.25, -0.5, 0.125, 0.75, -0.75],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            h = torch.tensor(
                [0.1, 0.9, 1.1, -0.2, 0.25, 1.5, -0.5, 0.75],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            v_last = torch.tensor(
                [0.0, 0.25, 0.5, -0.25, 0.125, 0.75, -0.5, 0.5],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            grad_x = torch.empty_like(grad_s)
            grad_v_init = torch.empty((4,), device=device, dtype=dtype)
            grad_r_tau = torch.empty((4,), device=device, dtype=torch.float32)
            _fp8_neuron_backward_probe_kernel[(1,)](
                grad_s,
                grad_v,
                h,
                v_last,
                grad_x,
                grad_v_init,
                grad_r_tau,
                N=4,
                compute_dtype=compute_tl_dtype,
                BLOCK=4,
            )
            torch.cuda.synchronize(device)
            outputs = (
                grad_x.to(dtype=torch.float32),
                grad_v_init.to(dtype=torch.float32),
                grad_r_tau,
                grad_r_tau.sum(),
            )
            if not all(torch.isfinite(out).all() for out in outputs):
                return _failure(False, "backward probe produced non-finite values")
    except Exception as e:
        return _failure(False, _probe_exception_reason(e))
    return _failure(True)


def _cache_key_parts(dtype, device, compute_dtype) -> tuple[torch.dtype, torch.device, str, str, str | None]:
    dtype = _normalize_fp8_dtype(dtype)
    device = _normalize_cuda_device(device)
    compute_dtype_name = normalize_triton_compute_dtype_name(compute_dtype)
    triton_version = (
        getattr(triton, "__version__", None) if triton is not None else None
    )
    if device.type == "cuda" and torch.cuda.is_available():
        capability = str(torch.cuda.get_device_capability(device))
    else:
        capability = "unavailable"
    return dtype, device, compute_dtype_name, capability, str(triton_version)


def _probe(dtype, device, compute_dtype="fp32") -> dict[str, Any]:
    dtype, device, compute_dtype_name, capability, triton_version = _cache_key_parts(
        dtype, device, compute_dtype
    )
    return _probe_cached(
        _torch_dtype_name(dtype),
        str(device),
        compute_dtype_name,
        capability,
        torch.__version__,
        str(triton_version),
    )


def _backward_probe(dtype, device, compute_dtype="fp32") -> dict[str, Any]:
    dtype, device, compute_dtype_name, capability, triton_version = _cache_key_parts(
        dtype, device, compute_dtype
    )
    return _backward_probe_cached(
        _torch_dtype_name(dtype),
        str(device),
        compute_dtype_name,
        capability,
        torch.__version__,
        str(triton_version),
    )


def supports_triton_fp8_e4m3fn(device) -> bool:
    dtype = getattr(torch, "float8_e4m3fn", None)
    return False if dtype is None else bool(_probe(dtype, device)["available"])


def supports_triton_fp8_e5m2(device) -> bool:
    dtype = getattr(torch, "float8_e5m2", None)
    return False if dtype is None else bool(_probe(dtype, device)["available"])


def supports_triton_fp8_neuron_forward(dtype, device, compute_dtype="fp32") -> bool:
    return bool(_probe(dtype, device, compute_dtype=compute_dtype)["available"])


def triton_fp8_neuron_capability(dtype, device, compute_dtype="fp32") -> dict[str, Any]:
    return _probe(dtype, device, compute_dtype=compute_dtype)


def triton_fp8_neuron_backward_capability(
    dtype, device, compute_dtype="fp32"
) -> dict[str, Any]:
    return _backward_probe(dtype, device, compute_dtype=compute_dtype)


def triton_fp8_neuron_capability_report(device) -> dict[str, Any]:
    device = _normalize_cuda_device(device)
    report = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "triton_available": triton is not None,
        "torch_version": torch.__version__,
        "triton_version": getattr(triton, "__version__", None)
        if triton is not None
        else None,
        "cuda_device_capability": None,
        "dtypes": {},
    }
    if device.type == "cuda" and torch.cuda.is_available():
        report["cuda_device_capability"] = torch.cuda.get_device_capability(device)

    for dtype_name, dtype in _fp8_dtype_candidates().items():
        if dtype is None:
            report["dtypes"][dtype_name] = _failure(
                False, f"{dtype_name} is unavailable in this PyTorch build"
            )
        else:
            report["dtypes"][dtype_name] = _probe(dtype, device)
    return report
