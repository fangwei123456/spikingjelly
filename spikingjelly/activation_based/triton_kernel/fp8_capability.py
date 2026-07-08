from __future__ import annotations

import functools
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except BaseException:
    triton = None
    tl = None


__all__ = [
    "supports_triton_fp8_e4m3fn",
    "supports_triton_fp8_e5m2",
    "supports_triton_fp8_neuron_forward",
    "triton_fp8_neuron_capability_report",
]


if triton is not None:

    @triton.jit
    def _fp8_neuron_forward_probe_kernel(
        x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x + 0.0
        tl.store(y_ptr + offsets, y, mask=mask)

else:
    _fp8_neuron_forward_probe_kernel = None


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _fp8_dtype_candidates() -> dict[str, torch.dtype | None]:
    return {
        "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    }


def _normalize_fp8_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        normalized = dtype
    elif isinstance(dtype, str):
        key = dtype.lower().replace("torch.", "")
        mapping = {}
        e4m3fn = getattr(torch, "float8_e4m3fn", None)
        e5m2 = getattr(torch, "float8_e5m2", None)
        if e4m3fn is not None:
            mapping.update(
                {
                    "float8_e4m3fn": e4m3fn,
                    "fp8_e4m3fn": e4m3fn,
                    "e4m3fn": e4m3fn,
                }
            )
        if e5m2 is not None:
            mapping.update(
                {
                    "float8_e5m2": e5m2,
                    "fp8_e5m2": e5m2,
                    "e5m2": e5m2,
                }
            )
        try:
            normalized = mapping[key]
        except KeyError as e:
            raise ValueError(f"Unsupported Triton FP8 dtype: {dtype!r}.") from e
    else:
        raise ValueError(
            "dtype must be a string or torch.dtype, "
            f"but got {type(dtype).__name__}."
        )

    if normalized not in {d for d in _fp8_dtype_candidates().values() if d is not None}:
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


@functools.lru_cache(maxsize=None)
def _probe_cached(
    dtype_name: str,
    device_str: str,
    capability: str,
    torch_version: str,
    triton_version: str,
):
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
        with torch.cuda.device(device):
            x = torch.tensor(
                [0.0, 0.25, -0.5, 1.0, 2.0, -2.0, 0.75, -0.75],
                device=device,
                dtype=torch.float32,
            ).to(dtype=dtype)
            y = torch.empty_like(x)
            _fp8_neuron_forward_probe_kernel[(1,)](x, y, N=x.numel(), BLOCK=8)
            torch.cuda.synchronize(device)
            yf = y.to(dtype=torch.float32)
            if not torch.isfinite(yf).all():
                return _failure(False, "probe produced non-finite values")
    except Exception as e:
        first_line = str(e).splitlines()[0] if str(e) else repr(e)
        return _failure(False, f"{type(e).__name__}: {first_line}")
    return _failure(True)


def _probe(dtype, device) -> dict[str, Any]:
    dtype = _normalize_fp8_dtype(dtype)
    device = _normalize_cuda_device(device)
    triton_version = (
        getattr(triton, "__version__", None) if triton is not None else None
    )
    if device.type == "cuda" and torch.cuda.is_available():
        capability = str(torch.cuda.get_device_capability(device))
    else:
        capability = "unavailable"
    return _probe_cached(
        _torch_dtype_name(dtype),
        str(device),
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


def supports_triton_fp8_neuron_forward(dtype, device) -> bool:
    return bool(_probe(dtype, device)["available"])


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
