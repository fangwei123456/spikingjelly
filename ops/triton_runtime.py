"""Borrowed from:
https://github.com/AllenYolk/flash-snn/tree/main/flashsnn/utils
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""

import functools
import os

import torch

from spikingjelly import configure
from spikingjelly.logger import logger

from . import triton_missing as dummy

triton_op = getattr(torch.library, "triton_op", None)

_TRITON_IMPORT_ERROR = None
try:
    import triton
    import triton.language as tl

    type_dict = {
        torch.bool: tl.int1,
        torch.float32: tl.float32,
        torch.float16: tl.float16,
    }
    type_str_dict = {
        torch.bool: "tl.int1",
        torch.float32: "tl.float32",
        torch.float16: "tl.float16",
    }

    # check bfloat16 support
    if torch.cuda.is_available():
        dc = torch.cuda.get_device_capability()
        if dc[0] >= 8 and hasattr(tl, "bfloat16") and hasattr(torch, "bfloat16"):
            type_dict[torch.bfloat16] = tl.bfloat16
            type_str_dict[torch.bfloat16] = "tl.bfloat16"
        else:
            logger.debug("bfloat16 is not supported on this device.")
    if hasattr(torch, "float8_e4m3fn"):
        if hasattr(tl, "float8e4m3fn"):
            type_dict[torch.float8_e4m3fn] = tl.float8e4m3fn
            type_str_dict[torch.float8_e4m3fn] = "tl.float8e4m3fn"
        elif hasattr(tl, "float8e4nv"):
            type_dict[torch.float8_e4m3fn] = tl.float8e4nv
            type_str_dict[torch.float8_e4m3fn] = "tl.float8e4nv"
    if hasattr(torch, "float8_e4m3fnuz") and hasattr(tl, "float8e4b15"):
        type_dict[torch.float8_e4m3fnuz] = tl.float8e4b15
        type_str_dict[torch.float8_e4m3fnuz] = "tl.float8e4b15"
    if hasattr(torch, "float8_e5m2"):
        if hasattr(tl, "float8e5m2"):
            type_dict[torch.float8_e5m2] = tl.float8e5m2
            type_str_dict[torch.float8_e5m2] = "tl.float8e5m2"
        elif hasattr(tl, "float8e5"):
            type_dict[torch.float8_e5m2] = tl.float8e5
            type_str_dict[torch.float8_e5m2] = "tl.float8e5"
    if hasattr(torch, "float8_e5m2fnuz") and hasattr(tl, "float8e5b16"):
        type_dict[torch.float8_e5m2fnuz] = tl.float8e5b16
        type_str_dict[torch.float8_e5m2fnuz] = "tl.float8e5b16"
except (ImportError, OSError, AttributeError, RuntimeError) as e:
    _TRITON_IMPORT_ERROR = e
    logger.info("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()
    type_dict = {}
    type_str_dict = {}


_TRITON_COMPUTE_DTYPE_ALIASES = {
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

_TRITON_STORAGE_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}
if hasattr(torch, "bfloat16"):
    _TRITON_STORAGE_DTYPE_ALIASES.update(
        {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    )
if hasattr(torch, "float8_e4m3fn"):
    _TRITON_STORAGE_DTYPE_ALIASES.update(
        {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "e4m3fn": torch.float8_e4m3fn,
        }
    )
if hasattr(torch, "float8_e5m2"):
    _TRITON_STORAGE_DTYPE_ALIASES.update(
        {
            "float8_e5m2": torch.float8_e5m2,
            "fp8_e5m2": torch.float8_e5m2,
            "e5m2": torch.float8_e5m2,
        }
    )

_TRITON_STORAGE_DTYPES = {torch.float32, torch.float16}
if hasattr(torch, "bfloat16"):
    _TRITON_STORAGE_DTYPES.add(torch.bfloat16)
if hasattr(torch, "float8_e4m3fn"):
    _TRITON_STORAGE_DTYPES.add(torch.float8_e4m3fn)
if hasattr(torch, "float8_e5m2"):
    _TRITON_STORAGE_DTYPES.add(torch.float8_e5m2)

TRITON_NEURON_DTYPE_FP32 = 0
TRITON_NEURON_DTYPE_FP16 = 1
TRITON_NEURON_DTYPE_BF16 = 2
TRITON_NEURON_DTYPE_FP8_E4M3FN = 3
TRITON_NEURON_DTYPE_FP8_E5M2 = 4


def normalize_triton_compute_dtype_name(compute_dtype: str | torch.dtype) -> str:
    if isinstance(compute_dtype, torch.dtype):
        if compute_dtype == torch.float32:
            return "fp32"
        if compute_dtype == torch.float16:
            return "fp16"
        if hasattr(torch, "bfloat16") and compute_dtype == torch.bfloat16:
            return "bf16"
        if hasattr(torch, "float8_e4m3fn") and compute_dtype == torch.float8_e4m3fn:
            return "fp8"
        if hasattr(torch, "float8_e5m2") and compute_dtype == torch.float8_e5m2:
            return "fp8"
        raise ValueError(f"Unsupported Triton compute dtype: {compute_dtype}.")
    if not isinstance(compute_dtype, str):
        raise ValueError(
            "compute_dtype must be a string or torch.dtype, "
            f"but got {type(compute_dtype).__name__}."
        )
    key = compute_dtype.lower()
    try:
        return _TRITON_COMPUTE_DTYPE_ALIASES[key]
    except KeyError as e:
        raise ValueError(
            "compute_dtype must be one of 'fp8', 'fp16', 'bf16', or 'fp32', "
            f"but got {compute_dtype!r}."
        ) from e


def normalize_triton_storage_dtype(storage_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(storage_dtype, torch.dtype):
        dtype = storage_dtype
    elif isinstance(storage_dtype, str):
        key = storage_dtype.lower().replace("torch.", "")
        if key in {"fp8", "float8"}:
            raise ValueError(
                "storage_dtype='fp8' is ambiguous; use 'float8_e4m3fn' "
                "or 'float8_e5m2'."
            )
        try:
            dtype = _TRITON_STORAGE_DTYPE_ALIASES[key]
        except KeyError as e:
            raise ValueError(
                f"Unsupported Triton storage dtype: {storage_dtype!r}."
            ) from e
    else:
        raise ValueError(
            "storage_dtype must be a string or torch.dtype, "
            f"but got {type(storage_dtype).__name__}."
        )

    if dtype not in _TRITON_STORAGE_DTYPES:
        raise ValueError(f"Unsupported Triton storage dtype: {dtype}.")
    return dtype


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return (hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn) or (
        hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2
    )


def torch_dtype_to_triton_neuron_dtype_id(dtype: torch.dtype) -> int:
    dtype = normalize_triton_storage_dtype(dtype)
    if dtype == torch.float32:
        return TRITON_NEURON_DTYPE_FP32
    if dtype == torch.float16:
        return TRITON_NEURON_DTYPE_FP16
    if hasattr(torch, "bfloat16") and dtype == torch.bfloat16:
        return TRITON_NEURON_DTYPE_BF16
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return TRITON_NEURON_DTYPE_FP8_E4M3FN
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return TRITON_NEURON_DTYPE_FP8_E5M2
    raise ValueError(f"Unsupported Triton neuron dtype: {dtype}.")


def triton_neuron_dtype_id_to_torch_dtype(dtype_id: int) -> torch.dtype:
    if dtype_id == TRITON_NEURON_DTYPE_FP32:
        return torch.float32
    if dtype_id == TRITON_NEURON_DTYPE_FP16:
        return torch.float16
    if dtype_id == TRITON_NEURON_DTYPE_BF16:
        if not hasattr(torch, "bfloat16"):
            raise ValueError("torch.bfloat16 is unavailable.")
        return torch.bfloat16
    if dtype_id == TRITON_NEURON_DTYPE_FP8_E4M3FN:
        if not hasattr(torch, "float8_e4m3fn"):
            raise ValueError("torch.float8_e4m3fn is unavailable.")
        return torch.float8_e4m3fn
    if dtype_id == TRITON_NEURON_DTYPE_FP8_E5M2:
        if not hasattr(torch, "float8_e5m2"):
            raise ValueError("torch.float8_e5m2 is unavailable.")
        return torch.float8_e5m2
    raise ValueError(f"Unsupported Triton neuron dtype id: {dtype_id}.")


def triton_compute_dtype_name_to_neuron_dtype_id(
    compute_dtype_name: str, storage_dtype: torch.dtype
) -> int:
    name = normalize_triton_compute_dtype_name(compute_dtype_name)
    if name == "fp32":
        return TRITON_NEURON_DTYPE_FP32
    if name == "fp16":
        return TRITON_NEURON_DTYPE_FP16
    if name == "bf16":
        return TRITON_NEURON_DTYPE_BF16
    if name == "fp8":
        storage_dtype = normalize_triton_storage_dtype(storage_dtype)
        if not is_fp8_dtype(storage_dtype):
            raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
        return torch_dtype_to_triton_neuron_dtype_id(storage_dtype)
    raise ValueError(f"Unsupported Triton compute dtype name: {compute_dtype_name!r}.")


def triton_neuron_compute_dtype_id_to_tl_dtype(dtype_id: int, storage_dtype_id: int):
    if dtype_id == TRITON_NEURON_DTYPE_FP32:
        if torch.float32 not in type_dict:
            raise ValueError("Triton fp32 compute dtype is unavailable.")
        return type_dict[torch.float32]
    if dtype_id == TRITON_NEURON_DTYPE_FP16:
        if torch.float16 not in type_dict:
            raise ValueError("Triton fp16 compute dtype is unavailable.")
        return type_dict[torch.float16]
    if dtype_id == TRITON_NEURON_DTYPE_BF16:
        if not hasattr(torch, "bfloat16") or torch.bfloat16 not in type_dict:
            raise ValueError("Triton bfloat16 compute dtype is unavailable.")
        return type_dict[torch.bfloat16]
    if dtype_id == TRITON_NEURON_DTYPE_FP8_E4M3FN:
        if storage_dtype_id != TRITON_NEURON_DTYPE_FP8_E4M3FN:
            raise ValueError("FP8 E4M3 compute requires E4M3 storage dtype.")
        tl_dtype = getattr(tl, "float8e4m3fn", None) or getattr(tl, "float8e4nv", None)
        if tl_dtype is None:
            raise ValueError("Triton float8e4m3fn/float8e4nv dtype is unavailable.")
        return tl_dtype
    if dtype_id == TRITON_NEURON_DTYPE_FP8_E5M2:
        if storage_dtype_id != TRITON_NEURON_DTYPE_FP8_E5M2:
            raise ValueError("FP8 E5M2 compute requires E5M2 storage dtype.")
        tl_dtype = getattr(tl, "float8e5m2", None) or getattr(tl, "float8e5", None)
        if tl_dtype is None:
            raise ValueError("Triton float8e5m2/float8e5 dtype is unavailable.")
        return tl_dtype
    raise ValueError(f"Unsupported Triton neuron compute dtype id: {dtype_id}.")


def resolve_triton_compute_dtype(
    compute_dtype: str | torch.dtype,
    storage_dtype: str | torch.dtype | None = None,
):
    name = normalize_triton_compute_dtype_name(compute_dtype)
    if name == "fp32":
        if torch.float32 not in type_dict:
            raise ValueError("Triton fp32 compute dtype is unavailable.")
        return type_dict[torch.float32]
    if name == "fp16":
        if torch.float16 not in type_dict:
            raise ValueError("Triton fp16 compute dtype is unavailable.")
        return type_dict[torch.float16]
    if name == "bf16":
        if not hasattr(torch, "bfloat16") or torch.bfloat16 not in type_dict:
            raise ValueError("Triton bfloat16 compute dtype is unavailable.")
        return type_dict[torch.bfloat16]
    if name == "fp8":
        if storage_dtype is None:
            raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
        storage_dtype = normalize_triton_storage_dtype(storage_dtype)
        if not is_fp8_dtype(storage_dtype):
            raise ValueError("compute_dtype='fp8' requires an FP8 storage_dtype.")
        if hasattr(torch, "float8_e4m3fn") and storage_dtype == torch.float8_e4m3fn:
            tl_dtype = getattr(tl, "float8e4m3fn", None) or getattr(
                tl, "float8e4nv", None
            )
            if tl_dtype is None:
                raise ValueError("Triton float8e4m3fn/float8e4nv dtype is unavailable.")
            return tl_dtype
        if hasattr(torch, "float8_e5m2") and storage_dtype == torch.float8_e5m2:
            tl_dtype = getattr(tl, "float8e5m2", None) or getattr(tl, "float8e5", None)
            if tl_dtype is None:
                raise ValueError("Triton float8e5m2/float8e5 dtype is unavailable.")
            return tl_dtype
        raise ValueError(
            f"Unsupported FP8 storage dtype for compute_dtype='fp8': {storage_dtype}."
        )
    raise ValueError(f"Unsupported Triton compute dtype name: {name!r}.")


def torch_dtype_for_triton_neuron_compute_dtype_id(dtype_id: int) -> torch.dtype:
    if dtype_id in (
        TRITON_NEURON_DTYPE_FP8_E4M3FN,
        TRITON_NEURON_DTYPE_FP8_E5M2,
    ):
        # PyTorch does not provide useful reductions for float8 tensors. Keep
        # reduction buffers in fp32 while the Triton kernel computes in fp8.
        return torch.float32
    return triton_neuron_dtype_id_to_torch_dtype(dtype_id)


def normalize_cuda_device(device: torch.device | str) -> torch.device:
    r"""
    **API Language** - :ref:`中文 <normalize_cuda_device-cn>` | :ref:`English <normalize_cuda_device-en>`

    ----

    .. _normalize_cuda_device-cn:

    * **中文**

    将设备转换为 :class:`torch.device`；CUDA 可用时，把未指定序号的
    ``"cuda"`` 解析为当前 CUDA 设备。

    :param device: 设备对象或设备字符串。
    :type device: torch.device | str
    :return: 带明确当前 CUDA 序号的设备，或原类型对应的非 CUDA 设备。
    :rtype: torch.device

    ----

    .. _normalize_cuda_device-en:

    * **English**

    Convert ``device`` to :class:`torch.device`. When CUDA is available, resolve
    an unindexed ``"cuda"`` device to the current CUDA device.

    :param device: Device object or device string.
    :type device: torch.device | str
    :return: Device with an explicit current CUDA index, or the corresponding
        non-CUDA device.
    :rtype: torch.device
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


_USE_TRITON_OP = (
    _TRITON_IMPORT_ERROR is None
    and triton_op is not None
    and os.getenv("SJ_USE_TRITON_OP", "1").strip().lower()
    not in ("0", "false", "off", "no")
)


def register_op(opname: str, mutates_args=()):
    if _USE_TRITON_OP:
        return triton_op(opname, mutates_args=mutates_args)
    custom_op = torch.library.custom_op(
        opname, mutates_args=mutates_args, device_types="cuda"
    )
    if _TRITON_IMPORT_ERROR is None:
        return custom_op

    def register_missing_triton(f):
        @functools.wraps(f)
        def unavailable(*args, **kwargs):
            raise ImportError(
                "Triton is not installed or failed to initialize."
            ) from _TRITON_IMPORT_ERROR

        return custom_op(unavailable)

    return register_missing_triton


if _USE_TRITON_OP:
    wrap_triton = torch.library.wrap_triton
else:

    def wrap_triton(kernel):
        return kernel


def use_static_range_for_triton_neuron_kernel(T: int) -> bool:
    threshold = configure.triton_neuron_kernel_static_range_max_T
    if threshold is None:
        return True
    return T <= threshold
