"""Borrowed from:
https://github.com/AllenYolk/flash-snn/tree/main/flashsnn/utils
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""

import contextlib
import functools
import os
from typing import Callable

import torch
from packaging import version

from . import dummy

try:
    from torch.library import triton_op

    _TRITON_OP_AVAILABLE = True
except BaseException:
    triton_op = dummy.DummyImport()
    _TRITON_OP_AVAILABLE = False

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
    dc = torch.cuda.get_device_capability()
    if dc[0] < 8 or not hasattr(tl, "bfloat16") or not hasattr(torch, "bfloat16"):
        print("bfloat16 is not supported on this device.")
    else:
        type_dict[torch.bfloat16] = tl.bfloat16
        type_str_dict[torch.bfloat16] = "tl.bfloat16"
except BaseException as e:
    import logging

    logging.info(f"spikingjelly.activation_based.triton_kernel.triton_utils: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()
    type_dict = {}
    type_str_dict = {}


@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)


def _env_flag_enabled(var_name: str) -> bool:
    v = os.getenv(var_name)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "off", "no")


def register_op(opname: str, mutates_args=()):
    if _env_flag_enabled("SJ_USE_TRITON_OP") and _TRITON_OP_AVAILABLE:
        return triton_op(opname, mutates_args=mutates_args)
    return torch.library.custom_op(opname, mutates_args=mutates_args)


def wrap_triton(kernel):
    if (
        _TRITON_OP_AVAILABLE
        and _env_flag_enabled("SJ_USE_TRITON_OP")
        and _env_flag_enabled("SJ_USE_WRAP_TRITON")
    ):
        return torch.library.wrap_triton(kernel)
    return kernel


def contiguous_and_device_guard(f: Callable) -> Callable:
    """Make sure all input tensors are contiguous and set to the same device."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        # find the first tensor in the argument list
        first_tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                first_tensor = arg
                break
        if first_tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    first_tensor = value
                    break
        if first_tensor is not None:
            ctx = torch.cuda.device(first_tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return f(*contiguous_args, **contiguous_kwargs)

    return wrapper


@functools.lru_cache(maxsize=None)
def _check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


if _check_pytorch_version("2.4"):
    amp_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    amp_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    amp_custom_fwd = torch.cuda.amp.custom_fwd
    amp_custom_bwd = torch.cuda.amp.custom_bwd

def cleanup_tmp_python_files():
    return None


def ensure_cleanup_tmp_python_files(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
