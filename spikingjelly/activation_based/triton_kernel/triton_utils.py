"""Borrowed from:
https://github.com/AllenYolk/flash-snn/tree/main/flashsnn/utils
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""
from typing import Callable
import functools
import contextlib
from packaging import version
from pathlib import Path
import threading
import atexit
import torch

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
except BaseException as e:
    import logging
    logging.info(f'spikingjelly.activation_based.triton_kernel.triton_utils: {e}')
    triton = None
    tl = None
    type_dict = {}
    type_str_dict = {}

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)

def contiguous_and_device_guard(f: Callable) -> Callable:
    """Make sure all input tensors are contiguous and set to the same device.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous()
            for i in args
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
def _check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


if _check_pytorch_version('2.4'):
    amp_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    amp_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    amp_custom_fwd = torch.cuda.amp.custom_fwd
    amp_custom_bwd = torch.cuda.amp.custom_bwd

_CLEANUP_TMP_PYTHON_FILES_REGISTERED = False
_CLEANUP_TMP_PYTHON_FILES_REGISTERED_LOCK = threading.Lock()


def cleanup_tmp_python_files():
    print("Cleaning up temporary python files!")
    for f in Path("/tmp").glob("*.py"):
        try:
            f.unlink(missing_ok=True)
        except BaseException as e:
            pass # ignore the errors


def ensure_cleanup_tmp_python_files(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _CLEANUP_TMP_PYTHON_FILES_REGISTERED
        with _CLEANUP_TMP_PYTHON_FILES_REGISTERED_LOCK:
            if not _CLEANUP_TMP_PYTHON_FILES_REGISTERED:
                atexit.register(cleanup_tmp_python_files)
                _CLEANUP_TMP_PYTHON_FILES_REGISTERED = True
        return fn(*args, **kwargs)

    return wrapper
