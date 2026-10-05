import threading
import weakref
from typing import Any

import numpy as np
import torch

from spikingjelly import configure
from spikingjelly.logger import logger

try:
    import cupy
except (ImportError, OSError) as e:
    logger.info("Optional CuPy dependency unavailable: {}", e)
    cupy = None


_INT32_MAX = np.iinfo(np.int32).max


def _as_cupy_int32(value: int, name: str):
    if not (-_INT32_MAX - 1 <= value <= _INT32_MAX):
        raise OverflowError(
            f"{name}={value} exceeds int32 range required by CUDA kernel launch metadata."
        )
    return cupy.asarray(value, dtype=np.int32)


def _scalar_to_cupy(py_dict: dict, ref: str):
    device = py_dict[ref].get_device()
    dtype = py_dict[ref].dtype

    with DeviceEnvironment(device):
        for key, value in py_dict.items():
            if isinstance(value, float):
                if dtype == torch.float32:
                    value = cupy.asarray(value, dtype=np.float32)
                elif dtype == torch.float16:
                    value = cupy.asarray([value, value], dtype=np.float16)
                else:
                    raise NotImplementedError(dtype)
                py_dict[key] = value
            elif isinstance(value, int):
                py_dict[key] = _as_cupy_int32(value, key)


_PYOBJ_LOCK = threading.Lock()
_PYOBJ_NEXT_ID = 0
_PYOBJ_ID_TO_ENTRY: dict[int, tuple[int, weakref.ReferenceType]] = {}
_PYOBJ_OBJECT_ID_TO_ID: dict[int, int] = {}


def _drop_python_object_locked(obj_id: int) -> None:
    entry = _PYOBJ_ID_TO_ENTRY.pop(obj_id, None)
    if entry is not None and _PYOBJ_OBJECT_ID_TO_ID.get(entry[0]) == obj_id:
        _PYOBJ_OBJECT_ID_TO_ID.pop(entry[0], None)


def _on_python_object_finalize(obj_id: int) -> None:
    with _PYOBJ_LOCK:
        _drop_python_object_locked(obj_id)


def register_python_object(obj: object) -> int:
    global _PYOBJ_NEXT_ID
    with _PYOBJ_LOCK:
        object_id = id(obj)
        obj_id = _PYOBJ_OBJECT_ID_TO_ID.get(object_id)
        if obj_id is not None:
            entry = _PYOBJ_ID_TO_ENTRY.get(obj_id)
            if entry is not None and entry[1]() is obj:
                return obj_id
            _drop_python_object_locked(obj_id)

        obj_id = _PYOBJ_NEXT_ID
        _PYOBJ_NEXT_ID += 1
        _PYOBJ_OBJECT_ID_TO_ID[object_id] = obj_id
        _PYOBJ_ID_TO_ENTRY[obj_id] = (
            object_id,
            weakref.ref(obj, lambda _ref, _id=obj_id: _on_python_object_finalize(_id)),
        )
    return obj_id


def resolve_python_object(obj_id: int) -> Any:
    with _PYOBJ_LOCK:
        entry = _PYOBJ_ID_TO_ENTRY.get(obj_id)
        obj = None if entry is None else entry[1]()
        if obj is None:
            _drop_python_object_locked(obj_id)
            raise RuntimeError(f"Unknown python object id={obj_id}.")
        return obj


def cal_blocks(numel: int, threads: int = -1):
    r"""
    **API Language** - :ref:`中文 <cal_blocks-cn>` | :ref:`English <cal_blocks-en>`

    ----

    .. _cal_blocks-cn:

    * **中文**

    :param numel: 并行执行的CUDA内核的数量
    :type numel: int
    :param threads: 每个cuda block中threads的数量，默认为-1，表示使用 ``configure.cuda_threads``
    :type threads: int
    :return: blocks的数量
    :rtype: int

    此函数返回 blocks的数量，用来按照 ``kernel((blocks,), (configure.cuda_threads,), ...)`` 调用 :class:`cupy.RawKernel`

    ----

    .. _cal_blocks-en:

    * **English**

    :param numel: the number of parallel CUDA kernels
    :type numel: int
    :param threads: the number of threads in each cuda block.
        The defaule value is -1, indicating to use ``configure.cuda_threads``
    :type threads: int
    :return: the number of blocks
    :rtype: int

    Returns the number of blocks to call :class:`cupy.RawKernel` by ``kernel((blocks,), (threads,), ...)``
    """
    if threads == -1:
        threads = configure.cuda_threads
    return (numel + threads - 1) // threads


def get_contiguous(*args):
    r"""
    **API Language** - :ref:`中文 <get_contiguous-cn>` | :ref:`English <get_contiguous-en>`

    ----

    .. _get_contiguous-cn:

    * **中文**

    将 ``*args`` 中所有的 ``torch.Tensor`` 或 ``cupy.ndarray`` 进行连续化。

    .. note::

        连续化的操作无法in-place，因此本函数返回一个新的list。

    :return: 一个元素全部为连续的 ``torch.Tensor`` 或 ``cupy.ndarray`` 的 ``list``
    :rtype: list

    ----

    .. _get_contiguous-en:

    * **English**

    :return: a list that contains the contiguous ``torch.Tensor`` or ``cupy.ndarray``
    :rtype: list

    Makes ``torch.Tensor`` or ``cupy.ndarray`` in ``*args`` to be contiguous

    .. admonition:: Note
        :class: note

        The making contiguous operation can not be done in-place. Hence, this function will return a new list.
    """
    ret_list = []

    for item in args:
        if isinstance(item, torch.Tensor):
            ret_list.append(item.contiguous())

        elif isinstance(item, cupy.ndarray):
            ret_list.append(cupy.ascontiguousarray(item))
        else:
            raise TypeError(type(item))
    return ret_list


def wrap_args_to_raw_kernel(device: int, *args):
    r"""
    **API Language** - :ref:`中文 <wrap_args_to_raw_kernel-cn>` | :ref:`English <wrap_args_to_raw_kernel-en>`

    ----

    .. _wrap_args_to_raw_kernel-cn:

    * **中文**

    :param device: raw kernel运行的CUDA设备
    :type device: int
    :return: 一个包含用来调用 :class:`cupy.RawKernel` 的 ``tuple``
    :rtype: tuple

    此函数可以包装 ``torch.Tensor`` 和 ``cupy.ndarray`` 并将其作为 :class:`cupy.RawKernel.__call__` 的 ``args``

    ----

    .. _wrap_args_to_raw_kernel-en:

    * **English**

    :param device: on which CUDA device the raw kernel will run
    :type device: int
    :return: a ``tuple`` that contains args to call :class:`cupy.RawKernel`
    :rtype: tuple

    This function can wrap ``torch.Tensor`` or ``cupy.ndarray`` to ``args`` in :class:`cupy.RawKernel.__call__`
    """
    # note that the input must be contiguous
    # check device and get data_ptr from tensor
    ret_list = []
    for item in args:
        if isinstance(item, torch.Tensor):
            assert item.get_device() == device
            assert item.is_contiguous()
            ret_list.append(item.data_ptr())

        elif isinstance(item, cupy.ndarray):
            assert item.device.id == device
            assert item.flags["C_CONTIGUOUS"]
            ret_list.append(item)

        else:
            raise TypeError
    return tuple(ret_list)


class DeviceEnvironment:
    def __init__(self, device: int):
        r"""
        **API Language** - :ref:`中文 <DeviceEnvironment.__init__-cn>` | :ref:`English <DeviceEnvironment.__init__-en>`

        ----

        .. _DeviceEnvironment.__init__-cn:

        * **中文**

        这个模块可以被用作在指定的 ``device`` 上执行CuPy函数的上下文，用来避免 `torch.cuda.current_device()` 被CuPy意外改变( https://github.com/cupy/cupy/issues/6569 )。

        代码示例：

        .. code-block:: python

            with DeviceEnvironment(device):
                kernel((blocks,), (configure.cuda_threads,), ...)


        ----

        .. _DeviceEnvironment.__init__-en:

        * **English**

        :param device: the CUDA device
        :type device: int

        This module is used as a context to make CuPy use the specific device, and avoids `torch.cuda.current_device()` is changed by CuPy ( https://github.com/cupy/cupy/issues/6569 ).

        Codes example:

        .. code-block:: python

            with DeviceEnvironment(device):
                kernel((blocks,), (configure.cuda_threads,), ...)
        """
        self.device = device
        self.previous_device = None
        self._stream = None

    def __enter__(self):
        current_device = torch.cuda.current_device()
        if current_device != self.device:
            torch.cuda.set_device(self.device)
            self.previous_device = current_device
        if cupy is not None:
            self._stream = cupy.cuda.ExternalStream(
                torch.cuda.current_stream(self.device).cuda_stream
            )
            self._stream.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._stream is not None:
                self._stream.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self.previous_device is not None:
                torch.cuda.set_device(self.previous_device)
