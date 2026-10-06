import threading
import weakref
from typing import Any

import torch

from spikingjelly.logger import logger

try:
    import cupy
except (ImportError, OSError) as e:
    logger.info("Optional CuPy dependency unavailable: {}", e)
    cupy = None


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
                kernel(grid, block, args)


        ----

        .. _DeviceEnvironment.__init__-en:

        * **English**

        :param device: the CUDA device
        :type device: int

        This module is used as a context to make CuPy use the specific device, and avoids `torch.cuda.current_device()` is changed by CuPy ( https://github.com/cupy/cupy/issues/6569 ).

        Codes example:

        .. code-block:: python

            with DeviceEnvironment(device):
                kernel(grid, block, args)
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
