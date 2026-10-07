import threading
import weakref
from typing import Any

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
