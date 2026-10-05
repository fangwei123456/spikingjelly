from ..native_loader import _load_native
from .autograd import _register_ops

_build_info = _load_native(__package__)
_register_ops("sj_if::native_forward", "sj_if::native_backward")
