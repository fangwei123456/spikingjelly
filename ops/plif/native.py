from ..native_loader import _load_native
from .autograd import _register_ops

_build_info = _load_native(__package__)
_register_ops("sj_plif::native_forward", "sj_plif::native_backward")
