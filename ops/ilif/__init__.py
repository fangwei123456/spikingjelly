"""Registered ilif implementations."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu

_selection = _CudaSelection(
    __name__,
    "sj_ilif",
    "SJ_ILIF_CUDA_IMPLEMENTATION",
    _cpu._forward_impl,
    _cpu._backward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_ilif", _cpu, _selection)
_forward = torch.ops.sj_ilif.forward.default
