"""Registered stbif inference implementations."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu
from . import reference as _reference

_selection = _CudaSelection(
    __name__,
    "sj_stbif",
    "SJ_STBIF_CUDA_IMPLEMENTATION",
    _reference._forward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_stbif", _cpu, _selection)
_forward = torch.ops.sj_stbif.forward.default
