"""Registered activation_aware_if inference implementations."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu
from . import reference as _reference

_selection = _CudaSelection(
    __name__,
    "sj_activation_aware_if",
    "SJ_ACTIVATION_AWARE_IF_CUDA_IMPLEMENTATION",
    _reference._forward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_activation_aware_if", _cpu, _selection)
_forward = torch.ops.sj_activation_aware_if.forward.default
