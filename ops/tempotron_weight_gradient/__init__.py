"""Private Tempotron peak-voltage weight-gradient operator."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu

_selection = _CudaSelection(
    __name__,
    "sj_tempotron_weight_gradient",
    "SJ_TEMPOTRON_WEIGHT_GRADIENT_CUDA_IMPLEMENTATION",
    _cpu._forward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_tempotron_weight_gradient", _cpu, _selection)
_forward = torch.ops.sj_tempotron_weight_gradient.forward.default
