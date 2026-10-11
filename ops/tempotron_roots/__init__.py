"""Private FP64 Tempotron threshold-root operator."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu

_selection = _CudaSelection(
    __name__,
    "sj_tempotron_roots",
    "SJ_TEMPOTRON_ROOTS_CUDA_IMPLEMENTATION",
    _cpu._forward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_tempotron_roots", _cpu, _selection)
_forward = torch.ops.sj_tempotron_roots.forward.default
