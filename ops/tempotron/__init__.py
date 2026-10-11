"""Private Tempotron block-state, threshold-root and weight-gradient operators."""

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu

_dispatches = []
for _operation, _environment in (
    ("roots", "SJ_TEMPOTRON_ROOTS_CUDA_IMPLEMENTATION"),
    ("states", "SJ_TEMPOTRON_RESPONSE_CUDA_IMPLEMENTATION"),
    ("weight_gradient", "SJ_TEMPOTRON_WEIGHT_GRADIENT_CUDA_IMPLEMENTATION"),
):
    _namespace = f"sj_tempotron_{_operation}"
    _forward_name = f"_{_operation}_impl"
    _selection = _CudaSelection(
        __name__,
        _namespace,
        _environment,
        getattr(_cpu, _forward_name),
        on_select=_update_cache_tag,
        forward_name=_forward_name,
    )
    _dispatches.append(_register_dispatch(_namespace, _cpu, _selection))

_roots = torch.ops.sj_tempotron_roots.forward.default
_states = torch.ops.sj_tempotron_states.forward.default
_weight_gradient = torch.ops.sj_tempotron_weight_gradient.forward.default
