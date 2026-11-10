"""Inference dispatch for the few-spike table decoders."""

import torch

from ..dispatch import _update_cache_tag
from ..selection import _CudaSelection, _require_automatic_torch
from . import cpu

_selection = _CudaSelection(
    __name__,
    "sj_few_spike",
    "SJ_FEW_SPIKE_CUDA_IMPLEMENTATION",
    cpu._forward_impl,
    on_select=_update_cache_tag,
)


def _forward(
    gate,
    theta,
    h,
    d,
    return_sequence,
    *,
    outlier=None,
    gates=None,
    split_threshold=0.0,
    clamp_value=None,
):
    valid = (
        theta.ndim == (2 if gates is not None else 1)
        and theta.shape == h.shape == d.shape
        and theta.numel() > 0
    )
    if gates is not None:
        valid = valid and gates.ndim == 1 and gates.numel() == theta.shape[0] - 1
    if clamp_value is not None:
        valid = valid and abs(clamp_value) <= torch.finfo(torch.float32).max
    if outlier is not None:
        valid = valid and gates is None and all(t.shape == theta.shape for t in outlier)
    if not valid:
        if _selection._requested not in ("auto", "torch"):
            _require_automatic_torch(
                _selection,
                gate.device,
                "incompatible few-spike table shapes or scalar range",
            )
        return None
    other_theta, other_h, other_d = (
        outlier if outlier is not None else (None, None, None)
    )
    return _selection.get_cuda_forward(gate.device)(
        gate,
        theta,
        h,
        d,
        return_sequence,
        other_theta,
        other_h,
        other_d,
        gates,
        split_threshold,
        clamp_value,
    )
