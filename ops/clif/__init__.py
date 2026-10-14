"""CLIF eager CUDA selection; custom spike functions retain their own autograd."""

import torch

from ..dispatch import _update_cache_tag
from ..selection import _CudaSelection, _require_automatic_torch
from . import cpu

_selection = _CudaSelection(
    __name__,
    "sj_clif",
    "SJ_CLIF_CUDA_IMPLEMENTATION",
    cpu._forward_impl,
    on_select=_update_cache_tag,
)


def step(x, v, m, tau, threshold, spike_function):
    if x.device.type != "cuda" or _selection._requested == "torch":
        return cpu._forward_impl(x, v, m, tau, threshold, spike_function)
    if _selection._requested == "auto" and torch.is_grad_enabled():
        _require_automatic_torch(
            _selection,
            x.device,
            "CLIF auto keeps Torch with gradients enabled; request triton explicitly",
        )
        return cpu._forward_impl(x, v, m, tau, threshold, spike_function)
    supported = (
        not torch.compiler.is_compiling()
        and not torch.version.hip
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and x.dtype == v.dtype == m.dtype
        and x.device == v.device == m.device
        and x.shape == v.shape == m.shape
        and isinstance(tau, (float, int))
        and isinstance(threshold, (float, int))
    )
    if not supported:
        _require_automatic_torch(
            _selection,
            x.device,
            "CLIF requires matching CUDA eager FP32/FP16/BF16 tensors",
        )
        return cpu._forward_impl(x, v, m, tau, threshold, spike_function)
    return _selection.get_cuda_forward(x.device)(
        x, v, m, tau, threshold, spike_function
    )
