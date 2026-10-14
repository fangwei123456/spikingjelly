"""CLIF eager CUDA selection; custom spike functions retain their own autograd."""

import torch

from ..selection import _CudaSelection, _require_automatic_torch
from . import cpu

_selection = _CudaSelection(
    __name__, "sj_clif", "SJ_CLIF_CUDA_IMPLEMENTATION", cpu._forward_impl
)


def step(x, v, m, tau, threshold, spike_function):
    if x.device.type != "cuda":
        return cpu._forward_impl(x, v, m, tau, threshold, spike_function)
    supported = (
        not torch.compiler.is_compiling()
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and x.dtype == v.dtype == m.dtype
        and x.device == v.device == m.device
        and x.shape == v.shape == m.shape
    )
    if not supported:
        _require_automatic_torch(
            _selection, x.device, "CLIF requires matching CUDA eager FP32/FP16/BF16 tensors"
        )
        return cpu._forward_impl(x, v, m, tau, threshold, spike_function)
    return _selection.get_cuda_forward(x.device)(
        x, v, m, tau, threshold, spike_function
    )
