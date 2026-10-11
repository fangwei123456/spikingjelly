"""Registered little-endian binary spike packing."""

import torch

from ..selection import _CudaSelection
from . import cpu as _cpu

_selection = _CudaSelection(
    __name__,
    "sj_spike_compress",
    "SJ_SPIKE_COMPRESS_CUDA_IMPLEMENTATION",
    torch.ops.sj_spike_compress.cpu_forward.default,
)


def _pack(x):
    return _selection.get_trace_forward(x.device)(x)


def _unpack(packed, shape, dtype=torch.uint8):
    implementation = (
        _cpu._unpack
        if packed.device.type == "cpu"
        else _selection._get_cuda_selection(packed.device).unpack
    )
    return implementation(packed, list(shape), dtype)
