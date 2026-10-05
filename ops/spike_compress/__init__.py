"""Registered little-endian binary spike packing."""

import torch

from ..selection import _CudaSelection
from . import cpu as _cpu  # noqa: F401

_selection = _CudaSelection(
    __name__,
    "sj_spike_compress",
    "SJ_SPIKE_COMPRESS_CUDA_IMPLEMENTATION",
    torch.ops.sj_spike_compress.cpu_forward.default,
)


def _pack(x):
    return _selection.get_trace_forward(x.device)(x)


def _unpack(packed, shape, dtype=torch.uint8):
    selected = _selection.get_trace_forward(packed.device)
    name = selected._schema.name.split("::")[1].replace("_forward", "_unpack")
    return getattr(torch.ops.sj_spike_compress, name).default(
        packed, list(shape), dtype
    )
