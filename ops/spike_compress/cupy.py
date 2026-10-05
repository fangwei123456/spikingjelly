import math
from functools import lru_cache

import cupy
import numpy as np
import torch

from .validation import _check_pack, _check_unpack, _pack_fake, _unpack_fake

_SOURCE = r"""
extern "C" __global__ void pack(const bool* x, unsigned char* y, long long size) {
    long long byte = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (byte * 8 >= size) return;
    unsigned char value = 0;
    for (int bit = 0; bit < 8; ++bit)
        if (byte * 8 + bit < size) value |= ((unsigned char)x[byte * 8 + bit]) << bit;
    y[byte] = value;
}
extern "C" __global__ void unpack(const unsigned char* x, unsigned char* y, long long size) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < size) y[i] = (x[i / 8] >> (i % 8)) & 1;
}
"""


@lru_cache(maxsize=2)
def _kernel(name):
    return cupy.RawKernel(_SOURCE, name)


@torch.library.custom_op(
    "sj_spike_compress::cupy_forward", mutates_args=(), device_types="cuda"
)
def _pack(x: torch.Tensor) -> torch.Tensor:
    _check_pack(x)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x = x.bool().contiguous()
        packed = torch.empty(
            ((x.numel() + 7) // 8,), device=x.device, dtype=torch.uint8
        )
        if packed.numel():
            _kernel("pack")(
                ((packed.numel() + 255) // 256,),
                (256,),
                (
                    np.uint64(x.data_ptr()),
                    np.uint64(packed.data_ptr()),
                    np.int64(x.numel()),
                ),
                stream=cupy.cuda.ExternalStream(
                    torch.cuda.current_stream(x.device).cuda_stream
                ),
            )
    return packed


@torch.library.custom_op(
    "sj_spike_compress::cupy_unpack", mutates_args=(), device_types="cuda"
)
def _unpack(packed: torch.Tensor, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    _check_unpack(packed, shape)
    with torch.cuda.device(packed.device), cupy.cuda.Device(packed.device.index):
        packed = packed.contiguous()
        result = torch.empty(shape, device=packed.device, dtype=torch.uint8)
        size = math.prod(shape)
        if size:
            _kernel("unpack")(
                ((size + 255) // 256,),
                (256,),
                (
                    np.uint64(packed.data_ptr()),
                    np.uint64(result.data_ptr()),
                    np.int64(size),
                ),
                stream=cupy.cuda.ExternalStream(
                    torch.cuda.current_stream(packed.device).cuda_stream
                ),
            )
    return result.to(dtype)


torch.library.register_fake("sj_spike_compress::cupy_forward", _pack_fake)
torch.library.register_fake("sj_spike_compress::cupy_unpack", _unpack_fake)
