import math

import torch

from .validation import _check_pack, _check_unpack, _pack_fake, _unpack_fake


def _forward_impl(x: torch.Tensor) -> torch.Tensor:
    _check_pack(x)
    x = x.bool().reshape(-1)
    packed = torch.zeros(((x.numel() + 7) // 8,), device=x.device, dtype=torch.uint8)
    for bit in range(8):
        part = x[bit::8].to(torch.uint8)
        packed[: part.numel()] |= part << bit
    return packed


_pack = torch.library.custom_op(
    "sj_spike_compress::cpu_forward", mutates_args=(), device_types=("cpu", "cuda")
)(_forward_impl)


@torch.library.custom_op(
    "sj_spike_compress::cpu_unpack", mutates_args=(), device_types=("cpu", "cuda")
)
def _unpack(packed: torch.Tensor, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    _check_unpack(packed, shape)
    indices = torch.arange(math.prod(shape), device=packed.device)
    return ((packed[indices // 8] >> (indices % 8)) & 1).to(dtype).reshape(shape)


torch.library.register_fake("sj_spike_compress::cpu_forward", _pack_fake)
torch.library.register_fake("sj_spike_compress::cpu_unpack", _unpack_fake)
