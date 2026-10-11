import math

import torch


def _check_pack(x):
    torch._check(x.layout == torch.strided, lambda: "spikes must be strided")


def _check_unpack(packed, shape):
    torch._check(
        packed.layout == torch.strided
        and packed.ndim == 1
        and packed.dtype == torch.uint8,
        lambda: "packed spikes must be a strided one-dimensional uint8 tensor",
    )
    for dimension in shape:
        torch._check(dimension >= 0, lambda: "shape dimensions must be nonnegative")
    torch._check(
        packed.numel() == (math.prod(shape) + 7) // 8,
        lambda: "packed length does not match the requested shape",
    )


def _pack_fake(x):
    _check_pack(x)
    return torch.empty(((x.numel() + 7) // 8,), device=x.device, dtype=torch.uint8)


def _unpack_fake(packed, shape, dtype):
    _check_unpack(packed, shape)
    return torch.empty(shape, device=packed.device, dtype=dtype)
