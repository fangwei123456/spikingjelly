from __future__ import annotations

from math import prod

import torch

__all__ = [
    "_conv_mul_add",
    "_is_spike",
    "_spike_nnz",
]


def _conv_mul_add(args, out):
    x, w, bias = args[:3]
    groups = args[8] if len(args) > 8 else 1
    mul_per_out = x.shape[1] // groups * prod(w.shape[2:])
    mul = out.numel() * mul_per_out
    add = out.numel() * max(mul_per_out - 1, 0)
    if bias is not None:
        add += out.numel()
    return int(mul), int(add)


def _is_spike(x: torch.Tensor | None) -> bool:
    """Check if a tensor contains binary spike values (0 or 1).
    :param x: Input tensor, may be ``None``
    :type x: torch.Tensor | None
    :return: ``True`` if all elements are 0 or 1 (boolean or numeric)
    :rtype: bool
    """
    if x is None or (not torch.is_tensor(x)):
        return False
    if x.dtype == torch.bool:
        return True
    if x.numel() == 0:
        return False
    return bool(x.eq(0).logical_or(x.eq(1)).all().item())


def _spike_nnz(x: torch.Tensor | None) -> int | None:
    """Count the number of non-zero elements in a binary spike tensor.
    Returns ``None`` if the tensor is not binary (not all 0/1).

    :param x: Input tensor, may be ``None``
    :type x: torch.Tensor | None
    :return: Number of non-zero elements, or ``None`` if not binary
    :rtype: int | None
    """
    if x is None or (not torch.is_tensor(x)):
        return None
    if x.dtype == torch.bool:
        return int(x.count_nonzero().item())
    if x.numel() == 0:
        return None
    is_binary = bool(x.eq(0).logical_or(x.eq(1)).all().item())
    if not is_binary:
        return None
    return int(x.count_nonzero().item())
