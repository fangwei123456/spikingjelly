from __future__ import annotations

import torch

from ..base import _infer_stage, _tensor_bits

__all__ = [
    "_add_nested",
    "_diff_simple_dict",
    "_diff_nested_dict",
    "_prod",
    "_is_spike",
    "_spike_nnz",
    "_tensor_bits",
    "_infer_stage",
]


def _prod(dims) -> int:
    """Compute the product of a sequence of dimensions.
    :param dims: Sequence of dimension sizes
    :type dims: Iterable[int]
    :return: Product of all dimensions
    :rtype: int
    """
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _add_nested(dst: dict[str, int], src: dict[str, int]):
    """Add values from ``src`` into ``dst`` in-place, key by key.
    :param dst: Destination dictionary (modified in-place)
    :type dst: dict[str, int]
    :param src: Source dictionary
    :type src: dict[str, int]
    """
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _diff_simple_dict(new: dict[str, int], old: dict[str, int]) -> dict[str, int]:
    """Compute the element-wise difference between two flat dictionaries.
    Only keys with non-zero deltas are included in the result.

    :param new: New dictionary
    :type new: dict[str, int]
    :param old: Old dictionary
    :type old: dict[str, int]
    :return: Dictionary of (key, delta) pairs where delta != 0
    :rtype: dict[str, int]
    """
    keys = set(new.keys()) | set(old.keys())
    out: dict[str, int] = {}
    for k in keys:
        delta = int(new.get(k, 0) - old.get(k, 0))
        if delta != 0:
            out[k] = delta
    return out


def _diff_nested_dict(
    new: dict[str, dict[str, int]], old: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Compute the nested element-wise difference between two dictionaries.
    :param new: New dictionary with nested structure
    :type new: dict[str, dict[str, int]]
    :param old: Old dictionary with nested structure
    :type old: dict[str, dict[str, int]]
    :return: Nested dictionary of deltas where inner values != 0
    :rtype: dict[str, dict[str, int]]
    """
    keys = set(new.keys()) | set(old.keys())
    out: dict[str, dict[str, int]] = {}
    for k in keys:
        delta = _diff_simple_dict(new.get(k, {}), old.get(k, {}))
        if delta:
            out[k] = delta
    return out


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
