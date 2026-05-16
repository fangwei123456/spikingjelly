from __future__ import annotations

from typing import Any

import torch

from .base import is_binary_tensor

__all__ = [
    "active_element_count",
    "dense_bytes",
    "dense_bytes_tree",
    "is_sparse_access_tensor",
    "sparse_bytes",
    "sparse_bytes_tree",
]


def active_element_count(x: torch.Tensor) -> int:
    return int(x.count_nonzero().item())


def dense_bytes(x: torch.Tensor) -> int:
    return int(x.numel()) * int(x.element_size())


def sparse_bytes(x: torch.Tensor) -> int:
    return active_element_count(x) * int(x.element_size())


def is_sparse_access_tensor(
    x: torch.Tensor, *, zero_ratio_threshold: float = 0.5
) -> bool:
    if x.numel() == 0:
        return False
    if x.is_meta:
        return False
    if x.dtype == torch.bool or is_binary_tensor(x):
        return True
    zero_ratio = 1.0 - float(x.count_nonzero().item()) / float(x.numel())
    return zero_ratio >= zero_ratio_threshold


def dense_bytes_tree(tree: Any) -> int:
    if torch.is_tensor(tree):
        return dense_bytes(tree)
    if isinstance(tree, (tuple, list)):
        return sum(dense_bytes_tree(item) for item in tree)
    if isinstance(tree, dict):
        return sum(dense_bytes_tree(item) for item in tree.values())
    return 0


def sparse_bytes_tree(tree: Any, *, zero_ratio_threshold: float = 0.5) -> int:
    if torch.is_tensor(tree):
        if is_sparse_access_tensor(tree, zero_ratio_threshold=zero_ratio_threshold):
            return sparse_bytes(tree)
        return dense_bytes(tree)
    if isinstance(tree, (tuple, list)):
        return sum(
            sparse_bytes_tree(item, zero_ratio_threshold=zero_ratio_threshold)
            for item in tree
        )
    if isinstance(tree, dict):
        return sum(
            sparse_bytes_tree(item, zero_ratio_threshold=zero_ratio_threshold)
            for item in tree.values()
        )
    return 0
