from __future__ import annotations

from typing import Any

import torch
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

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


_OPTIMIZER_HINTS = (
    "add_.",
    "sub_.",
    "mul_.",
    "div_.",
    "addcmul",
    "addcdiv",
    "lerp_",
    "copy_",
)


def _prod(dims) -> int:
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _add_nested(dst: dict[str, int], src: dict[str, int]):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _diff_simple_dict(new: dict[str, int], old: dict[str, int]) -> dict[str, int]:
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
    keys = set(new.keys()) | set(old.keys())
    out: dict[str, dict[str, int]] = {}
    for k in keys:
        delta = _diff_simple_dict(new.get(k, {}), old.get(k, {}))
        if delta:
            out[k] = delta
    return out


def _is_spike(x: torch.Tensor | None) -> bool:
    if x is None or (not torch.is_tensor(x)):
        return False
    if x.dtype == torch.bool:
        return True
    if x.numel() == 0:
        return False
    return bool(x.eq(0).logical_or_(x.eq(1)).all().item())


def _spike_nnz(x: torch.Tensor | None) -> int | None:
    if x is None or (not torch.is_tensor(x)):
        return None
    if x.dtype == torch.bool:
        return int(x.count_nonzero().item())
    is_binary = bool(x.eq(0).logical_or_(x.eq(1)).all().item())
    if not is_binary:
        return None
    return int(x.count_nonzero().item())


def _tensor_bits(x: Any) -> int:
    if not torch.is_tensor(x):
        return 0
    return int(x.numel() * x.element_size() * 8)


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _infer_stage(func, args, kwargs, out) -> str:
    op_name = resolve_name(func)
    if "backward" in op_name:
        return "backward"

    if torch.is_grad_enabled():
        return "forward"

    tensors = _collect_tensors((args, kwargs, out))
    has_grad_tensor = any(t.requires_grad for t in tensors)
    is_optimizer_like = any(hint in op_name for hint in _OPTIMIZER_HINTS)
    if has_grad_tensor and is_optimizer_like:
        return "optimizer"

    return "forward"
