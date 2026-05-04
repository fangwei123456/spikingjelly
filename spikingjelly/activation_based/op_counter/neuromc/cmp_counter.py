from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter
from .utils import _prod

aten = torch.ops.aten

__all__ = ["NeuroMCCmpCounter"]


def _cmp_element_wise(args, kwargs, out):
    return int(out.numel())


def _cmp_max_pool2d_with_indices(args, kwargs, out):
    kernel_size = args[1]
    y = out[0]
    return int(y.numel() * max(_prod(kernel_size) - 1, 0))


class NeuroMCCmpCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        if extra_rules is None:
            extra_rules = {}
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.eq.Tensor: _cmp_element_wise,
            aten.eq.Scalar: _cmp_element_wise,
            aten.ne.Tensor: _cmp_element_wise,
            aten.ne.Scalar: _cmp_element_wise,
            aten.lt.Tensor: _cmp_element_wise,
            aten.lt.Scalar: _cmp_element_wise,
            aten.le.Tensor: _cmp_element_wise,
            aten.le.Scalar: _cmp_element_wise,
            aten.gt.Tensor: _cmp_element_wise,
            aten.gt.Scalar: _cmp_element_wise,
            aten.ge.Tensor: _cmp_element_wise,
            aten.ge.Scalar: _cmp_element_wise,
            aten.logical_and.default: _cmp_element_wise,
            aten.logical_or.default: _cmp_element_wise,
            aten.logical_xor.default: _cmp_element_wise,
            aten.logical_not.default: _cmp_element_wise,
            aten.max_pool2d_with_indices.default: _cmp_max_pool2d_with_indices,
        }
        self.rules.update(extra_rules)
