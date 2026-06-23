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
        """
        Counter for comparison operations in the NeuroMC framework.
        **API Language** - :ref:`中文 <NeuroMCCmpCounter-cn>` | :ref:`English <NeuroMCCmpCounter-en>`

        ----

        .. _NeuroMCCmpCounter-cn:

        * **中文**

        NeuroMC比较运算计数器，继承自 :class:`NeuroMCBaseCounter`。

        专门用于在模型前向传播和反向传播过程中统计比较（Comparison）操作次数，
        包括逐元素比较（eq、ne、lt、le、gt、ge）和最大池化索引操作。
        覆盖的算子包括 ``aten.eq``、``aten.ne``、``aten.lt``、``aten.le``、``aten.gt``、
        ``aten.ge``、``aten.logical_and``、``aten.logical_or``、``aten.logical_xor``、
        ``aten.logical_not`` 以及 ``aten.max_pool2d_with_indices``。

        :param extra_rules: 以 ATen 算子为键的额外计数规则
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: 额外忽略的模块类型列表
        :type extra_ignore_modules: list[nn.Module] | None

        ----

        .. _NeuroMCCmpCounter-en:

        * **English**

        Comparison operation counter, inheriting from :class:`NeuroMCBaseCounter`.

        Counts comparison operations during model forward and backward passes,
        including element-wise comparisons (eq, ne, lt, le, gt, ge) and max-pooling
        indexing operations. Covers operators such as ``aten.eq``, ``aten.ne``,
        ``aten.lt``, ``aten.le``, ``aten.gt``, ``aten.ge``, ``aten.logical_and``,
        ``aten.logical_or``, ``aten.logical_xor``, ``aten.logical_not``, and
        ``aten.max_pool2d_with_indices``.

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        """
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
