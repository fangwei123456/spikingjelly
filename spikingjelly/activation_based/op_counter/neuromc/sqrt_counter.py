from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter

aten = torch.ops.aten

__all__ = ["NeuroMCSqrtCounter"]


def _sqrt_op(args, kwargs, out):
    return int(out.numel())


def _sqrt_native_batch_norm(args, kwargs, out):
    x = args[0]
    c = x.shape[1]
    return int(c)


class NeuroMCSqrtCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        """
        Counter for square root operations in the NeuroMC framework.
        **API Language:**
        :ref:`中文 <NeuroMCSqrtCounter-cn>` | :ref:`English <NeuroMCSqrtCounter-en>`

        ----

        .. _NeuroMCSqrtCounter-cn:

        * **中文**

        平方根运算计数器，继承自 :class:`NeuroMCBaseCounter`。

        专门用于统计模型执行过程中平方根（Sqrt）和逆平方根（Rsqrt）操作的次数，
        包括逐元素平方根计算以及 batch norm 中涉及的开方运算。

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None

        ----

        .. _NeuroMCSqrtCounter-en:

        * **English**

        Square root operation counter, inheriting from :class:`NeuroMCBaseCounter`.

        Counts square root and inverse square root operations performed during
        model execution, including element-wise sqrt/rsqrt operations and square
        root computations in batch normalization.

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        """
        if extra_rules is None:
            extra_rules = {}
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.sqrt.default: _sqrt_op,
            aten.sqrt_.default: _sqrt_op,
            aten.rsqrt.default: _sqrt_op,
            aten.native_batch_norm.default: _sqrt_native_batch_norm,
        }
        self.rules.update(extra_rules)
