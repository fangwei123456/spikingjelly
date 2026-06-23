from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter

aten = torch.ops.aten

__all__ = ["NeuroMCMuxCounter"]


def _mux_where(args, kwargs, out):
    return int(out.numel())


class NeuroMCMuxCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        """
        Counter for multiplexer/selection operations in the NeuroMC framework.
        **API Language** - :ref:`中文 <NeuroMCMuxCounter-cn>` | :ref:`English <NeuroMCMuxCounter-en>`

        ----

        .. _NeuroMCMuxCounter-cn:

        * **中文**

        多路选择器（MUX）操作计数器，继承自 :class:`NeuroMCBaseCounter`。

        专门用于统计模型执行过程中多路选择器相关的操作次数，
        例如条件选择、数据路由等操作的开销。

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None

        ----

        .. _NeuroMCMuxCounter-en:

        * **English**

        Multiplexer (MUX) operation counter, inheriting from :class:`NeuroMCBaseCounter`.

        Counts operations that involve selecting or routing values based on
        conditions during model execution.

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        """
        if extra_rules is None:
            extra_rules = {}
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.where.self: _mux_where,
            aten.where.ScalarOther: _mux_where,
            aten.where.ScalarSelf: _mux_where,
        }
        self.rules.update(extra_rules)
