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
    """Counter for multiplexer/selection operations in the NeuroMC framework.

    Tracks operations that involve selecting or routing values based on
    conditions, such as masked selects and conditional moves.
    """

    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        """
        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        :return: None
        :rtype: None
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
