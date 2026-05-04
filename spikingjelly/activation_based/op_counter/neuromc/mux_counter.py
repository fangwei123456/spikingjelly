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
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.where.self: _mux_where,
            aten.where.ScalarOther: _mux_where,
            aten.where.ScalarSelf: _mux_where,
        }
        self.rules.update(extra_rules)
