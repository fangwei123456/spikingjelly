from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch.nn as nn
from torch.overrides import resolve_name

from ..base import BaseCounter
from .utils import _infer_stage

__all__ = ["NeuroMCBaseCounter"]


class NeuroMCBaseCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {}
        self.ignore_modules: list[nn.Module] = []
        self.ignore_modules.extend(extra_ignore_modules)
        self.stage_records: dict[str, int] = defaultdict(int)
        self.op_records: dict[str, int] = defaultdict(int)
        self.stage_op_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        op_name = resolve_name(func)
        value = int(self.rules[func](args, kwargs, out))
        stage = _infer_stage(func, args, kwargs, out)
        self.stage_records[stage] += value
        self.op_records[op_name] += value
        self.stage_op_records[stage][op_name] += value
        return value

    def get_stage_counts(self) -> dict[str, int]:
        return dict(self.stage_records)

    def get_op_counts(self) -> dict[str, int]:
        return dict(self.op_records)

    def get_stage_op_counts(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_op_records.items()}
