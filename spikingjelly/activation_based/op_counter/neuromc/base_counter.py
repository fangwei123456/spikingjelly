from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch.nn as nn
from torch.overrides import resolve_name

from ..base import BaseCounter
from .utils import _infer_stage

__all__ = ["NeuroMCBaseCounter"]


class NeuroMCBaseCounter(BaseCounter):
    """Base counter for NeuroMC energy profiling framework.
    **API Language:**
    :ref:`中文 <NeuroMCBaseCounter-cn>` | :ref:`English <NeuroMCBaseCounter-en>`

    ----

    .. _NeuroMCBaseCounter-cn:

    * **中文**

    TODO: add Chinese description

    :rtype: None
    Provides generic recording infrastructure for tracking operation counts
    across stages, operators, and their combinations. Subclasses implement
    specific counting rules for different operation types.

    ----

    .. _NeuroMCBaseCounter-en:

    * **English**

    TODO: add English description

    :return: None
    :rtype: None
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
        if extra_ignore_modules is None:
            extra_ignore_modules = []
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {}
        self.rules.update(extra_rules)
        self.ignore_modules: list[nn.Module] = []
        self.ignore_modules.extend(extra_ignore_modules)
        self.stage_records: dict[str, int] = defaultdict(int)
        self.op_records: dict[str, int] = defaultdict(int)
        self.stage_op_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules: set[nn.Module] | None = None,
        parent_names: set[str] | None = None,
    ) -> int:
        rule = self.rules.get(func)
        if rule is None:
            return 0
        op_name = resolve_name(func)
        value = int(rule(args, kwargs, out))
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
