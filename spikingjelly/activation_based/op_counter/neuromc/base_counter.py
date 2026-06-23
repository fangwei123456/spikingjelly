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
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        """
        Base counter for NeuroMC energy profiling framework.
        **API Language** - :ref:`中文 <NeuroMCBaseCounter-cn>` | :ref:`English <NeuroMCBaseCounter-en>`

        ----

        .. _NeuroMCBaseCounter-cn:

        * **中文**

        神经形态计算（NeuroMC）能耗分析框架的基类计数器。

        提供通用的操作计数记录基础设施，按执行阶段（前向/反向/优化器）、
        算子类型以及它们的组合来跟踪计算操作数量。子类通过实现具体的
        计数规则来针对不同操作类型（加法、乘法、比较等）进行精确统计。

        :param extra_rules: 额外的计数规则，以 ATen 算子为键
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: 额外忽略的模块类型列表
        :type extra_ignore_modules: list[nn.Module] | None

        ----

        .. _NeuroMCBaseCounter-en:

        * **English**

        Base counter for the NeuroMC energy profiling framework.

        Provides a generic recording infrastructure for tracking operation counts
        across execution stages (forward/backward/optimizer), operator types,
        and their combinations. Subclasses implement specific counting rules
        for different operation types such as addition, multiplication, and comparison.

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
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
