from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from ..base import DispatchCounterMode
from .add_counter import NeuroMCAddCounter
from .cmp_counter import NeuroMCCmpCounter
from .config import MemoryHierarchyConfig
from .memory_residency_counter import NeuroMCMemoryResidencyCounter
from .memory_traffic_counter import NeuroMCMemoryTrafficCounter
from .mul_counter import NeuroMCMulCounter
from .mux_counter import NeuroMCMuxCounter
from .sqrt_counter import NeuroMCSqrtCounter
from .utils import _add_nested, _diff_nested_dict, _diff_simple_dict

__all__ = [
    "MemoryHierarchyConfig",
    "NeuroMCEnergyProfiler",
    "NeuroMCRuntimeEnergyReport",
    "estimate_neuromc_runtime_energy",
]


_DEFAULT_OP_COST_PJ = {
    "mul": 0.812,
    "add": 0.548,
    "cmp": 0.056,
    "sqrt": 0.514,
    "mux": 0.548 * (1.0 / 16.0),
}

_DEFAULT_MEMORY_LEVEL_WEIGHTS = {
    "reg": 2.0,
    "sram": 1.0,
    "dram": 0.25,
}

_IGNORED_OP_PREFIXES = (
    "aten.detach",
    "aten.view",
    "aten.t.default",
    "aten.transpose",
    "aten.permute",
    "aten.expand",
    "aten.slice",
    "aten.select",
    "aten.alias",
    "aten._unsafe_view",
    "aten.as_strided",
)


@dataclass
class NeuroMCRuntimeEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <NeuroMCRuntimeEnergyReport-cn>` | :ref:`English <NeuroMCRuntimeEnergyReport-en>`

    ----

    .. _NeuroMCRuntimeEnergyReport-cn:

    * **中文**

    NeuroMC runtime 能耗统计报告结构体。
    ``NeuroMCEnergyProfiler.get_report`` 与 ``estimate_neuromc_runtime_energy`` 的返回对象，包含总能耗、分项能耗、
    primitive 计数、分层访存统计以及警告信息。

    字段说明：

    - ``energy_total_pj`` ：总能耗（pJ）
    - ``energy_compute_pj`` ：计算能耗（pJ）
    - ``energy_memory_pj`` ：访存能耗（pJ）
    - ``energy_by_stage`` ：按阶段聚合的能耗
    - ``energy_by_op`` ：按 aten 算子聚合的能耗
    - ``primitive_counts`` ：primitive 统计（总量/分阶段/分算子）
    - ``memory_bits_by_level`` ：分层级/分阶段/分算子的 bit 统计
    - ``warnings`` ：模型边界和不支持算子的提示

    ----

    .. _NeuroMCRuntimeEnergyReport-en:

    * **English**

    Runtime energy report data structure for NeuroMC profiling.
    Returned by ``NeuroMCEnergyProfiler.get_report`` and
    ``estimate_neuromc_runtime_energy``, including total and breakdown energy,
    primitive counts, memory-bit statistics, and warnings.

    Field summary:

    - ``energy_total_pj`` : total energy in pJ
    - ``energy_compute_pj`` : compute energy in pJ
    - ``energy_memory_pj`` : memory energy in pJ
    - ``energy_by_stage`` : stage-wise energy
    - ``energy_by_op`` : aten-op-wise energy
    - ``primitive_counts`` : primitive statistics (total/by stage/by op)
    - ``memory_bits_by_level`` : memory-bit statistics by level/stage/op
    - ``warnings`` : model-boundary and unsupported-op warnings
    """

    energy_total_pj: float
    energy_compute_pj: float
    energy_memory_pj: float
    energy_by_stage: dict[str, float]
    energy_by_op: dict[str, float]
    primitive_counts: dict[str, Any]
    memory_bits_by_level: dict[str, Any]
    warnings: list[str]


class _AtenTraceMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.op_counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        op_name = resolve_name(func)
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        return func(*args, **kwargs)


def _call_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _resolve_loss_fn(loss_fn: Callable | None):
    if loss_fn is None:
        return None
    if isinstance(loss_fn, nn.Module):
        return loss_fn
    return loss_fn


def _clear_existing_grads(model: nn.Module, optimizer: torch.optim.Optimizer | None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        return

    for p in model.parameters():
        p.grad = None


def _filter_unsupported_ops(
    op_counts: dict[str, int], supported_ops: set[str]
) -> list[str]:
    unsupported = []
    for op_name, count in op_counts.items():
        if op_name in supported_ops:
            continue
        if op_name.startswith(_IGNORED_OP_PREFIXES):
            continue
        unsupported.append(f"{op_name} (calls={count})")
    return sorted(unsupported)


def _rule_key_to_name(rule_key: Any) -> str | None:
    if isinstance(rule_key, str):
        return rule_key
    try:
        return resolve_name(rule_key)
    except Exception:
        return None


class NeuroMCEnergyProfiler:
    def __init__(
        self,
        *,
        core_type: str = "fp_soma",
        op_cost_pj: dict[str, float] | None = None,
        memory_config: MemoryHierarchyConfig | None = None,
        memory_level_weights: dict[str, float] | None = None,
        strict: bool = False,
        verbose: bool = False,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <NeuroMCEnergyProfiler.__init__-cn>` | :ref:`English <NeuroMCEnergyProfiler.__init__-en>`

        ----

        .. _NeuroMCEnergyProfiler.__init__-cn:

        * **中文**

        NeuroMC 能耗统计器，可以作为上下文管理器使用： ``with NeuroMCEnergyProfiler(...) as p: ...``

        - 在上下文内通过 ``with p.stage("阶段名")`` 标注统计阶段
        - 结束后调用 ``p.get_report()`` 生成能耗报告

        :param core_type: 计算核心类型标识，会写入报告 ``primitive_counts["core_type"]``
        :type core_type: str

        :param op_cost_pj: primitive 单位能耗映射（pJ/op），可覆盖默认值
        :type op_cost_pj: dict[str, float] | None

        :param memory_config: 访存层级配置，未提供时使用 ``MemoryHierarchyConfig.neuromc_like_v1()``
        :type memory_config: MemoryHierarchyConfig | None

        :param memory_level_weights: ``memory_model="weighted"`` 时使用的层级加权系数
        :type memory_level_weights: dict[str, float] | None

        :param strict: 透传给 ``DispatchCounterMode`` 的严格模式开关
        :type strict: bool

        :param verbose: 透传给 ``DispatchCounterMode`` 的详细日志开关
        :type verbose: bool

        :param extra_ignore_modules: 额外需要忽略的模块列表
        :type extra_ignore_modules: list[nn.Module]

        ----

        .. _NeuroMCEnergyProfiler.__init__-en:

        * **English**

        Runtime energy profiler for NeuroMC-style estimation. Should be used as a context manager:
        ``with NeuroMCEnergyProfiler(...) as p: ...``

        - mark stages with ``with p.stage("stage_name")``
        - generate report via ``p.get_report()`` at the end

        :param core_type: compute-core label stored in ``primitive_counts["core_type"]``
        :type core_type: str

        :param op_cost_pj: per-primitive energy map (pJ/op), overrides defaults
        :type op_cost_pj: dict[str, float] | None

        :param memory_config: memory hierarchy config; defaults to ``MemoryHierarchyConfig.neuromc_like_v1()``
        :type memory_config: MemoryHierarchyConfig | None

        :param memory_level_weights: per-level weights used when ``memory_model="weighted"``
        :type memory_level_weights: dict[str, float] | None

        :param strict: strict-mode flag forwarded to ``DispatchCounterMode``
        :type strict: bool

        :param verbose: verbose flag forwarded to ``DispatchCounterMode``
        :type verbose: bool

        :param extra_ignore_modules: extra modules to ignore
        :type extra_ignore_modules: list[nn.Module]
        """
        self.core_type = core_type
        self.op_cost = dict(_DEFAULT_OP_COST_PJ)
        if op_cost_pj is not None:
            self.op_cost.update(op_cost_pj)

        self.memory_config = memory_config or MemoryHierarchyConfig.neuromc_like_v1()
        self.memory_config.validate()
        self.memory_level_weights = (
            dict(_DEFAULT_MEMORY_LEVEL_WEIGHTS)
            if memory_level_weights is None
            else dict(memory_level_weights)
        )
        self.strict = strict
        self.verbose = verbose
        if extra_ignore_modules is None:
            extra_ignore_modules = []
        self.extra_ignore_modules = list(extra_ignore_modules)

        self._stage_stack: list[str] = []
        self._warnings: list[str] = []

        self._counters = self._create_counters()
        self._counter_list = [
            self._counters["mul"],
            self._counters["add"],
            self._counters["cmp"],
            self._counters["sqrt"],
            self._counters["mux"],
            self._counters["memory"],
        ]

        self._dispatch_mode = DispatchCounterMode(
            self._counter_list, strict=self.strict, verbose=self.verbose
        )
        self._trace_mode = _AtenTraceMode()

        self._active = False
        self._stage_primitive_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"mul": 0, "add": 0, "cmp": 0, "sqrt": 0, "mux": 0}
        )
        self._stage_memory_level_bits: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._stage_memory_level_rw_bits: dict[str, dict[str, dict[str, int]]] = (
            defaultdict(
                lambda: defaultdict(
                    lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
                )
            )
        )
        self._stage_memory_op_level_bits: dict[str, dict[str, dict[str, int]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )
        self._stage_move_bits_by_edge: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._stage_move_bits_by_op: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def _create_counters(self):
        mul_counter = NeuroMCMulCounter(extra_ignore_modules=self.extra_ignore_modules)
        add_counter = NeuroMCAddCounter(extra_ignore_modules=self.extra_ignore_modules)
        cmp_counter = NeuroMCCmpCounter(extra_ignore_modules=self.extra_ignore_modules)
        sqrt_counter = NeuroMCSqrtCounter(
            extra_ignore_modules=self.extra_ignore_modules
        )
        mux_counter = NeuroMCMuxCounter(extra_ignore_modules=self.extra_ignore_modules)

        if self.memory_config.memory_model == "residency":
            mem_counter = NeuroMCMemoryResidencyCounter(
                memory_config=self.memory_config,
                extra_ignore_modules=self.extra_ignore_modules,
            )
            self._warnings.append(
                "Memory residency model is trace-driven and not cycle-accurate; "
                "NoC hop/bank scheduling effects are not modeled."
            )
        else:
            mem_counter = NeuroMCMemoryTrafficCounter(
                level_weights=self.memory_level_weights,
                extra_ignore_modules=self.extra_ignore_modules,
            )

        return {
            "mul": mul_counter,
            "add": add_counter,
            "cmp": cmp_counter,
            "sqrt": sqrt_counter,
            "mux": mux_counter,
            "memory": mem_counter,
        }

    def __enter__(self):
        self._trace_mode.__enter__()
        self._dispatch_mode.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        dispatch_ret = self._dispatch_mode.__exit__(exc_type, exc, tb)
        trace_ret = self._trace_mode.__exit__(exc_type, exc, tb)
        self._active = False
        return dispatch_ret or trace_ret

    @contextmanager
    def stage(self, name: str):
        r"""
        **API Language:**
        :ref:`中文 <NeuroMCEnergyProfiler.stage-cn>` | :ref:`English <NeuroMCEnergyProfiler.stage-en>`

        ----

        .. _NeuroMCEnergyProfiler.stage-cn:

        * **中文**

        标注一个统计阶段（例如 ``forward`` / ``backward`` / ``optimizer`` 或自定义阶段名）。
        该方法必须在 profiler 激活的上下文内调用，且目前不支持 stage 嵌套。

        :param name: 阶段名
        :type name: str

        ----

        .. _NeuroMCEnergyProfiler.stage-en:

        * **English**

        Mark one profiling stage (e.g., ``forward`` / ``backward`` /
        ``optimizer`` or any custom stage name).
        Must be used inside an active profiler context; nested stages are not
        supported in v1.

        :param name: stage name
        :type name: str
        """
        if not self._active:
            raise RuntimeError(
                "stage() can only be used inside active profiler context."
            )
        if self._stage_stack:
            raise RuntimeError("Nested stage() is not supported in v1.")

        before = self._snapshot_state()
        self._stage_stack.append(name)
        try:
            yield self
        finally:
            self._stage_stack.pop()
            after = self._snapshot_state()
            self._accumulate_stage_delta(name, before, after)

    def _add_warning(self, message: str):
        self._warnings.append(message)

    def _snapshot_state(self):
        mem_counter = self._counters["memory"]
        move_by_edge = {}
        move_by_op = {}
        if hasattr(mem_counter, "get_move_bits_by_edge"):
            move_by_edge = mem_counter.get_move_bits_by_edge()
            move_by_op = mem_counter.get_move_bits_by_op()

        return {
            "primitive": {
                "mul": self._counters["mul"].get_total(),
                "add": self._counters["add"].get_total(),
                "cmp": self._counters["cmp"].get_total(),
                "sqrt": self._counters["sqrt"].get_total(),
                "mux": self._counters["mux"].get_total(),
            },
            "memory_level_bits": mem_counter.get_level_bits(),
            "memory_level_rw_bits": mem_counter.get_level_rw_bits(),
            "memory_op_level_bits": mem_counter.get_op_level_bits(),
            "move_bits_by_edge": move_by_edge,
            "move_bits_by_op": move_by_op,
        }

    def _accumulate_stage_delta(
        self, stage_name: str, before: dict[str, Any], after: dict[str, Any]
    ):
        for primitive in ("mul", "add", "cmp", "sqrt", "mux"):
            delta = int(after["primitive"][primitive] - before["primitive"][primitive])
            if delta:
                self._stage_primitive_counts[stage_name][primitive] += delta

        level_delta = _diff_simple_dict(
            after["memory_level_bits"], before["memory_level_bits"]
        )
        _add_nested(self._stage_memory_level_bits[stage_name], level_delta)

        rw_delta = _diff_nested_dict(
            after["memory_level_rw_bits"], before["memory_level_rw_bits"]
        )
        for level, info in rw_delta.items():
            for rw, delta in info.items():
                self._stage_memory_level_rw_bits[stage_name][level][rw] += delta

        op_level_delta = _diff_nested_dict(
            after["memory_op_level_bits"], before["memory_op_level_bits"]
        )
        for op_name, level_info in op_level_delta.items():
            _add_nested(
                self._stage_memory_op_level_bits[stage_name][op_name], level_info
            )

        move_edge_delta = _diff_simple_dict(
            after["move_bits_by_edge"], before["move_bits_by_edge"]
        )
        _add_nested(self._stage_move_bits_by_edge[stage_name], move_edge_delta)

        move_op_delta = _diff_nested_dict(
            after["move_bits_by_op"], before["move_bits_by_op"]
        )
        for op_name, edge_info in move_op_delta.items():
            _add_nested(self._stage_move_bits_by_op[stage_name][op_name], edge_info)

    def _supported_ops(self) -> set[str]:
        supported_ops: set[str] = set()
        for counter in self._counter_list:
            for rule_key in counter.rules.keys():
                op_name = _rule_key_to_name(rule_key)
                if op_name is not None:
                    supported_ops.add(op_name)
        return supported_ops

    def get_report(self) -> NeuroMCRuntimeEnergyReport:
        r"""
        **API Language:**
        :ref:`中文 <NeuroMCEnergyProfiler.get_report-cn>` | :ref:`English <NeuroMCEnergyProfiler.get_report-en>`

        ----

        .. _NeuroMCEnergyProfiler.get_report-cn:

        * **中文**

        生成并返回完整能耗报告。
        报告包含：总能耗、计算/访存分项、按阶段统计、按算子统计、
        primitive 计数、访存 bit 统计以及 warnings。
        见 :class:`NeuroMCRuntimeEnergyReport` 字段说明。

        :return: NeuroMC runtime 能耗报告
        :rtype: NeuroMCRuntimeEnergyReport

        ----

        .. _NeuroMCEnergyProfiler.get_report-en:

        * **English**

        Build and return the full runtime energy report, including:
        total energy, compute/memory breakdown, stage-wise and op-wise energy,
        primitive counts, memory-bit statistics, and warnings.
        See the field descriptions in :class:`NeuroMCRuntimeEnergyReport` for details.

        :return: NeuroMC runtime energy report
        :rtype: NeuroMCRuntimeEnergyReport
        """
        primitive_totals = {
            "mul": self._counters["mul"].get_total(),
            "add": self._counters["add"].get_total(),
            "cmp": self._counters["cmp"].get_total(),
            "sqrt": self._counters["sqrt"].get_total(),
            "mux": self._counters["mux"].get_total(),
        }
        primitive_op = {
            "mul": self._counters["mul"].get_op_counts(),
            "add": self._counters["add"].get_op_counts(),
            "cmp": self._counters["cmp"].get_op_counts(),
            "sqrt": self._counters["sqrt"].get_op_counts(),
            "mux": self._counters["mux"].get_op_counts(),
        }

        mem_counter = self._counters["memory"]
        level_bits = mem_counter.get_level_bits()
        level_rw_bits = mem_counter.get_level_rw_bits()
        op_level_bits = mem_counter.get_op_level_bits()
        move_bits_by_edge = {}
        move_bits_by_op = {}
        if hasattr(mem_counter, "get_move_bits_by_edge"):
            move_bits_by_edge = mem_counter.get_move_bits_by_edge()
            move_bits_by_op = mem_counter.get_move_bits_by_op()

        primitive_stage: dict[str, dict[str, int]] = {
            "mul": defaultdict(int),
            "add": defaultdict(int),
            "cmp": defaultdict(int),
            "sqrt": defaultdict(int),
            "mux": defaultdict(int),
        }
        for stage_name, p_counts in self._stage_primitive_counts.items():
            for p_name, value in p_counts.items():
                primitive_stage[p_name][stage_name] += value

        stage_level_bits: dict[str, dict[str, int]] = {}
        for stage_name, level_info in self._stage_memory_level_bits.items():
            stage_level_bits[stage_name] = dict(level_info)

        primitive_labeled_totals = {
            p_name: int(sum(stage_dict.values()))
            for p_name, stage_dict in primitive_stage.items()
        }
        level_labeled_totals = defaultdict(int)
        for level_info in stage_level_bits.values():
            for level, bits in level_info.items():
                level_labeled_totals[level] += bits

        rw_by_stage = {
            stage: {level: dict(rw_info) for level, rw_info in level_info.items()}
            for stage, level_info in self._stage_memory_level_rw_bits.items()
        }
        by_stage_op = {
            stage: {op: dict(level_info) for op, level_info in op_info.items()}
            for stage, op_info in self._stage_memory_op_level_bits.items()
        }
        move_bits_by_stage = {
            stage: dict(edge_info)
            for stage, edge_info in self._stage_move_bits_by_edge.items()
        }
        move_bits_by_stage_op = {
            stage: {op: dict(edge_info) for op, edge_info in op_info.items()}
            for stage, op_info in self._stage_move_bits_by_op.items()
        }

        rw_labeled_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for level_info in rw_by_stage.values():
            for level, rw_info in level_info.items():
                for rw_name, bits in rw_info.items():
                    rw_labeled_totals[level][rw_name] += bits

        op_level_labeled_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for op_info in by_stage_op.values():
            for op_name, level_info in op_info.items():
                for level, bits in level_info.items():
                    op_level_labeled_totals[op_name][level] += bits

        move_edge_labeled_totals = defaultdict(int)
        for edge_info in move_bits_by_stage.values():
            for edge, bits in edge_info.items():
                move_edge_labeled_totals[edge] += bits

        move_op_labeled_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for op_info in move_bits_by_stage_op.values():
            for op_name, edge_info in op_info.items():
                for edge, bits in edge_info.items():
                    move_op_labeled_totals[op_name][edge] += bits

        unlabeled_primitive = {
            p_name: primitive_totals[p_name] - primitive_labeled_totals[p_name]
            for p_name in primitive_totals
        }
        unlabeled_level = {
            level: level_bits.get(level, 0) - int(level_labeled_totals.get(level, 0))
            for level in level_bits
        }
        unlabeled_rw = {
            level: {
                rw_name: level_rw_bits.get(level, {}).get(rw_name, 0)
                - int(rw_labeled_totals[level].get(rw_name, 0))
                for rw_name in level_rw_bits.get(level, {})
            }
            for level in level_rw_bits
        }
        unlabeled_op_level = {
            op_name: {
                level: op_level_bits.get(op_name, {}).get(level, 0)
                - int(op_level_labeled_totals[op_name].get(level, 0))
                for level in op_level_bits.get(op_name, {})
            }
            for op_name in op_level_bits
        }
        unlabeled_move_edge = {
            edge: move_bits_by_edge.get(edge, 0)
            - int(move_edge_labeled_totals.get(edge, 0))
            for edge in move_bits_by_edge
        }
        unlabeled_move_op = {
            op_name: {
                edge: move_bits_by_op.get(op_name, {}).get(edge, 0)
                - int(move_op_labeled_totals[op_name].get(edge, 0))
                for edge in move_bits_by_op.get(op_name, {})
            }
            for op_name in move_bits_by_op
        }

        if any(v != 0 for v in unlabeled_primitive.values()):
            for p_name, value in unlabeled_primitive.items():
                primitive_stage[p_name]["unlabeled"] += value

        if any(v != 0 for v in unlabeled_level.values()):
            stage_level_bits.setdefault("unlabeled", {})
            _add_nested(stage_level_bits["unlabeled"], unlabeled_level)

        if any(
            bits != 0
            for level_info in unlabeled_rw.values()
            for bits in level_info.values()
        ):
            rw_by_stage.setdefault("unlabeled", {})
            for level, rw_info in unlabeled_rw.items():
                rw_by_stage["unlabeled"].setdefault(level, {})
                _add_nested(rw_by_stage["unlabeled"][level], rw_info)

        if any(
            bits != 0
            for level_info in unlabeled_op_level.values()
            for bits in level_info.values()
        ):
            by_stage_op.setdefault("unlabeled", {})
            for op_name, level_info in unlabeled_op_level.items():
                by_stage_op["unlabeled"].setdefault(op_name, {})
                _add_nested(by_stage_op["unlabeled"][op_name], level_info)

        if any(v != 0 for v in unlabeled_move_edge.values()):
            move_bits_by_stage.setdefault("unlabeled", {})
            _add_nested(move_bits_by_stage["unlabeled"], unlabeled_move_edge)

        if any(
            bits != 0
            for edge_info in unlabeled_move_op.values()
            for bits in edge_info.values()
        ):
            move_bits_by_stage_op.setdefault("unlabeled", {})
            for op_name, edge_info in unlabeled_move_op.items():
                move_bits_by_stage_op["unlabeled"].setdefault(op_name, {})
                _add_nested(move_bits_by_stage_op["unlabeled"][op_name], edge_info)

        energy_compute_pj = (
            primitive_totals["mul"] * self.op_cost["mul"]
            + primitive_totals["add"] * self.op_cost["add"]
            + primitive_totals["cmp"] * self.op_cost["cmp"]
            + primitive_totals["sqrt"] * self.op_cost["sqrt"]
            + primitive_totals["mux"] * self.op_cost["mux"]
        )

        energy_memory_pj = 0.0
        for level, bits in level_bits.items():
            energy_memory_pj += bits * self.memory_config.energy_pj_per_bit.get(
                level, 0.0
            )

        energy_by_stage: dict[str, float] = {}
        all_stages = set()
        for p_name in primitive_stage:
            all_stages.update(primitive_stage[p_name].keys())
        all_stages.update(stage_level_bits.keys())

        for stage in all_stages:
            e = 0.0
            e += primitive_stage["mul"].get(stage, 0) * self.op_cost["mul"]
            e += primitive_stage["add"].get(stage, 0) * self.op_cost["add"]
            e += primitive_stage["cmp"].get(stage, 0) * self.op_cost["cmp"]
            e += primitive_stage["sqrt"].get(stage, 0) * self.op_cost["sqrt"]
            e += primitive_stage["mux"].get(stage, 0) * self.op_cost["mux"]
            for level, bits in stage_level_bits.get(stage, {}).items():
                e += bits * self.memory_config.energy_pj_per_bit.get(level, 0.0)
            energy_by_stage[stage] = e

        energy_by_op: dict[str, float] = {}
        all_ops = set()
        for p_name in primitive_op:
            all_ops.update(primitive_op[p_name].keys())
        all_ops.update(op_level_bits.keys())

        for op_name in all_ops:
            e = 0.0
            e += primitive_op["mul"].get(op_name, 0) * self.op_cost["mul"]
            e += primitive_op["add"].get(op_name, 0) * self.op_cost["add"]
            e += primitive_op["cmp"].get(op_name, 0) * self.op_cost["cmp"]
            e += primitive_op["sqrt"].get(op_name, 0) * self.op_cost["sqrt"]
            e += primitive_op["mux"].get(op_name, 0) * self.op_cost["mux"]
            for level, bits in op_level_bits.get(op_name, {}).items():
                e += bits * self.memory_config.energy_pj_per_bit.get(level, 0.0)
            energy_by_op[op_name] = e

        warnings = list(self._warnings)
        unsupported_ops = _filter_unsupported_ops(
            self._trace_mode.op_counts, self._supported_ops()
        )
        if unsupported_ops:
            warnings.append(
                "Unsupported aten ops use fallback=0 in compute primitives: "
                + ", ".join(unsupported_ops[:30])
            )
            if len(unsupported_ops) > 30:
                warnings.append(
                    f"... and {len(unsupported_ops) - 30} more unsupported ops."
                )

        energy_total_pj = energy_compute_pj + energy_memory_pj

        primitive_counts = {
            "totals": primitive_totals,
            "by_stage": {
                p_name: dict(stage_dict)
                for p_name, stage_dict in primitive_stage.items()
            },
            "by_op": primitive_op,
            "core_type": self.core_type,
        }

        memory_bits_by_level = {
            "memory_model": self.memory_config.memory_model,
            "totals": level_bits,
            "rw_totals": level_rw_bits,
            "by_stage": stage_level_bits,
            "rw_by_stage": rw_by_stage,
            "by_op": op_level_bits,
            "by_stage_op": by_stage_op,
            "move_bits_by_edge": move_bits_by_edge,
            "move_bits_by_stage": move_bits_by_stage,
            "move_bits_by_op": move_bits_by_op,
            "move_bits_by_stage_op": move_bits_by_stage_op,
        }

        return NeuroMCRuntimeEnergyReport(
            energy_total_pj=energy_total_pj,
            energy_compute_pj=energy_compute_pj,
            energy_memory_pj=energy_memory_pj,
            energy_by_stage=energy_by_stage,
            energy_by_op=energy_by_op,
            primitive_counts=primitive_counts,
            memory_bits_by_level=memory_bits_by_level,
            warnings=warnings,
        )

    def get_total(self) -> float:
        r"""
        **API Language:**
        :ref:`中文 <NeuroMCEnergyProfiler.get_total-cn>` | :ref:`English <NeuroMCEnergyProfiler.get_total-en>`

        ----

        .. _NeuroMCEnergyProfiler.get_total-cn:

        * **中文**

        返回总能耗（pJ）。等价于 ``self.get_report().energy_total_pj``。

        :return: 总能耗（pJ）
        :rtype: float

        ----

        .. _NeuroMCEnergyProfiler.get_total-en:

        * **English**

        Return total energy (pJ). Equivalent to
        ``self.get_report().energy_total_pj``.

        :return: total energy in pJ
        :rtype: float
        """
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
        r"""
        **API Language:**
        :ref:`中文 <NeuroMCEnergyProfiler.get_counts-cn>` | :ref:`English <NeuroMCEnergyProfiler.get_counts-en>`

        ----

        .. _NeuroMCEnergyProfiler.get_counts-cn:

        * **中文**

        返回计数结果字典，包含：

        - ``primitive_counts``
        - ``memory_bits_by_level``

        便于与现有 ``op_counter`` 计数接口风格对齐。

        :return: 计数字典
        :rtype: dict[str, Any]

        ----

        .. _NeuroMCEnergyProfiler.get_counts-en:

        * **English**

        Return count dictionaries containing:

        - ``primitive_counts``
        - ``memory_bits_by_level``

        This keeps alignment with existing ``op_counter`` count-style outputs.

        :return: count dictionary
        :rtype: dict[str, Any]
        """
        report = self.get_report()
        return {
            "primitive_counts": report.primitive_counts,
            "memory_bits_by_level": report.memory_bits_by_level,
        }


def estimate_neuromc_runtime_energy(
    model: nn.Module,
    inputs,
    *,
    target: torch.Tensor | None = None,
    loss_fn: Callable | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    core_type: str = "fp_soma",
    op_cost_pj: dict[str, float] | None = None,
    memory_cost_pj_per_bit: dict[str, float] | None = None,
    memory_level_weights: dict[str, float] | None = None,
    memory_model: str | None = None,
    memory_config: MemoryHierarchyConfig | None = None,
    strict: bool = False,
    verbose: bool = False,
    extra_ignore_modules: list[nn.Module] | None = None,
) -> NeuroMCRuntimeEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <estimate_neuromc_runtime_energy-cn>` | :ref:`English <estimate_neuromc_runtime_energy-en>`

    ----

    .. _estimate_neuromc_runtime_energy-cn:

    * **中文**

    NeuroMC runtime 便捷估计入口。
    该接口会执行一次真实运行流程并统计能耗：

    - ``forward`` 阶段：执行 ``model(inputs)``
    - 若提供 ``loss_fn`` 与 ``target``：执行 ``backward`` 阶段
    - 若同时提供 ``optimizer``：执行 ``optimizer`` 阶段（``step`` + ``zero_grad``）

    该函数内部基于 ``NeuroMCEnergyProfiler`` ，自动打阶段标签并生成报告。

    :param model: 待统计模型
    :type model: nn.Module

    :param inputs: 输入数据；若为 tuple/list 则按 ``model(*inputs)`` 调用
    :type inputs: Any

    :param target: 监督目标；当提供 ``loss_fn`` 时必填
    :type target: torch.Tensor | None

    :param loss_fn: 损失函数或可调用对象
    :type loss_fn: Callable | None

    :param optimizer: 优化器；仅当存在 ``loss_fn`` 时才会执行 ``step``
    :type optimizer: torch.optim.Optimizer | None

    :param core_type: 计算核心类型标识，会写入报告
    :type core_type: str

    :param op_cost_pj: primitive 单位能耗映射（pJ/op）
    :type op_cost_pj: dict[str, float] | None

    :param memory_cost_pj_per_bit: 覆盖 ``memory_config.energy_pj_per_bit`` 的单位 bit 成本
    :type memory_cost_pj_per_bit: dict[str, float] | None

    :param memory_level_weights: ``memory_model="weighted"`` 时的层级权重
    :type memory_level_weights: dict[str, float] | None

    :param memory_model: 可选覆盖 ``memory_config.memory_model``
    :type memory_model: str | None

    :param memory_config: 访存配置，默认使用 ``MemoryHierarchyConfig.neuromc_like_v1()``
    :type memory_config: MemoryHierarchyConfig | None

    :param strict: 透传给 ``DispatchCounterMode`` 的严格模式开关
    :type strict: bool

    :param verbose: 透传给 ``DispatchCounterMode`` 的详细日志开关
    :type verbose: bool

    :param extra_ignore_modules: 额外忽略模块
    :type extra_ignore_modules: list[nn.Module]

    :return: NeuroMC runtime 能耗报告
    :rtype: NeuroMCRuntimeEnergyReport

    ----

    .. _estimate_neuromc_runtime_energy-en:

    * **English**

    Convenience entry for NeuroMC runtime energy estimation.
    It runs one real execution flow and profiles energy:

    - ``forward`` stage: run ``model(inputs)``
    - if ``loss_fn`` and ``target`` are provided: run ``backward`` stage
    - if ``optimizer`` is also provided: run ``optimizer`` stage
      (``step`` + ``zero_grad``)

    Internally, this function uses ``NeuroMCEnergyProfiler`` with explicit
    stage tags and returns a report.

    :param model: model to profile
    :type model: nn.Module

    :param inputs: model input; tuple/list will be passed as ``model(*inputs)``
    :type inputs: Any

    :param target: supervision target; required when ``loss_fn`` is provided
    :type target: torch.Tensor | None

    :param loss_fn: loss function or callable
    :type loss_fn: Callable | None

    :param optimizer: optimizer; ``step`` is only executed when ``loss_fn`` exists
    :type optimizer: torch.optim.Optimizer | None

    :param core_type: compute-core label stored in report
    :type core_type: str

    :param op_cost_pj: per-primitive energy map (pJ/op)
    :type op_cost_pj: dict[str, float] | None

    :param memory_cost_pj_per_bit: per-bit cost override for ``memory_config.energy_pj_per_bit``
    :type memory_cost_pj_per_bit: dict[str, float] | None

    :param memory_level_weights: per-level weights for ``memory_model="weighted"``
    :type memory_level_weights: dict[str, float] | None

    :param memory_model: optional override for ``memory_config.memory_model``
    :type memory_model: str | None

    :param memory_config: memory hierarchy config; defaults to ``MemoryHierarchyConfig.neuromc_like_v1()``
    :type memory_config: MemoryHierarchyConfig | None

    :param strict: strict-mode flag forwarded to ``DispatchCounterMode``
    :type strict: bool

    :param verbose: verbose flag forwarded to ``DispatchCounterMode``
    :type verbose: bool

    :param extra_ignore_modules: extra modules to ignore
    :type extra_ignore_modules: list[nn.Module]

    :return: NeuroMC runtime energy report
    :rtype: NeuroMCRuntimeEnergyReport
    """

    cfg = (memory_config or MemoryHierarchyConfig.neuromc_like_v1()).copy()
    if memory_model is not None:
        cfg.memory_model = memory_model
    if memory_cost_pj_per_bit is not None:
        cfg.energy_pj_per_bit.update(memory_cost_pj_per_bit)
    cfg.validate()

    resolved_loss_fn = _resolve_loss_fn(loss_fn)
    _clear_existing_grads(model, optimizer)

    with NeuroMCEnergyProfiler(
        core_type=core_type,
        op_cost_pj=op_cost_pj,
        memory_config=cfg,
        memory_level_weights=memory_level_weights,
        strict=strict,
        verbose=verbose,
        extra_ignore_modules=extra_ignore_modules,
    ) as profiler:
        with profiler.stage("forward"):
            output = _call_model(model, inputs)
            loss = None
            if resolved_loss_fn is not None:
                if target is None:
                    raise ValueError("target is required when loss_fn is provided")
                loss = resolved_loss_fn(output, target)

        if resolved_loss_fn is not None:
            with profiler.stage("backward"):
                loss.backward()
            if optimizer is not None:
                with profiler.stage("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        elif optimizer is not None:
            profiler._add_warning(
                "optimizer is provided without loss_fn; optimizer.step() is skipped."
            )

        return profiler.get_report()
