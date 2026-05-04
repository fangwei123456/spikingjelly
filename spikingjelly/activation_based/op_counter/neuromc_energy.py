from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from .base import DispatchCounterMode
from .neuromc_counters import (
    MemoryHierarchyConfig,
    NeuroMCAddCounter,
    NeuroMCCmpCounter,
    NeuroMCMemoryResidencyCounter,
    NeuroMCMemoryTrafficCounter,
    NeuroMCMulCounter,
    NeuroMCMuxCounter,
    NeuroMCSqrtCounter,
)

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


def _add_nested(dst: dict[str, int], src: dict[str, int]):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _rule_key_to_name(rule_key: Any) -> str | None:
    if isinstance(rule_key, str):
        return rule_key
    try:
        return resolve_name(rule_key)
    except Exception:
        return None


def _diff_simple_dict(new: dict[str, int], old: dict[str, int]) -> dict[str, int]:
    keys = set(new.keys()) | set(old.keys())
    out: dict[str, int] = {}
    for k in keys:
        delta = int(new.get(k, 0) - old.get(k, 0))
        if delta != 0:
            out[k] = delta
    return out


def _diff_nested_dict(
    new: dict[str, dict[str, int]], old: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    keys = set(new.keys()) | set(old.keys())
    out: dict[str, dict[str, int]] = {}
    for k in keys:
        delta = _diff_simple_dict(new.get(k, {}), old.get(k, {}))
        if delta:
            out[k] = delta
    return out


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
        extra_ignore_modules: list[nn.Module] = [],
    ):
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

    def add_warning(self, message: str):
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

        unlabeled_primitive = {
            p_name: primitive_totals[p_name] - primitive_labeled_totals[p_name]
            for p_name in primitive_totals
        }
        unlabeled_level = {
            level: level_bits.get(level, 0) - int(level_labeled_totals.get(level, 0))
            for level in level_bits
        }

        if any(v != 0 for v in unlabeled_primitive.values()):
            for p_name, value in unlabeled_primitive.items():
                primitive_stage[p_name]["unlabeled"] += value

        if any(v != 0 for v in unlabeled_level.values()):
            stage_level_bits.setdefault("unlabeled", {})
            _add_nested(stage_level_bits["unlabeled"], unlabeled_level)

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
            "rw_by_stage": {
                stage: {level: dict(rw_info) for level, rw_info in level_info.items()}
                for stage, level_info in self._stage_memory_level_rw_bits.items()
            },
            "by_op": op_level_bits,
            "by_stage_op": {
                stage: {op: dict(level_info) for op, level_info in op_info.items()}
                for stage, op_info in self._stage_memory_op_level_bits.items()
            },
            "move_bits_by_edge": move_bits_by_edge,
            "move_bits_by_stage": {
                stage: dict(edge_info)
                for stage, edge_info in self._stage_move_bits_by_edge.items()
            },
            "move_bits_by_op": move_bits_by_op,
            "move_bits_by_stage_op": {
                stage: {op: dict(edge_info) for op, edge_info in op_info.items()}
                for stage, op_info in self._stage_move_bits_by_op.items()
            },
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
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
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
    extra_ignore_modules: list[nn.Module] = [],
) -> NeuroMCRuntimeEnergyReport:
    """
    Estimate runtime energy from one actual model execution trace.

    The convenience API runs a standard forward/backward/optimizer workflow and
    internally uses :class:`NeuroMCRuntimeEnergyCounter` with explicit stage tags.
    """

    cfg = (memory_config or MemoryHierarchyConfig.neuromc_like_v1()).copy()
    if memory_model is not None:
        cfg.memory_model = memory_model
    if memory_cost_pj_per_bit is not None:
        cfg.energy_pj_per_bit.update(memory_cost_pj_per_bit)
    cfg.validate()

    resolved_loss_fn = _resolve_loss_fn(loss_fn)

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
            profiler.add_warning(
                "optimizer is provided without loss_fn; optimizer.step() is skipped."
            )

        return profiler.get_report()
