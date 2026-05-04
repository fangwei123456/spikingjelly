from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from .base import DispatchCounterMode
from .neuromc_counters import (
    NeuroMCAddCounter,
    NeuroMCCmpCounter,
    NeuroMCMemoryTrafficCounter,
    NeuroMCMulCounter,
    NeuroMCMuxCounter,
    NeuroMCSqrtCounter,
)


__all__ = ["NeuroMCRuntimeEnergyReport", "estimate_neuromc_runtime_energy"]


_DEFAULT_OP_COST_PJ = {
    "mul": 0.812,
    "add": 0.548,
    "cmp": 0.056,
    "sqrt": 0.514,
    "mux": 0.548 * (1.0 / 16.0),
}

_DEFAULT_MEMORY_COST_PJ_PER_BIT = {
    "reg": 0.00901,
    "sram": 92.68282828 / 256.0,
    "dram": 1300.0 / 64.0,
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


def _filter_unsupported_ops(op_counts: dict[str, int], supported_ops: set[str]) -> list[str]:
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


def _create_counters(extra_ignore_modules, memory_level_weights):
    mul_counter = NeuroMCMulCounter(extra_ignore_modules=extra_ignore_modules)
    add_counter = NeuroMCAddCounter(extra_ignore_modules=extra_ignore_modules)
    cmp_counter = NeuroMCCmpCounter(extra_ignore_modules=extra_ignore_modules)
    sqrt_counter = NeuroMCSqrtCounter(extra_ignore_modules=extra_ignore_modules)
    mux_counter = NeuroMCMuxCounter(extra_ignore_modules=extra_ignore_modules)
    mem_counter = NeuroMCMemoryTrafficCounter(
        level_weights=memory_level_weights, extra_ignore_modules=extra_ignore_modules
    )
    counters = {
        "mul": mul_counter,
        "add": add_counter,
        "cmp": cmp_counter,
        "sqrt": sqrt_counter,
        "mux": mux_counter,
        "memory": mem_counter,
    }
    return counters


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
    strict: bool = False,
    verbose: bool = False,
    extra_ignore_modules: list[nn.Module] = [],
) -> NeuroMCRuntimeEnergyReport:
    """
    Estimate runtime energy from one actual model execution trace.

    The estimator captures real aten calls during execution and computes operation counts
    per primitive (mul/add/cmp/sqrt/mux), then maps them to energy with NeuroMC-like
    coefficients. It can optionally run backward and optimizer step if ``loss_fn`` and
    ``optimizer`` are provided.
    """

    op_cost = dict(_DEFAULT_OP_COST_PJ)
    if op_cost_pj is not None:
        op_cost.update(op_cost_pj)

    memory_cost = dict(_DEFAULT_MEMORY_COST_PJ_PER_BIT)
    if memory_cost_pj_per_bit is not None:
        memory_cost.update(memory_cost_pj_per_bit)

    warnings: list[str] = []
    primitive_totals = {"mul": 0, "add": 0, "cmp": 0, "sqrt": 0, "mux": 0}
    primitive_stage = {
        "mul": {},
        "add": {},
        "cmp": {},
        "sqrt": {},
        "mux": {},
    }
    primitive_op = {
        "mul": {},
        "add": {},
        "cmp": {},
        "sqrt": {},
        "mux": {},
    }
    level_bits: dict[str, int] = {}
    level_rw_bits: dict[str, dict[str, int]] = {}
    stage_level_bits: dict[str, dict[str, int]] = {}
    op_level_bits: dict[str, dict[str, int]] = {}

    def consume_stage(stage_name: str, counter_pack: dict[str, Any]):
        for p in ("mul", "add", "cmp", "sqrt", "mux"):
            c = counter_pack[p]
            p_total = c.get_total()
            primitive_totals[p] += p_total
            primitive_stage[p][stage_name] = primitive_stage[p].get(stage_name, 0) + p_total
            _add_nested(primitive_op[p], c.get_op_counts())

        mem = counter_pack["memory"]
        mem_level_bits = mem.get_level_bits()
        mem_level_rw_bits = mem.get_level_rw_bits()
        mem_op_level_bits = mem.get_op_level_bits()

        if stage_name not in stage_level_bits:
            stage_level_bits[stage_name] = {}
        _add_nested(stage_level_bits[stage_name], mem_level_bits)
        _add_nested(level_bits, mem_level_bits)

        for level, rw_info in mem_level_rw_bits.items():
            if level not in level_rw_bits:
                level_rw_bits[level] = {"read_bits": 0, "write_bits": 0, "total_bits": 0}
            for k, v in rw_info.items():
                level_rw_bits[level][k] = level_rw_bits[level].get(k, 0) + v

        for op_name, lv in mem_op_level_bits.items():
            if op_name not in op_level_bits:
                op_level_bits[op_name] = {}
            _add_nested(op_level_bits[op_name], lv)

    def run_phase(phase_name: str, fn: Callable[[], Any]):
        counter_pack = _create_counters(extra_ignore_modules, memory_level_weights)
        counter_list = [
            counter_pack["mul"],
            counter_pack["add"],
            counter_pack["cmp"],
            counter_pack["sqrt"],
            counter_pack["mux"],
            counter_pack["memory"],
        ]
        with DispatchCounterMode(counter_list, strict=strict, verbose=verbose):
            result = fn()
        consume_stage(phase_name, counter_pack)
        return result

    trace_mode = _AtenTraceMode()
    resolved_loss_fn = _resolve_loss_fn(loss_fn)
    loss = None

    with trace_mode:
        output = run_phase("forward", lambda: _call_model(model, inputs))
        if resolved_loss_fn is not None:
            if target is None:
                raise ValueError("target is required when loss_fn is provided")
            loss = resolved_loss_fn(output, target)
            run_phase("backward", lambda: loss.backward())
            if optimizer is not None:
                run_phase(
                    "optimizer",
                    lambda: (optimizer.step(), optimizer.zero_grad(set_to_none=True)),
                )
        elif optimizer is not None:
            warnings.append(
                "optimizer is provided without loss_fn; optimizer.step() is skipped."
            )

    energy_compute_pj = (
        primitive_totals["mul"] * op_cost["mul"]
        + primitive_totals["add"] * op_cost["add"]
        + primitive_totals["cmp"] * op_cost["cmp"]
        + primitive_totals["sqrt"] * op_cost["sqrt"]
        + primitive_totals["mux"] * op_cost["mux"]
    )

    energy_memory_pj = 0.0
    for level, bits in level_bits.items():
        energy_memory_pj += bits * memory_cost.get(level, 0.0)

    energy_by_stage: dict[str, float] = {}
    all_stages = set()
    for p_name in primitive_stage:
        all_stages.update(primitive_stage[p_name].keys())
    all_stages.update(stage_level_bits.keys())

    for stage in all_stages:
        e = 0.0
        e += primitive_stage["mul"].get(stage, 0) * op_cost["mul"]
        e += primitive_stage["add"].get(stage, 0) * op_cost["add"]
        e += primitive_stage["cmp"].get(stage, 0) * op_cost["cmp"]
        e += primitive_stage["sqrt"].get(stage, 0) * op_cost["sqrt"]
        e += primitive_stage["mux"].get(stage, 0) * op_cost["mux"]
        for level, bits in stage_level_bits.get(stage, {}).items():
            e += bits * memory_cost.get(level, 0.0)
        energy_by_stage[stage] = e

    energy_by_op: dict[str, float] = {}
    all_ops = set()
    for p_name in primitive_op:
        all_ops.update(primitive_op[p_name].keys())
    all_ops.update(op_level_bits.keys())
    for op_name in all_ops:
        e = 0.0
        e += primitive_op["mul"].get(op_name, 0) * op_cost["mul"]
        e += primitive_op["add"].get(op_name, 0) * op_cost["add"]
        e += primitive_op["cmp"].get(op_name, 0) * op_cost["cmp"]
        e += primitive_op["sqrt"].get(op_name, 0) * op_cost["sqrt"]
        e += primitive_op["mux"].get(op_name, 0) * op_cost["mux"]
        for level, bits in op_level_bits.get(op_name, {}).items():
            e += bits * memory_cost.get(level, 0.0)
        energy_by_op[op_name] = e

    supported_ops: set[str] = set()
    for counter in _create_counters(extra_ignore_modules, memory_level_weights).values():
        for rule_key in counter.rules.keys():
            op_name = _rule_key_to_name(rule_key)
            if op_name is not None:
                supported_ops.add(op_name)
    unsupported_ops = _filter_unsupported_ops(trace_mode.op_counts, supported_ops)
    if unsupported_ops:
        warnings.append(
            "Unsupported aten ops use fallback=0 in compute primitives: "
            + ", ".join(unsupported_ops[:30])
        )
        if len(unsupported_ops) > 30:
            warnings.append(f"... and {len(unsupported_ops) - 30} more unsupported ops.")

    energy_total_pj = energy_compute_pj + energy_memory_pj

    primitive_counts = {
        "totals": primitive_totals,
        "by_stage": primitive_stage,
        "by_op": primitive_op,
        "core_type": core_type,
    }
    memory_bits_by_level = {
        "totals": level_bits,
        "rw_totals": level_rw_bits,
        "by_stage": stage_level_bits,
        "by_op": op_level_bits,
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
