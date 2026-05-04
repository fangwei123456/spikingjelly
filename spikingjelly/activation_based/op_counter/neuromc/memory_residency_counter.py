from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from .base_counter import NeuroMCBaseCounter
from .config import MemoryHierarchyConfig
from .residency import MemoryResidencySimulator
from .utils import _infer_stage, _tensor_bits

aten = torch.ops.aten

__all__ = [
    "NeuroMCMemoryResidencyCounter",
    "_access_convolution_backward",
]


def _extract_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _access_mm(args, kwargs, out):
    x, y = args[:2]
    return [x, y], [out]


def _access_addmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    reads = [x, y]
    if beta != 0:
        reads.append(bias)
    return reads, [out]


def _access_bmm(args, kwargs, out):
    x, y = args[:2]
    return [x, y], [out]


def _access_baddbmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    reads = [x, y]
    if beta != 0:
        reads.append(bias)
    return reads, [out]


def _access_convolution(args, kwargs, out):
    x, w, bias = args[:3]
    reads = [x, w]
    if bias is not None:
        reads.append(bias)
    return reads, [out]


def _access_convolution_backward(args, kwargs, out):
    grad_out, x, w, _bias, *_rest, output_mask = args
    reads = [grad_out]
    writes: list[torch.Tensor] = []

    if not isinstance(out, (tuple, list)) or len(out) < 3:
        return reads, _extract_tensors(out)

    if output_mask[0]:
        reads.append(w)
        if torch.is_tensor(out[0]):
            writes.append(out[0])
    if output_mask[1]:
        reads.append(x)
        if torch.is_tensor(out[1]):
            writes.append(out[1])
    if output_mask[2]:
        if torch.is_tensor(out[2]):
            writes.append(out[2])
    return reads, writes


def _access_native_batch_norm(args, kwargs, out):
    x, gamma, beta, mean, var, train = args[:6]
    reads = [x, gamma, beta, mean, var]
    writes = [out[0]]
    if train:
        writes.extend([out[1], out[2]])
    return reads, writes


def _access_native_batch_norm_backward(args, kwargs, out):
    reads = [x for x in args[:7] if torch.is_tensor(x)]
    writes = _extract_tensors(out)
    return reads, writes


def _access_binary(args, kwargs, out):
    reads = [x for x in args[:2] if torch.is_tensor(x)]
    return reads, [out]


def _access_unary(args, kwargs, out):
    x = args[0]
    return [x], [out]


def _access_sum_mean(args, kwargs, out):
    x = args[0]
    return [x], [out]


def _access_where(args, kwargs, out):
    cond, x, y = args[:3]
    return [cond, x, y], [out]


_RESIDENCY_ACCESS_RULES: dict[Any, Callable] = {
    aten.mm.default: _access_mm,
    aten.addmm.default: _access_addmm,
    aten.bmm.default: _access_bmm,
    aten.baddbmm.default: _access_baddbmm,
    aten.convolution.default: _access_convolution,
    aten.convolution_backward.default: _access_convolution_backward,
    aten.native_batch_norm.default: _access_native_batch_norm,
    aten.native_batch_norm_backward.default: _access_native_batch_norm_backward,
    aten.add.Tensor: _access_binary,
    aten.add_.Tensor: _access_binary,
    aten.add.Scalar: _access_binary,
    aten.add_.Scalar: _access_binary,
    aten.sub.Tensor: _access_binary,
    aten.sub_.Tensor: _access_binary,
    aten.sub.Scalar: _access_binary,
    aten.sub_.Scalar: _access_binary,
    aten.rsub.Tensor: _access_binary,
    aten.rsub.Scalar: _access_binary,
    aten.mul.Tensor: _access_binary,
    aten.mul_.Tensor: _access_binary,
    aten.mul.Scalar: _access_binary,
    aten.mul_.Scalar: _access_binary,
    aten.div.Tensor: _access_binary,
    aten.div_.Tensor: _access_binary,
    aten.div.Scalar: _access_binary,
    aten.div_.Scalar: _access_binary,
    aten.sqrt.default: _access_unary,
    aten.sqrt_.default: _access_unary,
    aten.rsqrt.default: _access_unary,
    aten.sum.default: _access_sum_mean,
    aten.sum.dim_IntList: _access_sum_mean,
    aten.mean.dim: _access_sum_mean,
    aten.where.self: _access_where,
    aten.where.ScalarOther: _access_where,
    aten.where.ScalarSelf: _access_where,
}


class NeuroMCMemoryResidencyCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        memory_config: MemoryHierarchyConfig | None = None,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = dict(_RESIDENCY_ACCESS_RULES)
        self.rules.update(extra_rules)
        if memory_config is None:
            memory_config = MemoryHierarchyConfig.neuromc_like_v1(
                memory_model="residency"
            )
        self.simulator = MemoryResidencySimulator(memory_config)
        self.level_records: dict[str, int] = defaultdict(int)
        self.level_rw_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.stage_level_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.op_level_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        op_name = resolve_name(func)
        stage = _infer_stage(func, args, kwargs, out)
        before_level_rw = self.simulator.get_level_rw_bits()
        read_tensors, write_tensors = self.rules[func](args, kwargs, out)

        total_bits = 0
        for tensor in read_tensors:
            bits = _tensor_bits(tensor)
            total_bits += bits
            self.simulator.on_tensor_read(tensor, op_name)
        for tensor in write_tensors:
            bits = _tensor_bits(tensor)
            total_bits += bits
            self.simulator.on_tensor_write(tensor, op_name)

        after_level_rw = self.simulator.get_level_rw_bits()
        all_levels = set(before_level_rw.keys()) | set(after_level_rw.keys())
        for level in all_levels:
            old_info = before_level_rw.get(level, {})
            new_info = after_level_rw.get(level, {})
            delta_read = int(
                new_info.get("read_bits", 0) - old_info.get("read_bits", 0)
            )
            delta_write = int(
                new_info.get("write_bits", 0) - old_info.get("write_bits", 0)
            )
            delta_total = int(
                new_info.get("total_bits", 0) - old_info.get("total_bits", 0)
            )
            if delta_total == 0 and delta_read == 0 and delta_write == 0:
                continue
            self.level_records[level] += delta_total
            self.level_rw_records[level]["read_bits"] += delta_read
            self.level_rw_records[level]["write_bits"] += delta_write
            self.level_rw_records[level]["total_bits"] += delta_total
            self.stage_level_records[stage][level] += delta_total
            self.op_level_records[op_name][level] += delta_total

        self.stage_records[stage] += total_bits
        self.op_records[op_name] += total_bits
        self.stage_op_records[stage][op_name] += total_bits
        return total_bits

    def get_level_bits(self) -> dict[str, int]:
        return dict(self.level_records)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.level_rw_records.items()}

    def get_op_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.op_level_records.items()}

    def get_stage_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_level_records.items()}

    def get_move_bits_by_edge(self) -> dict[str, int]:
        return self.simulator.get_move_bits_by_edge()

    def get_move_bits_by_op(self) -> dict[str, dict[str, int]]:
        return self.simulator.get_move_bits_by_op()
