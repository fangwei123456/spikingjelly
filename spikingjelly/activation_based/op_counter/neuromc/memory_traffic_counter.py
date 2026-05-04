from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name

from .base_counter import NeuroMCBaseCounter
from .utils import _infer_stage, _tensor_bits

aten = torch.ops.aten

__all__ = ["NeuroMCMemoryTrafficCounter"]


def _memory_binary(args, kwargs, out):
    x, y = args[:2]
    read_bits = _tensor_bits(x) + _tensor_bits(y)
    write_bits = _tensor_bits(out)
    return read_bits, write_bits


def _memory_unary(args, kwargs, out):
    x = args[0]
    return _tensor_bits(x), _tensor_bits(out)


def _memory_sum_or_mean(args, kwargs, out):
    x = args[0]
    return _tensor_bits(x), _tensor_bits(out)


def _memory_mm(args, kwargs, out):
    x, y = args[:2]
    return _tensor_bits(x) + _tensor_bits(y), _tensor_bits(out)


def _memory_addmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    read = _tensor_bits(x) + _tensor_bits(y)
    if beta != 0:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_bmm(args, kwargs, out):
    x, y = args[:2]
    return _tensor_bits(x) + _tensor_bits(y), _tensor_bits(out)


def _memory_baddbmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    read = _tensor_bits(x) + _tensor_bits(y)
    if beta != 0:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_convolution(args, kwargs, out):
    x, w, bias = args[:3]
    read = _tensor_bits(x) + _tensor_bits(w)
    if bias is not None:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_convolution_backward(args, kwargs, out):
    grad_out, x, w, _bias, *_rest, output_mask = args
    read = _tensor_bits(grad_out)
    write = 0
    if output_mask[0]:
        read += _tensor_bits(w)
        write += _tensor_bits(out[0])
    if output_mask[1]:
        read += _tensor_bits(x)
        write += _tensor_bits(out[1])
    if output_mask[2]:
        write += _tensor_bits(out[2])
    return read, write


def _memory_native_batch_norm(args, kwargs, out):
    x, gamma, beta, mean, var, train = args[:6]
    read = (
        _tensor_bits(x)
        + _tensor_bits(gamma)
        + _tensor_bits(beta)
        + _tensor_bits(mean)
        + _tensor_bits(var)
    )
    write = _tensor_bits(out[0])
    if train:
        write += _tensor_bits(out[1]) + _tensor_bits(out[2])
    return read, write


def _memory_native_batch_norm_backward(args, kwargs, out):
    read = sum(_tensor_bits(x) for x in args[:7])
    if isinstance(out, (tuple, list)):
        write = sum(_tensor_bits(x) for x in out if torch.is_tensor(x))
    else:
        write = _tensor_bits(out)
    return read, write


def _memory_where(args, kwargs, out):
    cond, x, y = args[:3]
    read = _tensor_bits(cond) + _tensor_bits(x) + _tensor_bits(y)
    write = _tensor_bits(out)
    return read, write


class NeuroMCMemoryTrafficCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        level_weights: dict[str, float] | None = None,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.mm.default: _memory_mm,
            aten.addmm.default: _memory_addmm,
            aten.bmm.default: _memory_bmm,
            aten.baddbmm.default: _memory_baddbmm,
            aten.convolution.default: _memory_convolution,
            aten.convolution_backward.default: _memory_convolution_backward,
            aten.native_batch_norm.default: _memory_native_batch_norm,
            aten.native_batch_norm_backward.default: _memory_native_batch_norm_backward,
            aten.add.Tensor: _memory_binary,
            aten.add_.Tensor: _memory_binary,
            aten.add.Scalar: _memory_binary,
            aten.add_.Scalar: _memory_binary,
            aten.sub.Tensor: _memory_binary,
            aten.sub_.Tensor: _memory_binary,
            aten.sub.Scalar: _memory_binary,
            aten.sub_.Scalar: _memory_binary,
            aten.rsub.Tensor: _memory_binary,
            aten.rsub.Scalar: _memory_binary,
            aten.mul.Tensor: _memory_binary,
            aten.mul_.Tensor: _memory_binary,
            aten.mul.Scalar: _memory_binary,
            aten.mul_.Scalar: _memory_binary,
            aten.div.Tensor: _memory_binary,
            aten.div_.Tensor: _memory_binary,
            aten.div.Scalar: _memory_binary,
            aten.div_.Scalar: _memory_binary,
            aten.sqrt.default: _memory_unary,
            aten.sqrt_.default: _memory_unary,
            aten.rsqrt.default: _memory_unary,
            aten.sum.default: _memory_sum_or_mean,
            aten.sum.dim_IntList: _memory_sum_or_mean,
            aten.mean.dim: _memory_sum_or_mean,
            aten.where.self: _memory_where,
            aten.where.ScalarOther: _memory_where,
            aten.where.ScalarSelf: _memory_where,
        }
        self.rules.update(extra_rules)
        self.level_weights = level_weights or {"reg": 2.0, "sram": 1.0, "dram": 0.25}
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
        read_bits, write_bits = self.rules[func](args, kwargs, out)
        total_bits = int(read_bits + write_bits)
        stage = _infer_stage(func, args, kwargs, out)
        self.stage_records[stage] += total_bits
        self.op_records[op_name] += total_bits
        self.stage_op_records[stage][op_name] += total_bits

        for level, weight in self.level_weights.items():
            scaled_read = int(round(read_bits * weight))
            scaled_write = int(round(write_bits * weight))
            scaled_total = scaled_read + scaled_write
            self.level_records[level] += scaled_total
            self.level_rw_records[level]["read_bits"] += scaled_read
            self.level_rw_records[level]["write_bits"] += scaled_write
            self.level_rw_records[level]["total_bits"] += scaled_total
            self.stage_level_records[stage][level] += scaled_total
            self.op_level_records[op_name][level] += scaled_total
        return total_bits

    def get_level_bits(self) -> dict[str, int]:
        return dict(self.level_records)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.level_rw_records.items()}

    def get_stage_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_level_records.items()}

    def get_op_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.op_level_records.items()}
