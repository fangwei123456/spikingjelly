from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from .base import BaseCounter

aten = torch.ops.aten

__all__ = [
    "MemoryResidencySimulator",
    "MemoryResidencyCounter",
    "_access_convolution_backward",
]


_DEFAULT_CAPACITY_BITS = {
    "reg": float(256 * 1024 * 8),
    "sram": float(38 * 1024 * 1024 * 8),
    "dram": float("inf"),
}

_OPTIMIZER_HINTS = (
    "add_.",
    "sub_.",
    "mul_.",
    "div_.",
    "addcmul",
    "addcdiv",
    "lerp_",
    "copy_",
)


def _tensor_bits(x: Any) -> int:
    if not torch.is_tensor(x):
        return 0
    return int(x.numel() * x.element_size() * 8)


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _infer_stage(func, args, kwargs, out) -> str:
    op_name = resolve_name(func)
    if "backward" in op_name:
        return "backward"

    if torch.is_grad_enabled():
        return "forward"

    tensors = _collect_tensors((args, kwargs, out))
    has_grad_tensor = any(t.requires_grad for t in tensors)
    is_optimizer_like = any(hint in op_name for hint in _OPTIMIZER_HINTS)
    if has_grad_tensor and is_optimizer_like:
        return "optimizer"

    return "forward"


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
    reads = [x]
    for tensor in (gamma, beta, mean, var):
        if torch.is_tensor(tensor):
            reads.append(tensor)
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
    return [args[0]], [out]


def _access_sum_mean(args, kwargs, out):
    return [args[0]], [out]


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


class MemoryResidencySimulator:
    def __init__(
        self,
        config: Any | None = None,
        *,
        capacity_bits: dict[str, float] | None = None,
    ):
        if capacity_bits is None:
            if config is not None and hasattr(config, "capacity_bits"):
                capacity_bits = getattr(config, "capacity_bits")
            else:
                capacity_bits = dict(_DEFAULT_CAPACITY_BITS)
        self.capacity_bits = {
            "reg": float(capacity_bits.get("reg", float("inf"))),
            "sram": float(capacity_bits.get("sram", float("inf"))),
            "dram": float(capacity_bits.get("dram", float("inf"))),
        }
        self.reg_cache: OrderedDict[str, int] = OrderedDict()
        self.sram_cache: OrderedDict[str, int] = OrderedDict()
        self.usage_bits = {"reg": 0, "sram": 0}
        self.level_rw_bits: dict[str, dict[str, int]] = defaultdict(
            lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
        )
        self.op_level_rw_bits: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
            )
        )
        self.move_bits_by_edge: dict[str, int] = defaultdict(int)
        self.move_bits_by_op: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def _record_level_rw(self, level: str, rw: str, bits: int, op_name: str):
        if bits <= 0:
            return
        self.level_rw_bits[level][rw] += bits
        self.level_rw_bits[level]["total_bits"] += bits
        self.op_level_rw_bits[op_name][level][rw] += bits
        self.op_level_rw_bits[op_name][level]["total_bits"] += bits

    def _record_move(self, src: str, dst: str, bits: int, op_name: str):
        if bits <= 0:
            return
        edge = f"{src}->{dst}"
        self.move_bits_by_edge[edge] += bits
        self.move_bits_by_op[op_name][edge] += bits
        self._record_level_rw(src, "read_bits", bits, op_name)
        self._record_level_rw(dst, "write_bits", bits, op_name)

    def _touch(self, cache: OrderedDict[str, int], key: str):
        cache.move_to_end(key, last=True)

    def _touch_inclusive_levels(self, key: str):
        if key in self.reg_cache:
            self._touch(self.reg_cache, key)
        if key in self.sram_cache:
            self._touch(self.sram_cache, key)

    def _insert_sram(self, key: str, bits: int, op_name: str):
        if bits > self.capacity_bits["sram"]:
            return False
        if key in self.sram_cache:
            old = self.sram_cache[key]
            if bits > old:
                self.usage_bits["sram"] += bits - old
                self.sram_cache[key] = bits
            self._touch(self.sram_cache, key)
            return True

        while self.sram_cache and (
            self.usage_bits["sram"] + bits > self.capacity_bits["sram"]
        ):
            evict_key, evict_bits = self.sram_cache.popitem(last=False)
            self.usage_bits["sram"] -= evict_bits
            self._record_move("sram", "dram", evict_bits, op_name)
            reg_bits = self.reg_cache.pop(evict_key, None)
            if reg_bits is not None:
                self.usage_bits["reg"] -= reg_bits

        if self.usage_bits["sram"] + bits > self.capacity_bits["sram"]:
            return False

        self.sram_cache[key] = bits
        self.usage_bits["sram"] += bits
        return True

    def _insert_reg(self, key: str, bits: int, op_name: str):
        if bits > self.capacity_bits["reg"]:
            return False
        if key in self.reg_cache:
            old = self.reg_cache[key]
            if bits > old:
                self.usage_bits["reg"] += bits - old
                self.reg_cache[key] = bits
            self._touch(self.reg_cache, key)
            return True

        while self.reg_cache and (
            self.usage_bits["reg"] + bits > self.capacity_bits["reg"]
        ):
            evict_key, evict_bits = self.reg_cache.popitem(last=False)
            self.usage_bits["reg"] -= evict_bits
            inserted = self._insert_sram(evict_key, evict_bits, op_name)
            if inserted:
                self._record_move("reg", "sram", evict_bits, op_name)
            else:
                self._record_move("reg", "dram", evict_bits, op_name)

        if self.usage_bits["reg"] + bits > self.capacity_bits["reg"]:
            return False

        self.reg_cache[key] = bits
        self.usage_bits["reg"] += bits
        return True

    def _tensor_key(self, tensor: torch.Tensor) -> str:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        if storage_ptr == 0:
            storage_id = f"id={id(storage)}"
        else:
            storage_id = f"ptr={storage_ptr}"
        return f"{tensor.device}:{tensor.dtype}:{storage_id}"

    def reset(self):
        self.reg_cache.clear()
        self.sram_cache.clear()
        self.usage_bits = {"reg": 0, "sram": 0}
        self.level_rw_bits.clear()
        self.op_level_rw_bits.clear()
        self.move_bits_by_edge.clear()
        self.move_bits_by_op.clear()

    def on_tensor_read(self, tensor: torch.Tensor, op_name: str):
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        self._record_level_rw("dram", "read_bits", bits, op_name)
        sram_ok = self._insert_sram(key, bits, op_name)
        if sram_ok:
            self._record_move("dram", "sram", bits, op_name)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_move("dram", "reg", bits, op_name)
            self._record_level_rw("reg", "read_bits", bits, op_name)

    def on_tensor_write(self, tensor: torch.Tensor, op_name: str):
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "write_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_sram(key, bits, op_name):
            self._record_level_rw("sram", "write_bits", bits, op_name)
            return

        self._record_level_rw("dram", "write_bits", bits, op_name)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.level_rw_bits.items()}

    def get_op_level_rw_bits(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            op: {lv: dict(info) for lv, info in d.items()}
            for op, d in self.op_level_rw_bits.items()
        }

    def get_move_bits_by_edge(self) -> dict[str, int]:
        return dict(self.move_bits_by_edge)

    def get_move_bits_by_op(self) -> dict[str, dict[str, int]]:
        return {op: dict(d) for op, d in self.move_bits_by_op.items()}


class MemoryResidencyCounter(BaseCounter):
    def __init__(
        self,
        *,
        config: Any | None = None,
        capacity_bits: dict[str, float] | None = None,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        super().__init__()
        self.rules = dict(_RESIDENCY_ACCESS_RULES)
        if extra_rules is not None:
            self.rules.update(extra_rules)
        self.ignore_modules = list(extra_ignore_modules or [])
        self.simulator = MemoryResidencySimulator(config, capacity_bits=capacity_bits)
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
        self.stage_records: dict[str, int] = defaultdict(int)
        self.op_records: dict[str, int] = defaultdict(int)
        self.stage_op_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def reset(self):
        self.records.clear()
        self.level_records.clear()
        self.level_rw_records.clear()
        self.stage_level_records.clear()
        self.op_level_records.clear()
        self.stage_records.clear()
        self.op_records.clear()
        self.stage_op_records.clear()
        self.simulator.reset()

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        rule = self.rules.get(func)
        if rule is None:
            return 0
        op_name = resolve_name(func)
        stage = _infer_stage(func, args, kwargs, out)
        before_level_rw = self.simulator.get_level_rw_bits()
        read_tensors, write_tensors = rule(args, kwargs, out)

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
            delta_read = int(new_info.get("read_bits", 0) - old_info.get("read_bits", 0))
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
