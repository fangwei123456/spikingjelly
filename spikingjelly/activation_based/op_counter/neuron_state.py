from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from ..neuron.base_node import BaseNode
from .base import BaseCounter

__all__ = ["NeuronStateCounter"]


_IGNORED_OP_PREFIXES = (
    "aten.detach",
    "aten.view",
    "aten.alias",
    "aten._unsafe_view",
    "aten.as_strided",
    "aten.expand",
    "aten.slice",
    "aten.select.int",
    "aten.select_copy",
    "aten.unsqueeze",
    "aten.squeeze",
    "aten.transpose",
    "aten.permute",
    "aten.clone",
    "aten.copy_",
    "aten.lift_fresh",
    "aten._to_copy",
)

_ADD_OPS = {
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.add.Scalar",
    "aten.add_.Scalar",
    "aten.sub.Tensor",
    "aten.sub_.Tensor",
    "aten.sub.Scalar",
    "aten.sub_.Scalar",
    "aten.rsub.Tensor",
    "aten.rsub.Scalar",
}

_MUL_OPS = {
    "aten.mul.Tensor",
    "aten.mul_.Tensor",
    "aten.mul.Scalar",
    "aten.mul_.Scalar",
    "aten.div.Tensor",
    "aten.div_.Tensor",
    "aten.div.Scalar",
    "aten.div_.Scalar",
}

_COMP_OPS = {
    "aten.eq.Tensor",
    "aten.eq.Scalar",
    "aten.ne.Tensor",
    "aten.ne.Scalar",
    "aten.lt.Tensor",
    "aten.lt.Scalar",
    "aten.le.Tensor",
    "aten.le.Scalar",
    "aten.gt.Tensor",
    "aten.gt.Scalar",
    "aten.ge.Tensor",
    "aten.ge.Scalar",
}

_NONLINEAR_OPS = {
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.rsqrt.default",
    "aten.sqrt.default",
    "aten.sqrt_.default",
    "aten.exp.default",
    "aten.exp_.default",
    "aten.tanh.default",
    "aten.tanh_.default",
}

_SELECT_OPS = {
    "aten.where.self",
    "aten.where.ScalarOther",
    "aten.where.ScalarSelf",
}


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _numel_tree(tree: Any) -> int:
    return sum(int(x.numel()) for x in _collect_tensors(tree))


def _is_binary_tensor(x: torch.Tensor) -> bool:
    if x.dtype == torch.bool:
        return True
    return bool(x.eq(0).logical_or_(x.eq(1)).all().item())


class NeuronStateCounter(BaseCounter):
    def __init__(
        self,
        *,
        strict: bool = False,
        extra_state_rules: dict[type[nn.Module], Callable] | None = None,
    ):
        super().__init__()
        self.strict = strict
        self.extra_state_rules = dict(extra_state_rules or {})
        self.metric_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.projection_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.warnings: list[str] = []
        self._warned_modules: set[int] = set()
        self._pending_metrics: dict[str, int] | None = None
        self._pending_projection: dict[str, int] | None = None

    def has_rule(self, func) -> bool:
        return True

    def _warn_or_raise(self, module: BaseNode, message: str):
        if self.strict:
            raise ValueError(message)
        module_id = id(module)
        if module_id in self._warned_modules:
            return
        self._warned_modules.add(module_id)
        self.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def count_with_context(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        *,
        active_modules: set[nn.Module],
        parent_names: set[str],
    ) -> int:
        del parent_names
        active_base_nodes = [m for m in active_modules if isinstance(m, BaseNode)]
        if not active_base_nodes:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        for module in active_base_nodes:
            if module.backend != "torch":
                self._warn_or_raise(
                    module,
                    f"NeuronStateCounter only supports torch backend, got "
                    f"{module.backend!r} from {module.__class__.__name__}.",
                )
                self._pending_metrics = None
                self._pending_projection = None
                return 0

        state_tensor_ids: set[int] = set()
        for module in active_base_nodes:
            for value in module._memories.values():
                if torch.is_tensor(value):
                    state_tensor_ids.add(id(value))

        if not state_tensor_ids:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        for module in active_base_nodes:
            rule = self.extra_state_rules.get(type(module))
            if rule is None:
                continue
            breakdown = rule(module, func, args, kwargs, out, state_tensor_ids)
            if breakdown is not None:
                return self._store_breakdown(breakdown)

        op_name = resolve_name(func)
        if op_name.startswith(_IGNORED_OP_PREFIXES):
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        tensors_in = _collect_tensors((args, kwargs))
        state_tensors = [x for x in tensors_in if id(x) in state_tensor_ids]
        if not state_tensors:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        output_tensors = _collect_tensors(out)
        out_numel = _numel_tree(out)
        metrics = {
            "state_reads": sum(int(x.numel()) for x in state_tensors),
            "state_writes": 0,
            "state_adds": 0,
            "state_muls": 0,
            "state_comps": 0,
            "state_nonlinear_ops": 0,
            "state_reset_ops": 0,
            "state_select_ops": 0,
            "spike_triggered_ops": 0,
            "timestep_dense_ops": 0,
        }

        writes_state = False
        if op_name in _ADD_OPS:
            metrics["state_adds"] += out_numel
            writes_state = True
        elif op_name in _MUL_OPS:
            metrics["state_muls"] += out_numel
            writes_state = True
        elif op_name in _COMP_OPS:
            metrics["state_comps"] += out_numel
        elif op_name in _NONLINEAR_OPS:
            metrics["state_nonlinear_ops"] += out_numel
        elif op_name in _SELECT_OPS:
            metrics["state_select_ops"] += out_numel
            writes_state = True

        if writes_state and output_tensors:
            metrics["state_writes"] += out_numel
            non_state_tensors = [x for x in tensors_in if id(x) not in state_tensor_ids]
            has_spike_gate = any(_is_binary_tensor(x) for x in non_state_tensors)
            if has_spike_gate:
                metrics["state_reset_ops"] += out_numel
                metrics["state_select_ops"] += out_numel
                metrics["spike_triggered_ops"] += out_numel
            else:
                metrics["timestep_dense_ops"] += out_numel

        if not any(metrics.values()):
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        return self._store_breakdown(metrics)

    def _store_breakdown(self, metrics: dict[str, int]) -> int:
        projection = {
            "read_potential": metrics.get("state_reads", 0),
            "write_potential": metrics.get("state_writes", 0),
            "state_mac_like": metrics.get("state_muls", 0),
            "state_acc_like": metrics.get("state_adds", 0)
            + metrics.get("state_comps", 0)
            + metrics.get("state_nonlinear_ops", 0)
            + metrics.get("state_reset_ops", 0)
            + metrics.get("state_select_ops", 0),
        }
        self._pending_metrics = metrics
        self._pending_projection = projection
        return sum(metrics.values())

    def record(self, scope, func, value):
        super().record(scope, func, value)
        if self._pending_metrics is not None:
            for key, item in self._pending_metrics.items():
                self.metric_records[scope][key] += int(item)
        if self._pending_projection is not None:
            for key, item in self._pending_projection.items():
                self.projection_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None
        self._pending_projection = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        return {scope: dict(items) for scope, items in self.metric_records.items()}

    def get_projection_counts(self) -> dict[str, dict[str, int]]:
        return {scope: dict(items) for scope, items in self.projection_records.items()}
