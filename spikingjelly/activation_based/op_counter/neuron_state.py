from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from ..neuron.base_node import BaseNode
from .base import BaseCounter, is_binary_tensor

__all__ = ["NeuronStateCounter"]

aten = torch.ops.aten

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

_IGNORED_OPS = {
    aten.detach.default,
    aten.alias.default,
    aten._unsafe_view.default,
    aten.as_strided.default,
    aten.expand.default,
    aten.clone.default,
    aten.copy_.default,
    aten.lift_fresh.default,
    aten._to_copy.default,
}

_ADD_OPS = {
    aten.add.Tensor,
    aten.add_.Tensor,
    aten.add.Scalar,
    aten.add_.Scalar,
    aten.sub.Tensor,
    aten.sub_.Tensor,
    aten.sub.Scalar,
    aten.sub_.Scalar,
    aten.rsub.Tensor,
    aten.rsub.Scalar,
}

_MUL_OPS = {
    aten.mul.Tensor,
    aten.mul_.Tensor,
    aten.mul.Scalar,
    aten.mul_.Scalar,
    aten.div.Tensor,
    aten.div_.Tensor,
    aten.div.Scalar,
    aten.div_.Scalar,
}

_COMP_OPS = {
    aten.eq.Tensor,
    aten.eq.Scalar,
    aten.ne.Tensor,
    aten.ne.Scalar,
    aten.lt.Tensor,
    aten.lt.Scalar,
    aten.le.Tensor,
    aten.le.Scalar,
    aten.gt.Tensor,
    aten.gt.Scalar,
    aten.ge.Tensor,
    aten.ge.Scalar,
}

_NONLINEAR_OPS = {
    aten.sigmoid.default,
    aten.sigmoid_.default,
    aten.rsqrt.default,
    aten.sqrt.default,
    aten.sqrt_.default,
    aten.exp.default,
    aten.exp_.default,
    aten.tanh.default,
    aten.tanh_.default,
}

_INPLACE_NONLINEAR_OPS = {
    aten.sigmoid_.default,
    aten.sqrt_.default,
    aten.exp_.default,
    aten.tanh_.default,
}

_SELECT_OPS = {
    aten.where.self,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
}


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _storage_key(x: torch.Tensor) -> tuple[Any, ...]:
    try:
        storage_ptr = x.untyped_storage().data_ptr()
    except (RuntimeError, AttributeError):
        storage_ptr = id(x)
    return (
        x.device.type,
        x.device.index,
        x.dtype,
        storage_ptr,
    )


def _numel_tree(tree: Any) -> int:
    return sum(int(x.numel()) for x in _collect_tensors(tree))


def _bytes_tree(tree: Any) -> int:
    return sum(int(x.numel() * x.element_size()) for x in _collect_tensors(tree))


class NeuronStateCounter(BaseCounter):
    r"""
    **API Language:**
    :ref:`中文 <NeuronStateCounter-cn>` |
    :ref:`English <NeuronStateCounter-en>`

    ----

    .. _NeuronStateCounter-cn:

    * **中文**

    神经元内部状态计数器，用于统计 ``BaseNode`` 及其子类在运行时的状态读写和原语操作。

    该计数器会输出两类结果：

    - ``metric_records``：细粒度状态统计，如 ``state_reads``、``state_writes``、
      ``state_adds``、``state_nonlinear_ops`` 等
    - ``projection_records``：较粗粒度投影，如 ``read_potential``、
      ``write_potential``、``state_mac_like``、``state_acc_like``

    :param strict: 是否在遇到不支持的 backend 时直接抛异常
    :type strict: bool
    :param extra_state_rules: 额外的状态规则，格式为
      ``{module_type: callable}``。其中 ``callable`` 的签名为
      ``(module, func, args, kwargs, out, state_tensor_keys) -> dict | None``；
      其中 ``state_tensor_keys`` 为 ``_storage_key(tensor)`` 形式的键集合；
      若返回非 ``None``，则覆盖默认统计逻辑
    :type extra_state_rules: Optional[dict[type[nn.Module], Callable]]

    ----

    .. _NeuronStateCounter-en:

    * **English**

    Counter for tracking runtime state reads/writes and primitive operations
    inside ``BaseNode`` and its subclasses.

    It exposes two result families:

    - ``metric_records``: fine-grained state metrics such as ``state_reads``,
      ``state_writes``, ``state_adds``, and ``state_nonlinear_ops``
    - ``projection_records``: coarser projections such as ``read_potential``,
      ``write_potential``, ``state_mac_like``, and ``state_acc_like``

    :param strict: whether to raise immediately on unsupported backends
    :type strict: bool
    :param extra_state_rules: additional rules in the form
      ``{module_type: callable}``. The callable signature is
      ``(module, func, args, kwargs, out, state_tensor_keys) -> dict | None``;
      ``state_tensor_keys`` contains ``_storage_key(tensor)`` tuples, and a
      custom rule should compare against those keys rather than ``id(tensor)``.
      When it returns non-``None``, the default counting logic is overridden
    :type extra_state_rules: Optional[dict[type[nn.Module], Callable]]
    """
    def __init__(
        self,
        *,
        strict: bool = False,
        extra_state_rules: dict[type[nn.Module], Callable] | None = None,
    ):
        super().__init__()
        self.strict = strict
        rules = extra_state_rules or {}
        for module_type, rule in rules.items():
            if not callable(rule):
                raise TypeError(
                    "extra_state_rules values must be callable, "
                    f"got {type(rule).__name__} for {module_type.__name__}."
                )
        self.extra_state_rules = dict(rules)
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

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules: set[nn.Module] | None = None,
        parent_names: set[str] | None = None,
    ) -> int:
        active_modules = set() if active_modules is None else active_modules
        active_base_nodes = [m for m in active_modules if isinstance(m, BaseNode)]
        if not active_base_nodes:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        for module in active_base_nodes:
            backend = module.backend
            if backend is not None and backend != "torch":
                self._warn_or_raise(
                    module,
                    f"NeuronStateCounter only supports torch backend, got "
                    f"{backend!r} from {module.__class__.__name__}.",
                )
                self._pending_metrics = None
                self._pending_projection = None
                return 0

        state_tensor_keys: set[tuple[Any, ...]] = set()
        for module in active_base_nodes:
            for value in module._memories.values():
                if torch.is_tensor(value):
                    state_tensor_keys.add(_storage_key(value))

        if not state_tensor_keys:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        for module in active_base_nodes:
            rule = self.extra_state_rules.get(type(module))
            if rule is None:
                continue
            breakdown = rule(module, func, args, kwargs, out, state_tensor_keys)
            if breakdown is not None:
                return self._store_breakdown(breakdown)

        op_name = resolve_name(func)
        if func in _IGNORED_OPS or op_name.startswith(_IGNORED_OP_PREFIXES):
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        tensors_in = _collect_tensors((args, kwargs))
        state_tensors = [x for x in tensors_in if _storage_key(x) in state_tensor_keys]
        if not state_tensors:
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        output_tensors = _collect_tensors(out)
        out_numel = _numel_tree(out)
        out_bytes = _bytes_tree(out)
        state_buffer_bytes = max(
            (int(x.numel() * x.element_size()) for x in state_tensors), default=0
        )
        metrics = {
            "state_reads": sum(int(x.numel() * x.element_size()) for x in state_tensors),
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
        if func in _ADD_OPS:
            metrics["state_adds"] += out_numel
            writes_state = True
        elif func in _MUL_OPS:
            metrics["state_muls"] += out_numel
            writes_state = True
        elif func in _COMP_OPS:
            metrics["state_comps"] += out_numel
        elif func in _NONLINEAR_OPS:
            metrics["state_nonlinear_ops"] += out_numel
            writes_state = func in _INPLACE_NONLINEAR_OPS
        elif func in _SELECT_OPS:
            metrics["state_select_ops"] += out_numel
            writes_state = True

        if writes_state and output_tensors:
            metrics["state_writes"] += out_bytes
            non_state_tensors = [
                x for x in tensors_in if _storage_key(x) not in state_tensor_keys
            ]
            has_spike_gate = any(is_binary_tensor(x) for x in non_state_tensors)
            if has_spike_gate:
                metrics["state_reset_ops"] += out_numel
                if func not in _SELECT_OPS:
                    metrics["state_select_ops"] += out_numel
                metrics["spike_triggered_ops"] += out_numel
            else:
                metrics["timestep_dense_ops"] += out_numel

        if not any(metrics.values()):
            self._pending_metrics = None
            self._pending_projection = None
            return 0

        return self._store_breakdown(metrics, state_buffer_bytes=state_buffer_bytes)

    def _store_breakdown(
        self, metrics: dict[str, int], *, state_buffer_bytes: int = 0
    ) -> int:
        projection = {
            "read_potential": metrics.get("state_reads", 0),
            "write_potential": metrics.get("state_writes", 0),
            "state_mac_like": metrics.get("state_muls", 0),
            "potential_buffer_bytes": max(
                metrics.get("state_writes", 0), state_buffer_bytes
            ),
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
                if key == "potential_buffer_bytes":
                    self.projection_records[scope][key] = max(
                        self.projection_records[scope][key], int(item)
                    )
                else:
                    self.projection_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None
        self._pending_projection = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <NeuronStateCounter.get_metric_counts-cn>` |
        :ref:`English <NeuronStateCounter.get_metric_counts-en>`

        ----

        .. _NeuronStateCounter.get_metric_counts-cn:

        * **中文**

        :return: 按 scope 聚合的细粒度状态统计
        :rtype: dict[str, dict[str, int]]

        ----

        .. _NeuronStateCounter.get_metric_counts-en:

        * **English**

        :return: fine-grained state metrics aggregated by scope
        :rtype: dict[str, dict[str, int]]
        """
        return {scope: dict(items) for scope, items in self.metric_records.items()}

    def get_projection_counts(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <NeuronStateCounter.get_projection_counts-cn>` |
        :ref:`English <NeuronStateCounter.get_projection_counts-en>`

        ----

        .. _NeuronStateCounter.get_projection_counts-cn:

        * **中文**

        :return: 按 scope 聚合的状态投影统计
        :rtype: dict[str, dict[str, int]]

        ----

        .. _NeuronStateCounter.get_projection_counts-en:

        * **English**

        :return: projected state statistics aggregated by scope
        :rtype: dict[str, dict[str, int]]
        """
        return {scope: dict(items) for scope, items in self.projection_records.items()}
