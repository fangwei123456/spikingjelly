from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from ._sparse_memory import (
    active_element_count,
    dense_bytes,
    is_sparse_access_tensor,
    sparse_bytes,
)
from .base import BaseCounter
from .memory_access import MemoryAccessCounter

aten = torch.ops.aten
__all__ = ["AnalyticalMemoryCounter"]


class AnalyticalMemoryCounter(BaseCounter):
    def __init__(
        self,
        *,
        zero_ratio_threshold: float = 0.5,
        enable_sparse_memory_estimation: bool = True,
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__()
        dense_counter = MemoryAccessCounter(extra_ignore_modules=extra_ignore_modules)
        self.rules = dict(dense_counter.rules)
        self.ignore_modules = list(extra_ignore_modules)
        self.zero_ratio_threshold = zero_ratio_threshold
        self.enable_sparse_memory_estimation = enable_sparse_memory_estimation
        self.metric_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.warnings: list[str] = []
        self._warned_module_types: set[type[nn.Module]] = set()
        self._pending_metrics: dict[str, int] | None = None
        self._sparse_aware_ops = {
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
            aten.convolution.default,
        }
        self._supported_sparse_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        )

    def _tree_has_sparse_tensor(self, tree: Any) -> bool:
        if torch.is_tensor(tree):
            return self.enable_sparse_memory_estimation and is_sparse_access_tensor(
                tree, zero_ratio_threshold=self.zero_ratio_threshold
            )
        if isinstance(tree, (tuple, list)):
            return any(self._tree_has_sparse_tensor(item) for item in tree)
        if isinstance(tree, dict):
            return any(self._tree_has_sparse_tensor(item) for item in tree.values())
        return False

    def _warn_fallback(self, active_modules: set[nn.Module] | None = None) -> None:
        active_modules = set() if active_modules is None else active_modules
        unsupported_modules = [
            module
            for module in active_modules
            if len(list(module.children())) == 0
            if not isinstance(module, self._supported_sparse_modules)
        ]
        if not unsupported_modules:
            return
        module_type = type(unsupported_modules[0])
        if module_type in self._warned_module_types:
            return
        self._warned_module_types.add(module_type)
        message = (
            f"AnalyticalMemoryCounter falls back to dense lower-bound memory for "
            f"{module_type.__name__}."
        )
        self.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _store_metrics(self, metrics: dict[str, int]) -> int:
        metrics["memory_access_bytes"] = (
            metrics.get("read_in_bytes", 0)
            + metrics.get("read_params_bytes", 0)
            + metrics.get("write_out_bytes", 0)
            + metrics.get("other_memory_bytes", 0)
        )
        if "memory_buffer_bytes" not in metrics:
            metrics["memory_buffer_bytes"] = max(
                metrics.get("read_in_buffer_bytes", metrics.get("read_in_bytes", 0)),
                metrics.get(
                    "read_params_buffer_bytes", metrics.get("read_params_bytes", 0)
                ),
                metrics.get(
                    "write_out_buffer_bytes", metrics.get("write_out_bytes", 0)
                ),
                metrics.get(
                    "other_memory_buffer_bytes", metrics.get("other_memory_bytes", 0)
                ),
            )
        self._pending_metrics = metrics
        return metrics["memory_access_bytes"]

    def _max_tensor_bytes(self, tree: Any) -> int:
        if torch.is_tensor(tree):
            return dense_bytes(tree)
        if isinstance(tree, (tuple, list)):
            return max((self._max_tensor_bytes(item) for item in tree), default=0)
        if isinstance(tree, dict):
            return max((self._max_tensor_bytes(item) for item in tree.values()), default=0)
        return 0

    def _dense_fallback(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out: Any,
        active_modules: set[nn.Module] | None = None,
    ) -> int:
        value = int(self.rules[func](args, kwargs, out))
        if self._tree_has_sparse_tensor((args, kwargs)) or self._tree_has_sparse_tensor(
            out
        ):
            self._warn_fallback(active_modules=active_modules)
        return self._store_metrics(
            {
                "read_in_bytes": 0,
                "read_params_bytes": 0,
                "write_out_bytes": 0,
                "dense_memory_bytes": value,
                "sparse_memory_bytes": 0,
                "other_memory_bytes": value,
                "fallback_dense_ops": 1,
                "other_memory_buffer_bytes": max(
                    self._max_tensor_bytes((args, kwargs)),
                    self._max_tensor_bytes(out),
                ),
            }
        )

    def _linear_metrics(
        self, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, bias: torch.Tensor | None
    ) -> dict[str, int]:
        input_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
            x, zero_ratio_threshold=self.zero_ratio_threshold
        )
        output_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
            out, zero_ratio_threshold=self.zero_ratio_threshold
        )
        if not input_is_sparse:
            read_in = dense_bytes(x)
            read_params = dense_bytes(y)
            if bias is not None:
                read_params += dense_bytes(bias)
            write_out = dense_bytes(out)
            return {
                "read_in_bytes": read_in,
                "read_params_bytes": read_params,
                "write_out_bytes": write_out,
                "read_in_buffer_bytes": dense_bytes(x),
                "read_params_buffer_bytes": dense_bytes(y)
                + (dense_bytes(bias) if bias is not None else 0),
                "write_out_buffer_bytes": dense_bytes(out),
                "dense_memory_bytes": read_in + read_params + write_out,
                "sparse_memory_bytes": 0,
                "other_memory_bytes": 0,
            }

        active_inputs = active_element_count(x)
        read_in = sparse_bytes(x)
        read_params = active_inputs * y.shape[1] * int(y.element_size())
        if bias is not None:
            output_active = (
                active_element_count(out) if output_is_sparse else int(out.numel())
            )
            read_params += output_active * int(bias.element_size())
        write_out = sparse_bytes(out) if output_is_sparse else dense_bytes(out)
        total = read_in + read_params + write_out
        return {
            "read_in_bytes": read_in,
            "read_params_bytes": read_params,
            "write_out_bytes": write_out,
            "read_in_buffer_bytes": dense_bytes(x),
            "read_params_buffer_bytes": dense_bytes(y)
            + (dense_bytes(bias) if bias is not None else 0),
            "write_out_buffer_bytes": dense_bytes(out),
            "dense_memory_bytes": 0,
            "sparse_memory_bytes": total,
            "other_memory_bytes": 0,
        }

    def _conv_metrics(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        out: torch.Tensor,
        transposed: bool,
        groups: int,
    ) -> dict[str, int] | None:
        if transposed:
            return None
        input_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
            x, zero_ratio_threshold=self.zero_ratio_threshold
        )
        output_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
            out, zero_ratio_threshold=self.zero_ratio_threshold
        )
        if not input_is_sparse:
            read_in = dense_bytes(x)
            read_params = dense_bytes(w)
            if bias is not None:
                read_params += dense_bytes(bias)
            write_out = dense_bytes(out)
            return {
                "read_in_bytes": read_in,
                "read_params_bytes": read_params,
                "write_out_bytes": write_out,
                "read_in_buffer_bytes": dense_bytes(x),
                "read_params_buffer_bytes": dense_bytes(w)
                + (dense_bytes(bias) if bias is not None else 0),
                "write_out_buffer_bytes": dense_bytes(out),
                "dense_memory_bytes": read_in + read_params + write_out,
                "sparse_memory_bytes": 0,
                "other_memory_bytes": 0,
            }

        active_inputs = active_element_count(x)
        kernel_volume = 1
        for dim in w.shape[2:]:
            kernel_volume *= int(dim)
        out_channels_per_group = int(w.shape[0]) // groups
        read_in = sparse_bytes(x)
        read_params = (
            active_inputs * out_channels_per_group * kernel_volume * int(w.element_size())
        )
        if bias is not None:
            output_active = (
                active_element_count(out) if output_is_sparse else int(out.numel())
            )
            read_params += output_active * int(bias.element_size())
        write_out = sparse_bytes(out) if output_is_sparse else dense_bytes(out)
        total = read_in + read_params + write_out
        return {
            "read_in_bytes": read_in,
            "read_params_bytes": read_params,
            "write_out_bytes": write_out,
            "read_in_buffer_bytes": dense_bytes(x),
            "read_params_buffer_bytes": dense_bytes(w)
            + (dense_bytes(bias) if bias is not None else 0),
            "write_out_buffer_bytes": dense_bytes(out),
            "dense_memory_bytes": 0,
            "sparse_memory_bytes": total,
            "other_memory_bytes": 0,
        }

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules: set[nn.Module] | None = None,
        parent_names: set[str] | None = None,
    ) -> int:
        del parent_names
        self._pending_metrics = None

        if func not in self._sparse_aware_ops:
            return self._dense_fallback(
                func, args, kwargs, out, active_modules=active_modules
            )

        if func is aten.mm.default:
            x, y = args[:2]
            return self._store_metrics(self._linear_metrics(x, y, out, None))

        if func is aten.addmm.default:
            bias, x, y = args[:3]
            return self._store_metrics(self._linear_metrics(x, y, out, bias))

        if func is aten.bmm.default:
            x, y = args[:2]
            input_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
                x, zero_ratio_threshold=self.zero_ratio_threshold
            )
            output_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
                out, zero_ratio_threshold=self.zero_ratio_threshold
            )
            if not input_is_sparse:
                read_in = dense_bytes(x)
                read_params = dense_bytes(y)
                write_out = dense_bytes(out)
                return self._store_metrics(
                    {
                        "read_in_bytes": read_in,
                        "read_params_bytes": read_params,
                        "write_out_bytes": write_out,
                        "read_in_buffer_bytes": dense_bytes(x),
                        "read_params_buffer_bytes": dense_bytes(y),
                        "write_out_buffer_bytes": dense_bytes(out),
                        "dense_memory_bytes": read_in + read_params + write_out,
                        "sparse_memory_bytes": 0,
                        "other_memory_bytes": 0,
                    }
                )
            active_inputs = active_element_count(x)
            read_in = sparse_bytes(x)
            read_params = active_inputs * y.shape[2] * int(y.element_size())
            write_out = sparse_bytes(out) if output_is_sparse else dense_bytes(out)
            total = read_in + read_params + write_out
            return self._store_metrics(
                {
                    "read_in_bytes": read_in,
                    "read_params_bytes": read_params,
                    "write_out_bytes": write_out,
                    "read_in_buffer_bytes": dense_bytes(x),
                    "read_params_buffer_bytes": dense_bytes(y),
                    "write_out_buffer_bytes": dense_bytes(out),
                    "dense_memory_bytes": 0,
                    "sparse_memory_bytes": total,
                    "other_memory_bytes": 0,
                }
            )

        if func is aten.baddbmm.default:
            bias, x, y = args[:3]
            alpha = kwargs.get("alpha", 1)
            beta = kwargs.get("beta", 1)
            if alpha != 1:
                return self._dense_fallback(
                    func, args, kwargs, out, active_modules=active_modules
                )
            input_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
                x, zero_ratio_threshold=self.zero_ratio_threshold
            )
            output_is_sparse = self.enable_sparse_memory_estimation and is_sparse_access_tensor(
                out, zero_ratio_threshold=self.zero_ratio_threshold
            )
            if not input_is_sparse:
                read_in = dense_bytes(x)
                read_params = dense_bytes(y)
                if beta != 0:
                    read_params += dense_bytes(bias)
                write_out = dense_bytes(out)
                return self._store_metrics(
                    {
                        "read_in_bytes": read_in,
                        "read_params_bytes": read_params,
                        "write_out_bytes": write_out,
                        "read_in_buffer_bytes": dense_bytes(x),
                        "read_params_buffer_bytes": dense_bytes(y)
                        + (dense_bytes(bias) if beta != 0 else 0),
                        "write_out_buffer_bytes": dense_bytes(out),
                        "dense_memory_bytes": read_in + read_params + write_out,
                        "sparse_memory_bytes": 0,
                        "other_memory_bytes": 0,
                    }
                )
            active_inputs = active_element_count(x)
            read_in = sparse_bytes(x)
            read_params = active_inputs * y.shape[2] * int(y.element_size())
            if beta != 0:
                output_active = (
                    active_element_count(out) if output_is_sparse else int(out.numel())
                )
                read_params += output_active * int(bias.element_size())
            write_out = sparse_bytes(out) if output_is_sparse else dense_bytes(out)
            total = read_in + read_params + write_out
            return self._store_metrics(
                {
                    "read_in_bytes": read_in,
                    "read_params_bytes": read_params,
                    "write_out_bytes": write_out,
                    "read_in_buffer_bytes": dense_bytes(x),
                    "read_params_buffer_bytes": dense_bytes(y)
                    + (dense_bytes(bias) if beta != 0 else 0),
                    "write_out_buffer_bytes": dense_bytes(out),
                    "dense_memory_bytes": 0,
                    "sparse_memory_bytes": total,
                    "other_memory_bytes": 0,
                }
            )

        x, w, bias, _stride, _padding, _dilation, transposed, _output_padding, groups = (
            args[:9]
        )
        metrics = self._conv_metrics(x, w, bias, out, transposed, groups)
        if metrics is None:
            return self._dense_fallback(
                func, args, kwargs, out, active_modules=active_modules
            )
        return self._store_metrics(metrics)

    def record(self, scope, func, value):
        super().record(scope, func, value)
        if self._pending_metrics is not None:
            for key, item in self._pending_metrics.items():
                if key == "memory_buffer_bytes":
                    self.metric_records[scope][key] = max(
                        self.metric_records[scope][key], int(item)
                    )
                else:
                    self.metric_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        return {scope: dict(items) for scope, items in self.metric_records.items()}

    def get_extra_counts(self) -> dict[str, dict[str, int]]:
        return self.get_metric_counts()
