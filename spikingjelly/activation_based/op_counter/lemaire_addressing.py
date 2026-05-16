from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter, is_binary_tensor

aten = torch.ops.aten
__all__ = ["LemaireAddressingCounter"]


def _prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


class LemaireAddressingCounter(BaseCounter):
    def __init__(self):
        super().__init__()
        self.rules: dict[Any, Callable] = {
            aten.mm.default: self._count_mm,
            aten.addmm.default: self._count_addmm,
            aten.bmm.default: self._count_bmm,
            aten.baddbmm.default: self._count_baddbmm,
            aten.convolution.default: self._count_convolution,
        }
        self.metric_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._pending_metrics: dict[str, int] | None = None

    def _pick_supported_module(
        self, active_modules: set[nn.Module] | None, expected_type: Any
    ) -> nn.Module | None:
        if active_modules is None:
            return None
        for module in active_modules:
            if isinstance(module, expected_type):
                return module
        return None

    def _store_metrics(self, *, acc_addr: int = 0, mac_addr: int = 0) -> int:
        metrics = {
            "acc_addr": int(acc_addr),
            "mac_addr": int(mac_addr),
        }
        self._pending_metrics = metrics
        return metrics["acc_addr"] + metrics["mac_addr"]

    def _count_linear_tensor(self, x: torch.Tensor, out: torch.Tensor) -> int:
        if is_binary_tensor(x):
            return self._store_metrics(
                acc_addr=int(x.count_nonzero().item()) * int(out.shape[-1])
            )
        return self._store_metrics(acc_addr=int(x.numel()) + int(out.numel()))

    def _count_mm(self, args: tuple, kwargs: dict, out, module: nn.Linear) -> int:
        del kwargs, module
        x = args[0]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            self._pending_metrics = None
            return 0
        return self._count_linear_tensor(x, out)

    def _count_addmm(self, args: tuple, kwargs: dict, out, module: nn.Linear) -> int:
        del kwargs, module
        x = args[1]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            self._pending_metrics = None
            return 0
        return self._count_linear_tensor(x, out)

    def _count_bmm(self, args: tuple, kwargs: dict, out, module: nn.Linear) -> int:
        del kwargs, module
        x = args[0]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            self._pending_metrics = None
            return 0
        return self._count_linear_tensor(x, out)

    def _count_baddbmm(self, args: tuple, kwargs: dict, out, module: nn.Linear) -> int:
        del kwargs, module
        x = args[1]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            self._pending_metrics = None
            return 0
        return self._count_linear_tensor(x, out)

    def _count_convolution(
        self, args: tuple, kwargs: dict, out, module: nn.Module
    ) -> int:
        del kwargs
        x, _w, _bias, _stride, _padding, _dilation, transposed, _output_padding, _groups = (
            args[:9]
        )
        if transposed or not torch.is_tensor(x) or not torch.is_tensor(out):
            self._pending_metrics = None
            return 0
        kernel_size = module.kernel_size
        kernel_volume = kernel_size if isinstance(kernel_size, int) else _prod(kernel_size)
        if is_binary_tensor(x):
            spike_num_in = int(x.count_nonzero().item())
            out_channels_per_group = int(module.out_channels) // int(module.groups)
            return self._store_metrics(
                acc_addr=spike_num_in * out_channels_per_group * kernel_volume,
                mac_addr=spike_num_in * 2,
            )
        return self._store_metrics(
            acc_addr=int(x.numel()) + int(out.numel()) + int(module.out_channels) * kernel_volume
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
        del parent_names
        if func in (
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        ):
            module = self._pick_supported_module(active_modules, nn.Linear)
            if module is None:
                self._pending_metrics = None
                return 0
            return int(self.rules[func](args, kwargs, out, module))

        if func == aten.convolution.default:
            module = self._pick_supported_module(active_modules, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            if module is None:
                self._pending_metrics = None
                return 0
            return int(self.rules[func](args, kwargs, out, module))

        self._pending_metrics = None
        return 0

    def record(self, scope, func, value):
        super().record(scope, func, value)
        if self._pending_metrics is not None:
            for key, item in self._pending_metrics.items():
                self.metric_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        return {scope: dict(items) for scope, items in self.metric_records.items()}
