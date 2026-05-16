from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter, is_binary_tensor

aten = torch.ops.aten
__all__ = ["LemaireAddressingCounter"]


def _prod(dims):
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _address_linear(x: torch.Tensor, out: torch.Tensor) -> dict[str, int]:
    if is_binary_tensor(x):
        return {
            "acc_addr": int(x.count_nonzero().item()) * int(out.shape[-1]),
            "mac_addr": 0,
        }
    return {
        "acc_addr": int(x.numel()) + int(out.numel()),
        "mac_addr": 0,
    }


def _address_mm(args, kwargs, out):
    del kwargs
    x = args[0]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_addmm(args, kwargs, out):
    del kwargs
    x = args[1]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_bmm(args, kwargs, out):
    del kwargs
    x = args[0]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_baddbmm(args, kwargs, out):
    del kwargs
    x = args[1]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_convolution(args, kwargs, out):
    del kwargs
    x, w, _bias, _stride, _padding, _dilation, transposed, _output_padding, groups = (
        args[:9]
    )
    if transposed or not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}

    kernel_volume = _prod(w.shape[2:])
    out_channels = int(w.shape[0])
    if is_binary_tensor(x):
        spike_num_in = int(x.count_nonzero().item())
        out_channels_per_group = out_channels // int(groups)
        return {
            "acc_addr": spike_num_in * out_channels_per_group * kernel_volume,
            "mac_addr": spike_num_in * 2,
        }

    return {
        "acc_addr": int(x.numel()) + int(out.numel()) + out_channels * kernel_volume,
        "mac_addr": 0,
    }


class LemaireAddressingCounter(BaseCounter):
    def __init__(self):
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _address_mm,
            aten.addmm.default: _address_addmm,
            aten.bmm.default: _address_bmm,
            aten.baddbmm.default: _address_baddbmm,
            aten.convolution.default: _address_convolution,
        }
        self.ignore_modules = []
        self.metric_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._pending_metrics: dict[str, int] | None = None

    def _active_linear(self, active_modules: set[nn.Module] | None) -> bool:
        if active_modules is None:
            return False
        return any(isinstance(module, nn.Linear) for module in active_modules)

    def _active_conv(self, active_modules: set[nn.Module] | None) -> bool:
        if active_modules is None:
            return False
        return any(
            isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            for module in active_modules
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
            if not self._active_linear(active_modules):
                self._pending_metrics = None
                return 0
        elif func == aten.convolution.default:
            if not self._active_conv(active_modules):
                self._pending_metrics = None
                return 0
        else:
            self._pending_metrics = None
            return 0

        metrics = self.rules[func](args, kwargs, out)
        self._pending_metrics = {
            "acc_addr": int(metrics["acc_addr"]),
            "mac_addr": int(metrics["mac_addr"]),
        }
        return self._pending_metrics["acc_addr"] + self._pending_metrics["mac_addr"]

    def record(self, scope, func, value):
        super().record(scope, func, value)
        if self._pending_metrics is not None:
            for key, item in self._pending_metrics.items():
                self.metric_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        return {scope: dict(items) for scope, items in self.metric_records.items()}
