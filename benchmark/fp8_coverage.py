from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from spikingjelly.activation_based.precision.float8_base import Float8LinearStepModule
from spikingjelly.activation_based.precision.float8_conv import (
    Float8PointwiseConv1dStepModule,
)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _module_macs(module: nn.Module, output: Any) -> int:
    output_tensor = _first_tensor(output)
    if output_tensor is None:
        return 0
    if isinstance(module, (nn.Linear, Float8LinearStepModule)):
        return output_tensor.numel() * module.in_features
    if isinstance(module, Float8PointwiseConv1dStepModule):
        return output_tensor.numel() * module.in_channels
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        kernel_elements = math.prod(module.kernel_size)
        return (
            output_tensor.numel()
            * (module.in_channels // module.groups)
            * kernel_elements
        )
    return 0


class _FP8CoverageTracker:
    def __init__(
        self,
        model: nn.Module,
        conversion_report: dict,
    ) -> None:
        converted = {
            "" if name == "<root>" else name
            for name in conversion_report.get("converted_modules", ())
        }
        self._records: dict[str, dict[str, Any]] = {}
        self._handles = []
        selected_prefixes: list[str] = []
        types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            Float8LinearStepModule,
            Float8PointwiseConv1dStepModule,
        )
        for name, module in model.named_modules():
            if any(name.startswith(prefix + ".") for prefix in selected_prefixes):
                continue
            if not isinstance(module, types):
                continue
            selected_prefixes.append(name)
            requested_fp8 = name in converted
            self._records[name] = {
                "name": name or "<root>",
                "type": type(module).__name__,
                "requested_fp8": requested_fp8,
                "calls": 0,
                "macs": 0,
            }
            self._handles.append(
                module.register_forward_hook(self._hook(name), always_call=True)
            )

    def _hook(self, name: str):
        def hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            record = self._records[name]
            record["calls"] += 1
            record["macs"] += _module_macs(module, output)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self) -> dict[str, Any]:
        executed = [record for record in self._records.values() if record["calls"]]
        total_macs = sum(record["macs"] for record in executed)
        fp8_macs = sum(record["macs"] for record in executed if record["requested_fp8"])
        return {
            "executed_module_count": len(executed),
            "requested_fp8_module_count": sum(
                record["requested_fp8"] for record in executed
            ),
            "total_dense_macs": total_macs,
            "requested_fp8_dense_macs": fp8_macs,
            "requested_fp8_dense_mac_coverage": (
                fp8_macs / total_macs if total_macs else 0.0
            ),
            "modules": executed,
        }
