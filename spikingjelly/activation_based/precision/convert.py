from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .. import layer
from ..neuron.base_node import BaseNode, SimpleBaseNode
from .float8_base import wrap_float8_linear_module


@dataclass
class ConversionReport:
    total_modules: int = 0
    convertible_linear: int = 0
    convertible_torch_linear: int = 0
    convertible_modules: list[str] = field(default_factory=list)
    converted_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    high_precision_modules: list[str] = field(default_factory=list)
    unsupported_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_modules": self.total_modules,
            "convertible_linear": self.convertible_linear,
            "convertible_torch_linear": self.convertible_torch_linear,
            "convertible_modules": self.convertible_modules,
            "converted_modules": self.converted_modules,
            "skipped_modules": self.skipped_modules,
            "high_precision_modules": self.high_precision_modules,
            "unsupported_modules": self.unsupported_modules,
        }


def analyze_convertible_modules(model: nn.Module) -> ConversionReport:
    report = ConversionReport()
    unsupported_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention)
    high_precision_types = (
        BaseNode,
        SimpleBaseNode,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
    )
    for name, module in model.named_modules():
        report.total_modules += 1
        if isinstance(module, layer.Linear):
            report.convertible_linear += 1
            report.convertible_modules.append(name or "<root>")
        elif isinstance(module, nn.Linear):
            report.convertible_torch_linear += 1
            report.convertible_modules.append(name or "<root>")
        elif isinstance(module, high_precision_types):
            report.high_precision_modules.append(name or "<root>")
        elif isinstance(module, unsupported_types):
            report.unsupported_modules.append(name or "<root>")
    return report


def convert_model_for_precision(model: nn.Module, policy) -> tuple[nn.Module, ConversionReport]:
    report = analyze_convertible_modules(model)
    if getattr(policy, "name", "") == "fp8-torchao":
        from torchao.float8.float8_linear import Float8Linear
        fp8_config = getattr(policy, "float8_linear_config", None)
        if fp8_config is None:
            raise RuntimeError(
                "Float8TorchAOPolicy.check_capability() must be called before convert_model_for_precision()."
            )

        if isinstance(model, (nn.Linear, layer.Linear)):
            converted = Float8Linear.from_float(model, config=fp8_config)
            report.converted_modules.append("<root>")
            return wrap_float8_linear_module(model, converted), report

        visited = set()

        def recursive_convert(module: nn.Module, prefix: str = "", memo=None):
            if memo is None:
                memo = {}
            if module in visited:
                return
            visited.add(module)
            for child_name, child in list(module.named_children()):
                child_fqn = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, (nn.Linear, layer.Linear)):
                    if child in memo:
                        wrapped = memo[child]
                        if isinstance(module, nn.ModuleList):
                            module[int(child_name)] = wrapped
                        elif isinstance(module, nn.ModuleDict):
                            module[child_name] = wrapped
                        else:
                            setattr(module, child_name, wrapped)
                        report.converted_modules.append(child_fqn)
                        continue
                    converted = Float8Linear.from_float(child, config=fp8_config)
                    wrapped = wrap_float8_linear_module(child, converted)
                    memo[child] = wrapped
                    if isinstance(module, nn.ModuleList):
                        module[int(child_name)] = wrapped
                    elif isinstance(module, nn.ModuleDict):
                        module[child_name] = wrapped
                    else:
                        setattr(module, child_name, wrapped)
                    report.converted_modules.append(child_fqn)
                else:
                    report.skipped_modules.append(child_fqn)
                    recursive_convert(child, child_fqn, memo)

        recursive_convert(model)
    return model, report
