from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .. import layer
from ..neuron.base_node import BaseNode, SimpleBaseNode
from .float8_conv import is_supported_pointwise_conv1d


@dataclass
class ConversionReport:
    total_modules: int = 0
    convertible_linear: int = 0
    convertible_torch_linear: int = 0
    convertible_pointwise_conv1d: int = 0
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
            "convertible_pointwise_conv1d": self.convertible_pointwise_conv1d,
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
        elif is_supported_pointwise_conv1d(module):
            report.convertible_pointwise_conv1d += 1
            report.convertible_modules.append(name or "<root>")
        elif isinstance(module, high_precision_types):
            report.high_precision_modules.append(name or "<root>")
        elif isinstance(module, unsupported_types):
            report.unsupported_modules.append(name or "<root>")
    return report


def convert_model_for_precision(
    model: nn.Module, policy
) -> tuple[nn.Module, ConversionReport]:
    """Analyse then delegate module-level conversion to *policy*.
    The default policy returns the model unchanged; policies that require
    structural changes (e.g. float8 kernel substitution) override
    ``_convert_modules`` to perform the actual transformation.
    """
    report = analyze_convertible_modules(model)
    model = policy._convert_modules(model, report)
    return model, report
