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
    convertible_layer_norm: int = 0
    convertible_modules: list[str] = field(default_factory=list)
    converted_modules: list[str] = field(default_factory=list)
    converted_patterns: list[dict[str, str]] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    high_precision_modules: list[str] = field(default_factory=list)
    unsupported_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_modules": self.total_modules,
            "convertible_linear": self.convertible_linear,
            "convertible_torch_linear": self.convertible_torch_linear,
            "convertible_pointwise_conv1d": self.convertible_pointwise_conv1d,
            "convertible_layer_norm": self.convertible_layer_norm,
            "convertible_modules": self.convertible_modules,
            "converted_modules": self.converted_modules,
            "converted_patterns": self.converted_patterns,
            "skipped_modules": self.skipped_modules,
            "high_precision_modules": self.high_precision_modules,
            "unsupported_modules": self.unsupported_modules,
        }


def analyze_convertible_modules(
    model: nn.Module,
    *,
    include_layer_norm: bool = False,
) -> ConversionReport:
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
        elif include_layer_norm and isinstance(module, nn.LayerNorm):
            report.convertible_layer_norm += 1
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
    report = analyze_convertible_modules(
        model,
        include_layer_norm=getattr(policy, "supports_layer_norm_conversion", False),
    )
    model = policy._convert_modules(model, report)
    return model, report


def _configure_triton_neurons(model: nn.Module, config, device) -> dict:
    from ..neuron.integrate_and_fire import IFNode
    from ..neuron.lif import LIFNode
    from ..neuron.plif import ParametricLIFNode
    from ..triton_kernel.neuron_kernel.utils import (
        _prepare_triton_neuron_execution_plan,
    )
    from ..triton_kernel.triton_utils import normalize_triton_storage_dtype

    neuron_types = {
        IFNode: "if",
        LIFNode: "lif",
        ParametricLIFNode: "plif",
    }
    if config.triton_storage is None:
        for module in model.modules():
            if type(module) in neuron_types:
                module._triton_precision = None
        return {"converted_modules": [], "unsupported_modules": []}

    targets = []
    unsupported = []
    precision = (
        normalize_triton_storage_dtype(config.triton_storage),
        config.triton_fwd,
        config.triton_bwd,
    )
    for name, module in model.named_modules():
        neuron_type = neuron_types.get(type(module))
        if neuron_type is None or module.backend != "triton":
            continue
        if module.step_mode != "m":
            unsupported.append(name or "<root>")
            continue
        targets.append((name or "<root>", module, neuron_type))

    if unsupported:
        raise RuntimeError(
            "Triton neuron precision requires multi-step IF/LIF/PLIF nodes: "
            + ", ".join(unsupported)
        )
    if not targets:
        raise RuntimeError(
            "Triton neuron precision requested, but no multi-step IF/LIF/PLIF "
            "nodes with backend='triton' were found."
        )

    for _, _, neuron_type in targets:
        _prepare_triton_neuron_execution_plan(
            neuron_type=neuron_type,
            device=device,
            storage_dtype=precision[0],
            forward_compute_dtype=precision[1],
            backward_compute_dtype=precision[2],
        )
    for _, module, _ in targets:
        module._triton_precision = precision
    return {
        "converted_modules": [name for name, _, _ in targets],
        "unsupported_modules": [],
    }
