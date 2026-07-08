from __future__ import annotations

from contextlib import nullcontext
import warnings

import torch
import torch.nn as nn

from .. import layer
from .float8_base import wrap_float8_linear_module
from .policy import PrecisionPolicy


def _import_te_pytorch():
    import transformer_engine.pytorch as te

    return te


def _replace_child(module: nn.Module, child_name: str, wrapped: nn.Module) -> None:
    if isinstance(module, nn.ModuleDict):
        module[child_name] = wrapped
    elif isinstance(module, (nn.ModuleList, nn.Sequential)):
        try:
            module[int(child_name)] = wrapped
        except ValueError:
            setattr(module, child_name, wrapped)
    else:
        setattr(module, child_name, wrapped)


def _copy_linear_parameters(source: nn.Module, target: nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        source_bias = getattr(source, "bias", None)
        target_bias = getattr(target, "bias", None)
        if source_bias is not None and target_bias is not None:
            target_bias.copy_(source_bias)


def _make_te_linear(source: nn.Module, TELinear: type) -> nn.Module:
    bias = getattr(source, "bias", None) is not None
    kwargs = {
        "bias": bias,
        "params_dtype": source.weight.dtype,
    }
    try:
        converted = TELinear(source.in_features, source.out_features, **kwargs)
    except TypeError:
        converted = TELinear(source.in_features, source.out_features, bias=bias)
    converted = converted.to(device=source.weight.device, dtype=source.weight.dtype)
    _copy_linear_parameters(source, converted)
    return converted


def _te_recursive_convert(root: nn.Module, TELinear: type, report) -> None:
    memo: dict[nn.Module, nn.Module] = {}
    visited: set[int] = set()

    def _walk(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module._modules.items()):
            if child is None:
                continue
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.Linear, layer.Linear)):
                if child in memo:
                    _replace_child(module, child_name, memo[child])
                    report.converted_modules.append(child_fqn)
                    continue
                converted = _make_te_linear(child, TELinear)
                wrapped = wrap_float8_linear_module(child, converted)
                memo[child] = wrapped
                _replace_child(module, child_name, wrapped)
                report.converted_modules.append(child_fqn)
            else:
                report.skipped_modules.append(child_fqn)
                child_id = id(child)
                if child_id in visited:
                    continue
                visited.add(child_id)
                _walk(child, child_fqn)

    _walk(root)


class Float8TransformerEnginePolicy(PrecisionPolicy):
    name = "fp8-te"

    def __init__(
        self,
        device_type: str = "cuda",
        strict: str = "warn",
        fp8_recipe: str = "auto",
    ):
        super().__init__()
        self.device_type = device_type
        self.strict = strict
        self.fp8_recipe = fp8_recipe

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": "transformer-engine",
            "device_type": self.device_type,
            "strict": self.strict,
            "fp8_recipe": self.fp8_recipe,
            "autocast": True,
            "grad_scaler": False,
        }

    def check_capability(self, model, device) -> None:
        super().check_capability(model, device)
        self._ensure_model_on_device(model, device)
        if self.fp8_recipe != "auto":
            warnings.warn(
                "fp8_recipe is currently ignored by the Transformer Engine backend "
                "and only kept for future extension.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _ensure_model_on_device(self, model, device) -> None:
        model_devices = {p.device for p in model.parameters()}
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if model_devices and any(d != target_device for d in model_devices):
            raise RuntimeError(
                f"All model parameters must be moved to the target CUDA device "
                f"'{target_device}' (e.g. model.to('{target_device}')) before "
                "calling prepare_model_for_precision() for 'fp8-te'."
            )

    def autocast_context(self):
        try:
            te = _import_te_pytorch()
        except ImportError:
            return nullcontext()
        return te.fp8_autocast(enabled=True)

    def _convert_modules(self, model, report):
        te = _import_te_pytorch()
        TELinear = te.Linear

        if isinstance(model, (nn.Linear, layer.Linear)):
            converted = _make_te_linear(model, TELinear)
            report.converted_modules.append("<root>")
            return wrap_float8_linear_module(model, converted)

        _te_recursive_convert(model, TELinear, report)
        return model


__all__ = ["Float8TransformerEnginePolicy"]
