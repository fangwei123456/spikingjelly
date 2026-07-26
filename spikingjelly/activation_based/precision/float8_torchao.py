from __future__ import annotations

import torch
import torch.nn as nn

from .. import layer
from .float8_base import wrap_float8_linear_module
from .float8_conv import (
    is_supported_pointwise_conv1d,
    make_linear_from_pointwise_conv1d,
    wrap_float8_pointwise_conv1d_module,
)
from .policy import PrecisionPolicy


def _replace_child(module: nn.Module, child_name: str, wrapped: nn.Module) -> None:
    """Safely replace a child module in *module*."""
    if isinstance(module, nn.ModuleDict):
        module[child_name] = wrapped
    elif isinstance(module, (nn.ModuleList, nn.Sequential)):
        try:
            module[int(child_name)] = wrapped
        except ValueError:
            setattr(module, child_name, wrapped)
    else:
        setattr(module, child_name, wrapped)


def _fp8_recursive_convert(
    root: nn.Module,
    fp8_config: object,
    Float8Linear: type,
    report,
) -> None:
    """Walk *root* and replace every ``nn.Linear`` / ``layer.Linear`` with
    a ``Float8Linear``-wrapped equivalent.

    Parameters are passed explicitly so the function can be tested in
    isolation without depending on closure state.
    """
    memo: dict[nn.Module, nn.Module] = {}
    visited: set[int] = set()

    def _walk(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module._modules.items()):
            if child is None:
                continue
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            is_pointwise_conv1d = is_supported_pointwise_conv1d(child)
            if isinstance(child, (nn.Linear, layer.Linear)) or is_pointwise_conv1d:
                if child in memo:
                    _replace_child(module, child_name, memo[child])
                    report.converted_modules.append(child_fqn)
                    continue
                if is_pointwise_conv1d:
                    source = make_linear_from_pointwise_conv1d(child)
                    converted = Float8Linear.from_float(source, config=fp8_config)
                    wrapped = wrap_float8_pointwise_conv1d_module(child, converted)
                else:
                    converted = Float8Linear.from_float(child, config=fp8_config)
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


class Float8TorchAOPolicy(PrecisionPolicy):
    name = "fp8-torchao"

    def __init__(
        self,
        device_type: str = "cuda",
    ):
        super().__init__()
        self.device_type = device_type
        self.float8_linear_config = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": "torchao",
            "device_type": self.device_type,
            "float8_linear_config": (
                type(self.float8_linear_config).__name__
                if self.float8_linear_config is not None
                else None
            ),
            "autocast": False,
            "grad_scaler": False,
        }

    def check_capability(self, model, device) -> None:
        super().check_capability(model, device)
        from torchao.float8 import Float8LinearConfig

        self.float8_linear_config = Float8LinearConfig()
        if self.device_type != "cuda":
            raise RuntimeError(
                "precision='fp8-torchao' is only supported on CUDA in the current stage."
            )
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                "precision='fp8-torchao' requires a CUDA device in the current stage."
            )
        model_devices = {p.device for p in model.parameters()}
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if model_devices and any(d != target_device for d in model_devices):
            raise RuntimeError(
                f"All model parameters must be moved to the target CUDA device "
                f"'{target_device}' (e.g. model.to('{target_device}')) before "
                "calling prepare_model_for_precision() for 'fp8-torchao'."
            )

    def _convert_modules(self, model, report):
        from torchao.float8.float8_linear import Float8Linear

        fp8_config = self.float8_linear_config
        if fp8_config is None:
            raise RuntimeError(
                "Float8TorchAOPolicy.check_capability() must be called "
                "before convert_model_for_precision()."
            )

        if isinstance(model, (nn.Linear, layer.Linear)):
            converted = Float8Linear.from_float(model, config=fp8_config)
            report.converted_modules.append("<root>")
            return wrap_float8_linear_module(model, converted)
        if is_supported_pointwise_conv1d(model):
            source = make_linear_from_pointwise_conv1d(model)
            converted = Float8Linear.from_float(source, config=fp8_config)
            report.converted_modules.append("<root>")
            return wrap_float8_pointwise_conv1d_module(model, converted)

        _fp8_recursive_convert(model, fp8_config, Float8Linear, report)
        return model
