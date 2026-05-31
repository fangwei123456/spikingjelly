from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .. import layer
from .float8_base import wrap_float8_linear_module
from .policy import PrecisionPolicy


def _torchao_available() -> bool:
    try:
        import torchao  # noqa: F401
        return True
    except ImportError:
        return False


def _replace_child(
    module: nn.Module, child_name: str, wrapped: nn.Module
) -> None:
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
            if isinstance(child, (nn.Linear, layer.Linear)):
                if child in memo:
                    _replace_child(module, child_name, memo[child])
                    report.converted_modules.append(child_fqn)
                    continue
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
        strict: str = "warn",
        fp8_recipe: str = "auto",
    ):
        super().__init__()
        self.device_type = device_type
        self.strict = strict
        self.fp8_recipe = fp8_recipe
        self.float8_linear_config = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": "torchao",
            "device_type": self.device_type,
            "strict": self.strict,
            "fp8_recipe": self.fp8_recipe,
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
        self._ensure_torchao_available()
        self._configure_float8()
        self._ensure_cuda_device(device)
        self._ensure_model_on_device(model, device)

    def _ensure_torchao_available(self) -> None:
        if not _torchao_available():
            raise RuntimeError(
                "precision='fp8-torchao' requires torchao, but torchao is not installed."
            )

    def _configure_float8(self) -> None:
        from torchao.float8 import Float8LinearConfig

        if self.fp8_recipe != "auto":
            warnings.warn(
                "fp8_recipe is currently ignored by the torchao backend "
                "and only kept for future extension.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.float8_linear_config = Float8LinearConfig()

    def _ensure_cuda_device(self, device) -> None:
        if self.device_type != "cuda":
            raise RuntimeError(
                "precision='fp8-torchao' is only supported on CUDA in the current stage."
            )
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                "precision='fp8-torchao' requires a CUDA device in the current stage."
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

        _fp8_recursive_convert(model, fp8_config, Float8Linear, report)
        return model
