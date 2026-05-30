from __future__ import annotations

import importlib.util
import warnings

import torch

from .policy import PrecisionPolicy


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
        if importlib.util.find_spec("torchao") is None:
            raise RuntimeError(
                "precision='fp8-torchao' requires torchao, but torchao is not installed."
            )
        from torchao.float8 import Float8LinearConfig

        if self.fp8_recipe != "auto":
            warnings.warn(
                "fp8_recipe is currently ignored by the torchao backend and only kept for future extension.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        if model_devices and any(d != target_device for d in model_devices):
            raise RuntimeError(
                f"All model parameters must be moved to the target CUDA device '{target_device}' "
                f"(e.g. model.to('{target_device}')) before calling "
                "prepare_model_for_precision() for 'fp8-torchao'."
            )
