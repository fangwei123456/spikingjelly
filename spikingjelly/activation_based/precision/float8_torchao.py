from __future__ import annotations

import importlib.util

from .policy import PrecisionPolicy


class Float8TorchAOPolicy(PrecisionPolicy):
    name = "fp8-torchao"

    def __init__(self, device_type: str = "cuda", strict: str = "warn"):
        super().__init__()
        self.device_type = device_type
        self.strict = strict
        self.float8_linear_config = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": "torchao",
            "device_type": self.device_type,
            "strict": self.strict,
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

        self.float8_linear_config = Float8LinearConfig()
        if self.device_type != "cuda":
            raise RuntimeError(
                "precision='fp8-torchao' is only supported on CUDA in the current stage."
            )
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                "precision='fp8-torchao' requires a CUDA device in the current stage."
            )
