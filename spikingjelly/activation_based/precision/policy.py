from __future__ import annotations

from contextlib import nullcontext

import torch

from .capability import build_capability_report, validate_capability
from .convert import convert_model_for_precision


class PrecisionPolicy:
    name = "fp32"

    def __init__(self):
        self._capability_report = None
        self._conversion_report = None

    def check_capability(self, model, device) -> None:
        report = build_capability_report(model, device, self.name)
        validate_capability(report)
        self._capability_report = report

    def prepare_model(self, model):
        model, report = convert_model_for_precision(model, self)
        self._conversion_report = report
        return model

    def autocast_context(self):
        return nullcontext()

    def create_grad_scaler(self):
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "autocast": False,
            "grad_scaler": False,
        }

    def capability_report(self) -> dict:
        return self._capability_report or {"requested_mode": self.name}

    def conversion_report(self) -> dict:
        if self._conversion_report is None:
            return {
                "total_modules": 0,
                "convertible_linear": 0,
                "convertible_torch_linear": 0,
                "convertible_modules": [],
                "converted_modules": [],
                "skipped_modules": [],
                "high_precision_modules": [],
                "unsupported_modules": [],
            }
        return self._conversion_report.to_dict()


class FP32Policy(PrecisionPolicy):
    name = "fp32"


class _AutocastPolicy(PrecisionPolicy):
    amp_dtype: torch.dtype
    name: str

    def __init__(self, device_type: str = "cuda"):
        super().__init__()
        self.device_type = device_type

    def autocast_context(self):
        return torch.amp.autocast(self.device_type, dtype=self.amp_dtype)

    def create_grad_scaler(self):
        if self.device_type != "cuda":
            return None
        return torch.cuda.amp.GradScaler()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "autocast": True,
            "device_type": self.device_type,
            "dtype": str(self.amp_dtype),
            "grad_scaler": self.device_type == "cuda",
        }


class FP16Policy(_AutocastPolicy):
    name = "fp16"
    amp_dtype = torch.float16


class BF16Policy(_AutocastPolicy):
    name = "bf16"
    amp_dtype = torch.bfloat16
