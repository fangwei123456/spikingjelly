from __future__ import annotations

from contextlib import nullcontext

import torch

from .capability import build_capability_report, validate_capability
from .convert import ConversionReport, convert_model_for_precision


class PrecisionPolicy:
    name = "fp32"

    def __init__(self):
        self._capability_report = None
        self._conversion_report = None

    def set_capability_report(self, report) -> None:
        self._capability_report = report

    def check_capability(self, model, device) -> None:
        report = build_capability_report(model, device, self.name)
        self._capability_report = report
        validate_capability(report)

    def prepare_model(self, model):
        model, report = convert_model_for_precision(model, self)
        self._conversion_report = report
        return model

    def _convert_modules(self, model, report):
        """Subclasses override to transform modules (e.g. float8 substitution)."""
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
        report = self._conversion_report or ConversionReport()
        return report.to_dict()


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
        try:
            return torch.amp.GradScaler("cuda")
        except AttributeError:
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
