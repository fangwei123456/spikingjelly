import importlib.util

from .api import PrecisionArtifacts, prepare_model_for_precision, save_precision_reports
from .capability import build_capability_report, validate_capability
from .config import PrecisionConfig
from .convert import (
    ConversionReport,
    analyze_convertible_modules,
    convert_model_for_precision,
)
from .float8_base import Float8LinearStepModule, wrap_float8_linear_module
from .policy import BF16Policy, FP16Policy, FP32Policy, PrecisionPolicy
from .runtime import resolve_precision_policy

__all__ = [
    "PrecisionArtifacts",
    "PrecisionConfig",
    "prepare_model_for_precision",
    "save_precision_reports",
    "build_capability_report",
    "validate_capability",
    "ConversionReport",
    "analyze_convertible_modules",
    "convert_model_for_precision",
    "Float8LinearStepModule",
    "wrap_float8_linear_module",
    "PrecisionPolicy",
    "FP32Policy",
    "FP16Policy",
    "BF16Policy",
    "resolve_precision_policy",
]

if importlib.util.find_spec("torchao") is not None:
    from .float8_torchao import Float8TorchAOPolicy

    __all__.append("Float8TorchAOPolicy")
