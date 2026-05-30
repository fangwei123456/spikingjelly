from __future__ import annotations

import torch

from .config import PrecisionConfig
from .float8_torchao import Float8TorchAOPolicy
from .policy import BF16Policy, FP16Policy, FP32Policy


def normalize_precision_mode(config: PrecisionConfig | str | dict | object) -> str:
    return PrecisionConfig.from_any(config).mode.lower()


def resolve_precision_policy(config: PrecisionConfig | str | dict | object):
    cfg = PrecisionConfig.from_any(config)
    mode = cfg.mode.lower()
    device = cfg.device or "cuda"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    if mode == "fp32":
        return FP32Policy()
    if mode == "fp16":
        return FP16Policy(device_type=device_type)
    if mode == "bf16":
        return BF16Policy(device_type=device_type)
    if mode == "fp8-torchao":
        return Float8TorchAOPolicy(
            device_type=device_type,
            strict=cfg.strictness,
        )

    raise ValueError(
        f"Unsupported precision mode {mode!r}. "
        "Supported modes in the current stage are: fp32, fp16, bf16, fp8-torchao."
    )
