from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = True
except ImportError:
    ZeroRedundancyOptimizer = None
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = False


def build_snn_optimizer(
    module: nn.Module,
    mode: str,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_sharding: str = "none",
    foreach: Optional[bool] = None,
    optimizer_cls=torch.optim.Adam,
    **optimizer_kwargs,
):
    if optimizer_sharding not in ("none", "zero"):
        raise ValueError(
            f"Unsupported optimizer_sharding='{optimizer_sharding}'. Expected 'none' or 'zero'."
        )

    if optimizer_sharding == "zero":
        if mode != "dp":
            raise ValueError(
                "optimizer_sharding='zero' is currently supported for pure 'dp' mode only."
            )
        if not dist.is_initialized():
            raise RuntimeError(
                "optimizer_sharding='zero' requires an initialized torch.distributed process group."
            )
        if not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE:
            raise RuntimeError(
                "torch.distributed.optim.ZeroRedundancyOptimizer is unavailable in the current PyTorch build."
            )
        return ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer_cls,
            lr=lr,
            weight_decay=weight_decay,
            foreach=foreach,
            **optimizer_kwargs,
        )

    return optimizer_cls(
        module.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        foreach=foreach,
        **optimizer_kwargs,
    )
