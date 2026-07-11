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
    """Build an optimizer for an SNN distributed training mode.

    .. admonition:: Chinese

        为 SNN 分布式训练构造优化器，并在纯数据并行模式下可选启用
        ``ZeroRedundancyOptimizer``。

    :param module: Model whose parameters are optimized.
    :type module: torch.nn.Module
    :param mode: Distributed mode, such as ``"dp"``.
    :type mode: str
    :param lr: Learning rate.
    :type lr: float
    :param weight_decay: Weight decay.
    :type weight_decay: float
    :param optimizer_sharding: ``"none"`` or ``"zero"``.
    :type optimizer_sharding: str
    :param foreach: Optional foreach flag passed to the optimizer.
    :type foreach: bool or None
    :param optimizer_cls: Optimizer class to instantiate.
    :return: Optimizer instance.
    """
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
