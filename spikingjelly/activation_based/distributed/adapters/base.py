from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch.nn as nn

from ..dtensor import SNNDistributedAnalysis
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime


@runtime_checkable
class SNNDistributedAdapter(Protocol):
    name: str

    def analyze(self, model: nn.Module) -> SNNDistributedAnalysis:
        ...

    def apply(
        self,
        model: nn.Module,
        plan: SNNDistributedPlan,
        *,
        device_type: str = "cuda",
        device_mesh=None,
    ) -> SNNDistributedRuntime:
        ...


def infer_model_family(model: nn.Module) -> Optional[str]:
    if hasattr(model, "module"):
        model = model.module
    class_name = type(model).__name__.lower()
    if "spikformer" in class_name:
        return "spikformer"
    if "cifar10dvsvgg" in class_name:
        return "cifar10dvs_vgg"
    return None
