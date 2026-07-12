from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch.nn as nn

from ..analysis import SNNDistributedAnalysis
from ..execution import build_eager_config, configure_snn_distributed
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime


def build_distributed_runtime(
    model: nn.Module,
    plan: SNNDistributedPlan,
    *,
    device_type: str = "cuda",
    device_mesh=None,
    **config_overrides,
) -> SNNDistributedRuntime:
    if plan.mode == "pp":
        raise ValueError(
            "Pipeline plans require the pipeline-specific runtime builder, "
            "not build_distributed_runtime()."
        )
    config = build_eager_config(
        mode=plan.mode,
        device_type=device_type,
        mesh_shape=plan.mesh_shape or plan.topology.mesh_shape,
        device_mesh=device_mesh,
        tp_mesh_dim=plan.tp_mesh_dim,
        dp_mesh_dim=plan.dp_mesh_dim,
        auto_tensor_parallel=plan.mode in ("tp", "fsdp2_tp"),
        **config_overrides,
    )
    configured_model, mesh, analysis = configure_snn_distributed(model, config)
    return SNNDistributedRuntime(
        kind="eager",
        model=configured_model,
        mesh=mesh,
        analysis=analysis,
        plan=plan,
    )


@runtime_checkable
class SNNDistributedAdapter(Protocol):
    name: str

    def analyze(self, model: nn.Module) -> SNNDistributedAnalysis: ...

    def apply(
        self,
        model: nn.Module,
        plan: SNNDistributedPlan,
        *,
        device_type: str = "cuda",
        device_mesh=None,
    ) -> SNNDistributedRuntime: ...


_FAMILY_PATTERNS = {
    "spikformer": "spikformer",
    "cifar10dvsvgg": "cifar10dvs_vgg",
}


def infer_model_family(model: nn.Module) -> Optional[str]:
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, nn.Module):
        model = wrapped
    class_name = type(model).__name__.lower()
    for pattern, family in _FAMILY_PATTERNS.items():
        if pattern in class_name:
            return family
    return None
