from __future__ import annotations

import torch.nn as nn

from ..dtensor import (
    SNNDistributedAnalysis,
    SNNDistributedConfig,
    analyze_snn_distributed_capability,
    configure_snn_distributed,
)
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime


class CIFAR10DVSVGGAdapter:
    name = "cifar10dvs_vgg"

    def analyze(self, model: nn.Module) -> SNNDistributedAnalysis:
        return analyze_snn_distributed_capability(
            model, tensor_parallel_roots=["classifier"]
        )

    def apply(
        self,
        model: nn.Module,
        plan: SNNDistributedPlan,
        *,
        device_type: str = "cuda",
        device_mesh=None,
    ) -> SNNDistributedRuntime:
        fsdp_shard_roots = None
        fsdp_shard_module_root = True
        if plan.mode == "fsdp2":
            fsdp_shard_roots = ["features", "classifier"]
        elif plan.mode == "fsdp2_tp":
            fsdp_shard_roots = ["features"]
            fsdp_shard_module_root = False
        config = SNNDistributedConfig(
            device_type=device_type,
            mesh_shape=plan.mesh_shape or plan.topology.mesh_shape,
            device_mesh=device_mesh,
            tp_mesh_dim=plan.tp_mesh_dim,
            dp_mesh_dim=plan.dp_mesh_dim,
            enable_data_parallel=plan.mode == "dp",
            enable_fsdp2=plan.mode in ("fsdp2", "fsdp2_tp"),
            fsdp_shard_roots=fsdp_shard_roots,
            fsdp_shard_module_root=fsdp_shard_module_root,
            tensor_parallel_roots=["classifier"] if plan.mode in ("tp", "fsdp2_tp") else None,
            auto_tensor_parallel=plan.mode in ("tp", "fsdp2_tp"),
            experimental_conv_tensor_parallel=(
                plan.mode in ("tp", "fsdp2_tp")
                and plan.experimental_features.allow_experimental_conv_tp
            ),
            conv_tensor_parallel_roots=["features"]
            if (
                plan.mode in ("tp", "fsdp2_tp")
                and plan.experimental_features.allow_experimental_conv_tp
            )
            else None,
        )
        configured_model, mesh, analysis = configure_snn_distributed(model, config)
        return SNNDistributedRuntime(
            kind="eager",
            model=configured_model,
            mesh=mesh,
            analysis=analysis,
            plan=plan,
        )
