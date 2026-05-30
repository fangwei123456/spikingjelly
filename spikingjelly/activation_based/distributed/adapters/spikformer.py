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


class SpikformerAdapter:
    name = "spikformer"

    def analyze(self, model: nn.Module) -> SNNDistributedAnalysis:
        return analyze_snn_distributed_capability(model, tensor_parallel_roots=["head"])

    def apply(
        self,
        model: nn.Module,
        plan: SNNDistributedPlan,
        *,
        device_type: str = "cuda",
        device_mesh=None,
    ) -> SNNDistributedRuntime:
        enable_spikformer_tp = plan.mode in ("tp", "fsdp2_tp")
        enable_experimental_tp = (
            enable_spikformer_tp
            and plan.experimental_features.allow_experimental_spikformer_tp
        )
        num_blocks = len(getattr(model, "blocks", ()))
        fsdp_shard_roots = None
        fsdp_shard_module_root = True
        if plan.mode in ("fsdp2", "fsdp2_tp"):
            fsdp_shard_roots = ["patch_embed"] + [f"blocks.{i}" for i in range(num_blocks)]
            if plan.mode == "fsdp2":
                fsdp_shard_roots.append("head")
            else:
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
            tensor_parallel_roots=["head"] if enable_spikformer_tp else None,
            auto_tensor_parallel=enable_spikformer_tp,
            experimental_spikformer_tensor_parallel=enable_experimental_tp,
            spikformer_tensor_parallel_roots=["blocks"]
            if enable_experimental_tp
            else None,
            experimental_spikformer_patch_stem_tensor_parallel=enable_experimental_tp,
            spikformer_patch_stem_tensor_parallel_roots=["patch_embed"]
            if enable_experimental_tp
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
