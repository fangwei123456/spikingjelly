from __future__ import annotations

import torch.nn as nn

from ..dtensor import SNNDistributedAnalysis, analyze_snn_distributed_capability
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime
from .base import build_distributed_runtime


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
        if plan.mode == "pp":
            raise NotImplementedError(
                "Pipeline parallelism ('pp') is not supported by SpikformerAdapter.apply()."
            )
        enable_spikformer_tp = plan.mode in ("tp", "fsdp2_tp")
        enable_experimental_tp = (
            enable_spikformer_tp
            and plan.experimental_features.allow_experimental_spikformer_tp
        )
        num_blocks = len(getattr(model, "blocks", ()))
        fsdp_shard_roots: list[str] | None = None
        fsdp_shard_module_root = True
        if plan.mode in ("fsdp2", "fsdp2_tp"):
            fsdp_shard_roots = ["patch_embed"] + [
                f"blocks.{i}" for i in range(num_blocks)
            ]
            if plan.mode == "fsdp2":
                fsdp_shard_roots.append("head")
            else:
                fsdp_shard_module_root = False
        return build_distributed_runtime(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
            tensor_parallel_roots=["head"] if enable_spikformer_tp else None,
            fsdp_shard_roots=fsdp_shard_roots,
            fsdp_shard_module_root=fsdp_shard_module_root,
            experimental_spikformer_tensor_parallel=enable_experimental_tp,
            spikformer_tensor_parallel_roots=(
                ["blocks"] if enable_experimental_tp else None
            ),
            experimental_spikformer_patch_stem_tensor_parallel=enable_experimental_tp,
            spikformer_patch_stem_tensor_parallel_roots=(
                ["patch_embed"] if enable_experimental_tp else None
            ),
        )
