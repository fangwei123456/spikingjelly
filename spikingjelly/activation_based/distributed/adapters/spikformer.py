from __future__ import annotations

import torch.nn as nn

from ..analysis import SNNDistributedAnalysis, analyze_snn_distributed_capability
from ..config import EagerParallelPolicy
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
        fsdp2_tp_shard_roots = tuple(
            ["patch_embed"] + [f"blocks.{i}" for i in range(num_blocks)]
        )
        policy = EagerParallelPolicy(
            linear_tensor_parallel_roots=("head",),
            spikformer_tensor_parallel_roots=("blocks",),
            spikformer_patch_stem_tensor_parallel_roots=("patch_embed",),
            fsdp_shard_roots=fsdp2_tp_shard_roots + ("head",),
            fsdp2_tp_shard_roots=fsdp2_tp_shard_roots,
            fsdp_shard_module_root=True,
            fsdp2_tp_shard_module_root=False,
        )
        return build_distributed_runtime(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
            policy=policy,
            enable_linear_tensor_parallel=enable_spikformer_tp,
            enable_spikformer_tensor_parallel=enable_experimental_tp,
            enable_spikformer_patch_stem_tensor_parallel=enable_experimental_tp,
        )
