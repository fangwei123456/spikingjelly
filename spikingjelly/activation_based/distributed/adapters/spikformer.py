from __future__ import annotations

import torch.nn as nn

from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime
from .base import build_distributed_runtime


class SpikformerAdapter:
    name = "spikformer"

    def apply(
        self,
        model: nn.Module,
        plan: SNNDistributedPlan,
        *,
        device_type: str = "cuda",
        device_mesh=None,
    ) -> SNNDistributedRuntime:
        tensor_parallel = plan.mode in ("tp", "fsdp2_tp")
        experimental_tp = (
            tensor_parallel
            and plan.experimental_features.allow_experimental_spikformer_tp
        )
        block_roots = ("patch_embed",) + tuple(
            f"blocks.{i}" for i in range(len(model.blocks))
        )
        fsdp_roots = None
        shard_module_root = True
        if plan.mode == "fsdp2":
            fsdp_roots = block_roots + ("head",)
        elif plan.mode == "fsdp2_tp":
            fsdp_roots = block_roots
            shard_module_root = False
        return build_distributed_runtime(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
            tensor_parallel_roots=("head",) if tensor_parallel else None,
            spikformer_tensor_parallel_roots=("blocks",) if experimental_tp else None,
            spikformer_patch_stem_tensor_parallel_roots=("patch_embed",)
            if experimental_tp
            else None,
            fsdp_shard_roots=fsdp_roots,
            fsdp_shard_module_root=shard_module_root,
        )
