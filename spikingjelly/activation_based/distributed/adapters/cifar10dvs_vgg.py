from __future__ import annotations

import torch.nn as nn

from ..dtensor import SNNDistributedAnalysis, analyze_snn_distributed_capability
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime
from .base import build_distributed_runtime


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
        if plan.mode == "pp":
            raise NotImplementedError(
                "Pipeline parallelism ('pp') is not supported by CIFAR10DVSVGGAdapter.apply()."
            )
        fsdp_shard_roots: list[str] | None = None
        fsdp_shard_module_root = True
        if plan.mode == "fsdp2":
            fsdp_shard_roots = ["features", "classifier"]
        elif plan.mode == "fsdp2_tp":
            fsdp_shard_roots = ["features"]
            fsdp_shard_module_root = False
        analysis_roots = (
            list(plan.tensor_parallel_roots)
            if plan.tensor_parallel_roots is not None
            else ["classifier"]
        )
        enable_conv_tp = (
            plan.mode in ("tp", "fsdp2_tp")
            and plan.experimental_features.allow_experimental_conv_tp
        )
        return build_distributed_runtime(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
            tensor_parallel_roots=analysis_roots,
            fsdp_shard_roots=fsdp_shard_roots,
            fsdp_shard_module_root=fsdp_shard_module_root,
            experimental_conv_tensor_parallel=enable_conv_tp,
            conv_tensor_parallel_roots=["features"] if enable_conv_tp else None,
        )
