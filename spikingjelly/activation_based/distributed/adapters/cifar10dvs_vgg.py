from __future__ import annotations

import torch.nn as nn

from ..analysis import SNNDistributedAnalysis, analyze_snn_distributed_capability
from ..config import EagerParallelPolicy
from ..planner import SNNDistributedPlan
from ..runtime import SNNDistributedRuntime
from .base import build_distributed_runtime


def build_cifar10dvs_vgg_eager_policy(
    tensor_parallel_roots=("classifier",),
) -> EagerParallelPolicy:
    """Build the eager parallel policy for CIFAR10-DVS VGG models.

    .. admonition:: Chinese

        构造 CIFAR10-DVS VGG 模型在 eager 分布式路径中复用的并行策略。

    :param tensor_parallel_roots: Linear roots used by tensor parallelism.
    :type tensor_parallel_roots: sequence[str]
    :return: Eager distributed policy for the model family.
    :rtype: EagerParallelPolicy
    """
    return EagerParallelPolicy(
        linear_tensor_parallel_roots=tuple(tensor_parallel_roots),
        conv_tensor_parallel_roots=("features",),
        fsdp_shard_roots=("features", "classifier"),
        fsdp2_tp_shard_roots=("features",),
        fsdp_shard_module_root=True,
        fsdp2_tp_shard_module_root=False,
    )


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
        analysis_roots = (
            tuple(plan.tensor_parallel_roots)
            if plan.tensor_parallel_roots is not None
            else ("classifier",)
        )
        enable_conv_tp = (
            plan.mode in ("tp", "fsdp2_tp")
            and plan.experimental_features.allow_experimental_conv_tp
        )
        policy = build_cifar10dvs_vgg_eager_policy(analysis_roots)
        return build_distributed_runtime(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
            policy=policy,
            enable_conv_tensor_parallel=enable_conv_tp,
            tensor_parallel_roots=list(analysis_roots),
        )
