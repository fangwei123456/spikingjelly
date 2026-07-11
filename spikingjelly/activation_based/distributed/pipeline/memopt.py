import inspect
import time
from typing import Sequence, Tuple

from spikingjelly.activation_based.distributed.pipeline.runtime import (
    SNNPipelineRuntime,
)


def recommend_pipeline_memopt_stages(
    stage_costs: Sequence[float],
    stage_budget_ratio: float = 0.5,
) -> Tuple[int, ...]:
    if not stage_costs:
        return ()
    if stage_budget_ratio <= 0.0 or stage_budget_ratio > 1.0:
        raise ValueError(
            f"stage_budget_ratio must be in (0, 1], but got {stage_budget_ratio}."
        )
    num_stages = len(stage_costs)
    target_count = max(1, min(num_stages, int(round(num_stages * stage_budget_ratio))))
    ranked = sorted(
        range(num_stages),
        key=lambda idx: (float(stage_costs[idx]), -idx),
        reverse=True,
    )
    selected = tuple(sorted(ranked[:target_count]))
    return selected


def apply_pipeline_stage_memopt(
    runtime: SNNPipelineRuntime,
    *,
    memopt_level: int,
    compress_x: bool = False,
    stage_budget_ratio: float = 0.5,
    use_plan_cache: bool = True,
) -> Tuple[SNNPipelineRuntime, float, bool]:
    if memopt_level <= 0:
        runtime.memopt_selected_stage_indices = ()
        return runtime, 0.0, False

    if runtime.model_family == "cifar10dvs_vgg":
        from spikingjelly.activation_based.examples.memopt.models import VGGBlock

        target_types = (VGGBlock,)
    elif runtime.model_family == "spikformer":
        from spikingjelly.activation_based.layer.attention import SpikingSelfAttention
        from spikingjelly.activation_based.model.spikformer import (
            SpikformerConv2dBNLIF,
            SpikformerMLP,
        )

        target_types = (SpikformerConv2dBNLIF, SpikingSelfAttention, SpikformerMLP)
    else:
        raise ValueError(
            f"Unsupported pipeline model_family='{runtime.model_family}' for memopt."
        )

    selected = recommend_pipeline_memopt_stages(
        runtime.stage_costs,
        stage_budget_ratio=stage_budget_ratio,
    )
    runtime.memopt_selected_stage_indices = selected
    local_selected_pairs = [
        (
            logical_idx,
            runtime.stage_modules[local_idx],
            runtime.stage_input_examples[local_idx],
        )
        for local_idx, logical_idx in enumerate(runtime.local_stage_indices)
        if logical_idx in selected
    ]
    if not local_selected_pairs:
        return runtime, 0.0, False

    from spikingjelly.activation_based.memopt import memory_optimization

    supports_plan_cache = (
        "use_plan_cache" in inspect.signature(memory_optimization).parameters
    )

    start = time.time()
    for logical_idx, stage_wrapper, stage_input_example in local_selected_pairs:
        if stage_input_example is None:
            raise RuntimeError(
                f"Pipeline memopt requires a stage_input_example for logical stage {logical_idx}."
            )
        optimize_kwargs = dict(
            dummy_input=(stage_input_example,),
            compress_x=compress_x,
            level=memopt_level,
            verbose=False,
        )
        if supports_plan_cache:
            optimize_kwargs["use_plan_cache"] = use_plan_cache
        optimized = memory_optimization(
            stage_wrapper.inner,
            target_types,
            **optimize_kwargs,
        )
        stage_wrapper.inner = optimized.to(runtime.device)
        stage_wrapper.refresh_reset_modules()
    return runtime, (time.time() - start) * 1000.0, True
