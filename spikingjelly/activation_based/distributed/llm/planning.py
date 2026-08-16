"""Static parallel planning for SNN language models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from .config import TrainingConfig


@dataclass(frozen=True)
class _Analysis:
    tensor_sharded_layer_parameters: int
    replicated_layer_parameters: int
    final_norm_parameters: int
    word_embedding_parameters: int
    position_embedding_parameters: int
    output_parameters: int


def _analyze_model(config: TrainingConfig) -> _Analysis:
    transformer = config.model.transformer
    hidden = transformer.hidden_size
    ffn_hidden = transformer.ffn_hidden_size
    heads = transformer.num_attention_heads
    query_groups = transformer.num_query_groups or heads
    query_projection = getattr(transformer, "kv_channels", None) or hidden // heads
    query_projection *= heads
    attention = int(2 * hidden * query_projection * (1 + query_groups / heads))
    mlp = int(2 * hidden * ffn_hidden * (1.5 if transformer.gated_linear_unit else 1))
    norm = 2 * hidden * (1 if transformer.normalization == "RMSNorm" else 2)
    output = hidden * config.model.vocab_size
    position = (
        hidden * config.model.max_sequence_length
        if config.model.position_embedding_type == "learned_absolute"
        else 0
    )
    return _Analysis(attention + mlp, norm, norm // 2, output, position, output)


def _divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def _parameter_memory_bytes(
    config: TrainingConfig,
    analysis: _Analysis,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_sharding_size: int,
) -> float:
    transformer = config.model.transformer
    if transformer.fp16:
        bytes_per_parameter = 4 + 16 / data_parallel_sharding_size
    elif transformer.bf16:
        bytes_per_parameter = 6 + 12 / data_parallel_sharding_size
    else:
        bytes_per_parameter = 8 + 8 / data_parallel_sharding_size

    layers_per_stage = transformer.num_layers / pipeline_parallel_size
    parameters = (
        analysis.tensor_sharded_layer_parameters
        * layers_per_stage
        / tensor_parallel_size
        + analysis.replicated_layer_parameters * layers_per_stage
    )
    input_stage = (
        analysis.word_embedding_parameters / tensor_parallel_size
        + analysis.position_embedding_parameters
    )
    if pipeline_parallel_size == 1:
        parameters += input_stage + analysis.final_norm_parameters
        if not config.model.share_embeddings_and_output_weights:
            parameters += analysis.output_parameters / tensor_parallel_size
    else:
        output_stage = (
            analysis.output_parameters / tensor_parallel_size
            + analysis.final_norm_parameters
        )
        parameters += max(input_stage, output_stage)
    return parameters * bytes_per_parameter


def _activation_memory_bytes(
    config: TrainingConfig,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    selective_recompute: bool,
) -> float:
    transformer = config.model.transformer
    sequence = config.sequence_length / context_parallel_size
    batch = config.micro_batch_size * config.model.time_steps
    hidden = transformer.hidden_size
    layers = transformer.num_layers
    element_scale = 2 if transformer.params_dtype == torch.float32 else 1

    if selective_recompute:
        activation = (
            sequence
            * batch
            * hidden
            * (18 + 4 * transformer.ffn_hidden_size / hidden)
            * layers
            / tensor_parallel_size
        )
        activation += 8 * sequence * batch * pipeline_parallel_size
        activation += sequence * batch * hidden * pipeline_parallel_size
    else:
        activation = (
            sequence * batch * hidden * (10 + 24 / tensor_parallel_size) * layers
        )
        activation += 8 * sequence * batch * pipeline_parallel_size
        activation += sequence * batch * hidden * pipeline_parallel_size
    if pipeline_parallel_size == 1:
        if selective_recompute:
            activation += (
                sequence * batch * hidden * 4 * (1 + config.model.vocab_size / hidden)
            )
        else:
            activation += (
                sequence * batch * config.model.vocab_size / tensor_parallel_size
                + sequence * batch * hidden
            ) * 2
    if not selective_recompute:
        activation *= 1.05
    return activation * element_scale


def _candidate(
    config: TrainingConfig,
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    use_snn_memopt: bool,
    selective_recompute: bool,
) -> TrainingConfig:
    transformer = replace(
        config.model.transformer,
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=tensor_parallel_size > 1,
        expert_tensor_parallel_size=tensor_parallel_size,
        pipeline_dtype=config.model.transformer.params_dtype
        if pipeline_parallel_size > 1
        else config.model.transformer.pipeline_dtype,
        microbatch_group_size_per_vp_stage=pipeline_parallel_size,
        recompute_granularity="selective" if selective_recompute else None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=["core_attn"] if selective_recompute else None,
    )
    return replace(
        config,
        model=replace(config.model, transformer=transformer),
        use_snn_memopt=use_snn_memopt,
    )


def plan_training(
    config: TrainingConfig,
    *,
    world_size: int,
    device_memory_bytes: int,
    objective: Literal["throughput", "memory"] = "throughput",
    memory_fraction: float = 0.9,
) -> TrainingConfig:
    r"""Return a feasible TP/PP/CP and memory policy for ``train``.

    **API Language** - 中文 | English

    **中文：** 使用声明式模型配置和 MCore ``TransformerConfig`` 的静态内存模型，
    枚举当前 ``world_size`` 可行的 TP、PP 和 CP 拓扑，并直接返回可传给
    :func:`train` 的新配置。输入配置不会被修改。规划器优先不使用 MCore
    recompute；显存不足时仅尝试 SpikingJelly memopt 与不重叠的 MCore
    ``core_attn`` selective recompute。不会自动启用 MCore full recompute。

    :param config: 训练配置；其中已有的 TP、PP、CP 和 recompute 策略将被重新规划。
    :type config: TrainingConfig
    :param world_size: 可用 GPU 数量。
    :type world_size: int
    :param device_memory_bytes: 单张 GPU 可用显存字节数。
    :type device_memory_bytes: int
    :param objective: ``"throughput"`` 或 ``"memory"``。
    :type objective: str
    :param memory_fraction: 静态估计最多使用的显存比例。
    :type memory_fraction: float
    :return: 可直接传给 :func:`train` 的新配置。
    :rtype: TrainingConfig
    :raises ValueError: 输入无效，或没有满足拓扑与显存约束的方案。

    **English:** Uses declarative model facts and the MCore
    ``TransformerConfig`` to enumerate feasible TP, PP, and CP topologies for
    ``world_size``, returning a new configuration that can be passed directly to
    :func:`train`. The input is not mutated. The planner first avoids MCore
    recomputation. If memory remains insufficient, it only tries SpikingJelly
    memopt combined with non-overlapping MCore selective ``core_attn``
    recomputation; MCore full recomputation is never selected automatically.

    :param config: Training configuration. Existing TP, PP, CP, and recomputation
        choices are replanned.
    :type config: TrainingConfig
    :param world_size: Number of available GPUs.
    :type world_size: int
    :param device_memory_bytes: Available bytes on each GPU.
    :type device_memory_bytes: int
    :param objective: ``"throughput"`` or ``"memory"``.
    :type objective: str
    :param memory_fraction: Maximum fraction consumed by the static estimate.
    :type memory_fraction: float
    :return: A new configuration accepted directly by :func:`train`.
    :rtype: TrainingConfig
    :raises ValueError: If inputs are invalid or no topology satisfies all constraints.
    """
    if world_size <= 0 or device_memory_bytes <= 0:
        raise ValueError("world_size and device_memory_bytes must be positive.")
    if objective not in {"throughput", "memory"}:
        raise ValueError("objective must be 'throughput' or 'memory'.")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1].")

    transformer = config.model.transformer
    analysis = _analyze_model(config)
    budget = device_memory_bytes * memory_fraction
    topologies: list[tuple[int, int, int, int]] = []
    for tensor_parallel_size in _divisors(world_size):
        if transformer.num_attention_heads % tensor_parallel_size:
            continue
        query_groups = transformer.num_query_groups or transformer.num_attention_heads
        if max(query_groups, tensor_parallel_size) % min(
            query_groups, tensor_parallel_size
        ):
            continue
        for pipeline_parallel_size in _divisors(world_size // tensor_parallel_size):
            if transformer.num_layers % pipeline_parallel_size:
                continue
            for context_parallel_size in _divisors(
                world_size // (tensor_parallel_size * pipeline_parallel_size)
            ):
                model_parallel_size = (
                    tensor_parallel_size
                    * pipeline_parallel_size
                    * context_parallel_size
                )
                data_parallel_size = world_size // model_parallel_size
                if context_parallel_size > 1 and config.sequence_length % (
                    2 * context_parallel_size
                ):
                    continue
                samples_per_microbatch = config.micro_batch_size * data_parallel_size
                if config.global_batch_size % samples_per_microbatch:
                    continue
                num_microbatches = config.global_batch_size // samples_per_microbatch
                if num_microbatches < pipeline_parallel_size:
                    continue
                topologies.append(
                    (
                        tensor_parallel_size,
                        pipeline_parallel_size,
                        context_parallel_size,
                        data_parallel_size,
                    )
                )

    if not topologies:
        raise ValueError("No topology satisfies the model and batch constraints.")

    smallest_estimate = float("inf")
    for selective_recompute in (False, True):
        feasible: list[tuple[float, TrainingConfig, tuple[int, int, int]]] = []
        for tp, pp, cp, dp in topologies:
            if selective_recompute and (transformer.fp8 is not None or cp > 1):
                continue
            estimate = _parameter_memory_bytes(
                config, analysis, tp, pp, dp * cp
            ) + _activation_memory_bytes(
                config,
                tp,
                pp,
                cp,
                selective_recompute,
            )
            smallest_estimate = min(smallest_estimate, estimate)
            if estimate <= budget:
                feasible.append(
                    (
                        estimate,
                        _candidate(
                            config,
                            tensor_parallel_size=tp,
                            pipeline_parallel_size=pp,
                            context_parallel_size=cp,
                            use_snn_memopt=(
                                config.use_snn_memopt
                                or objective == "memory"
                                or selective_recompute
                            ),
                            selective_recompute=selective_recompute,
                        ),
                        (tp, pp, cp),
                    )
                )
        if feasible:
            if objective == "throughput":
                return min(
                    feasible,
                    key=lambda value: (
                        value[2][1],
                        value[2][2],
                        value[2][0],
                        value[0],
                    ),
                )[1]
            return min(
                feasible,
                key=lambda value: (
                    value[0],
                    value[2][1],
                    value[2][2],
                    value[2][0],
                ),
            )[1]

    raise ValueError(
        "No topology fits the memory budget: "
        f"budget={int(budget)} bytes, smallest estimate={int(smallest_estimate)} bytes."
    )


__all__ = ["plan_training"]
