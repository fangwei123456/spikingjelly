from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .topology import SNNDistributedTopology

try:
    from torch.distributed.fsdp import fully_shard

    FSDP2_AVAILABLE = True
except ImportError:
    fully_shard = None
    FSDP2_AVAILABLE = False

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = True
except ImportError:
    ZeroRedundancyOptimizer = None
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = False

try:
    from torch.distributed.pipelining import PipelineStage

    PIPELINING_AVAILABLE = True
except ImportError:
    PipelineStage = None
    PIPELINING_AVAILABLE = False

try:
    from torch.distributed.tensor.parallel import parallelize_module

    TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    parallelize_module = None
    TENSOR_PARALLEL_AVAILABLE = False


SNN_DISTRIBUTED_PREFERENCES = ("speed", "memory", "capacity")


@dataclass(frozen=True)
class DistributedFeatureSet:
    allow_experimental_conv_tp: bool = False
    allow_experimental_spikformer_tp: bool = False
    allow_pipeline: bool = True
    allow_zero_optimizer: bool = True


@dataclass(frozen=True)
class SNNDistributedPlan:
    mode: str
    objective: str
    topology: SNNDistributedTopology
    model_family: str
    backend: str
    batch_size: int
    optimizer_strategy: str
    memopt_level: int
    rationale: Tuple[str, ...]
    notes: Tuple[str, ...]
    tensor_parallel_roots: Optional[Tuple[str, ...]] = None
    mesh_shape: Optional[Tuple[int, ...]] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    pp_microbatches: Optional[int] = None
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 1
    pp_layout: Optional[Tuple[int, ...]] = None
    pp_delay_wgrad: bool = False
    experimental_features: DistributedFeatureSet = DistributedFeatureSet()


@dataclass(frozen=True)
class SNNDistributedRecommendation:
    r"""
    **API Language** - :ref:`中文 <SNNDistributedRecommendation-cn>` | :ref:`English <SNNDistributedRecommendation-en>`

    ----

    .. _SNNDistributedRecommendation-cn:

    * **中文**

    SNN 分布式策略推荐。基于分析结果推荐最优并行配置。

    ----

    .. _SNNDistributedRecommendation-en:

    * **English**

    SNN distributed strategy recommendation.
    """

    prefer: str
    model: str
    world_size: int
    mode: str
    optimizer_sharding: str = "none"
    memopt_level: int = 0
    mesh_shape: Optional[Tuple[int, ...]] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    pp_microbatches: Optional[int] = None
    pp_memopt_stage_budget_ratio: float = 0.5
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 1
    pp_layout: Optional[Tuple[int, ...]] = None
    pp_delay_wgrad: bool = False
    rationale: Tuple[str, ...] = ()


def recommended_pipeline_microbatches(batch_size: int, num_stages: int) -> int:
    r"""
    **API Language** - :ref:`中文 <recommended_pipeline_microbatches-cn>` | :ref:`English <recommended_pipeline_microbatches-en>`

    ----

    .. _recommended_pipeline_microbatches-cn:

    * **中文**

    推荐流水线并行的微批次数量。

    ----

    .. _recommended_pipeline_microbatches-en:

    * **English**

    Recommend microbatches for pipeline parallelism.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, but got {num_stages}.")
    if batch_size < num_stages:
        raise ValueError(
            f"batch_size ({batch_size}) must be >= num_stages ({num_stages}) for pipeline "
            "parallelism with the current microbatch splitting implementation."
        )

    target = min(batch_size, num_stages * 4)
    for candidate in range(target, num_stages - 1, -1):
        if batch_size % candidate == 0:
            return candidate
    return num_stages


def _recommended_fsdp2_tp_mesh_shape(world_size: int) -> Optional[Tuple[int, int]]:
    if world_size < 4 or world_size % 2 != 0:
        return None
    return (world_size // 2, 2)


def recommend_snn_distributed_strategy(
    model: str,
    world_size: int,
    prefer: str,
    batch_size: int,
    backend: str = "inductor",
    zero_redundancy_optimizer_available: Optional[bool] = None,
    pipelining_available: Optional[bool] = None,
    fsdp2_available: Optional[bool] = None,
    tensor_parallel_available: Optional[bool] = None,
) -> SNNDistributedRecommendation:
    r"""
    **API Language** - :ref:`中文 <recommend_snn_distributed_strategy-cn>` | :ref:`English <recommend_snn_distributed_strategy-en>`

    ----

    .. _recommend_snn_distributed_strategy-cn:

    * **中文**

    推荐 SNN 分布式训练策略。

    ----

    .. _recommend_snn_distributed_strategy-en:

    * **English**

    Recommend SNN distributed strategy.
    """
    prefer = prefer.lower()
    if prefer not in SNN_DISTRIBUTED_PREFERENCES:
        raise ValueError(
            f"Unsupported prefer='{prefer}'. Expected one of {SNN_DISTRIBUTED_PREFERENCES}."
        )

    zero_available = (
        ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE
        if zero_redundancy_optimizer_available is None
        else zero_redundancy_optimizer_available
    )
    pipeline_available = (
        PIPELINING_AVAILABLE if pipelining_available is None else pipelining_available
    )
    fsdp_available = FSDP2_AVAILABLE if fsdp2_available is None else fsdp2_available
    tp_available = (
        TENSOR_PARALLEL_AVAILABLE
        if tensor_parallel_available is None
        else tensor_parallel_available
    )

    model_family = "spikformer" if model.startswith("spikformer") else model
    rationale: List[str] = [
        f"prefer='{prefer}' with model='{model_family}', world_size={world_size}, backend='{backend}'."
    ]

    if world_size <= 1:
        if prefer == "speed":
            rationale.append(
                "Single-rank run keeps the simplest local path with no distributed overhead."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="none",
                rationale=tuple(rationale),
            )
        rationale.append(
            "Single-rank run falls back to local training and uses memopt for memory savings."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="none",
            memopt_level=1,
            rationale=tuple(rationale),
        )

    if prefer == "speed":
        if model_family == "cifar10dvs_vgg" and fsdp_available and tp_available:
            mesh_shape = _recommended_fsdp2_tp_mesh_shape(world_size)
            if mesh_shape is not None:
                rationale.append(
                    "Use fsdp2_tp on multi-GPU CIFAR10DVSVGG because current inductor benchmarks show the best global throughput there."
                )
                return SNNDistributedRecommendation(
                    prefer=prefer,
                    model=model,
                    world_size=world_size,
                    mode="fsdp2_tp",
                    mesh_shape=mesh_shape,
                    tp_mesh_dim=1,
                    dp_mesh_dim=0,
                    rationale=tuple(rationale),
                )
        rationale.append(
            "Use data parallel training for the simplest throughput-oriented path, enabling ZeRO optimizer state sharding when available."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="dp",
            optimizer_sharding="zero" if zero_available else "none",
            dp_mesh_dim=0,
            rationale=tuple(rationale),
        )

    if prefer == "memory":
        mesh_shape = _recommended_fsdp2_tp_mesh_shape(world_size)
        if fsdp_available and tp_available and mesh_shape is not None:
            rationale.append(
                "Combine FSDP2 and TP to shard both parameters and activations, and enable memopt level 1 for the strongest memory reduction."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="fsdp2_tp",
                memopt_level=1,
                mesh_shape=mesh_shape,
                tp_mesh_dim=1,
                dp_mesh_dim=0,
                rationale=tuple(rationale),
            )
        if tp_available:
            rationale.append(
                "Use tensor parallel with memopt level 1 when two-dimensional FSDP2+TP is unavailable."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="tp",
                memopt_level=1,
                mesh_shape=(world_size,),
                rationale=tuple(rationale),
            )
        if fsdp_available:
            rationale.append(
                "Fall back to FSDP2 with memopt level 1 when TP is unavailable."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="fsdp2",
                memopt_level=1,
                dp_mesh_dim=0,
                rationale=tuple(rationale),
            )
        rationale.append(
            "Fall back to DP + memopt level 1 because TP/FSDP2 are unavailable."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="dp",
            optimizer_sharding="zero" if zero_available else "none",
            memopt_level=1,
            dp_mesh_dim=0,
            rationale=tuple(rationale),
        )

    if pipeline_available:
        if batch_size >= world_size * 2 and world_size >= 2:
            pp_virtual_stages = 2
        elif batch_size >= world_size:
            pp_virtual_stages = 1
        else:
            pp_virtual_stages = 0
        if pp_virtual_stages == 0:
            rationale.append(
                "Pipeline parallelism is skipped because the global batch is smaller than the number of physical stages."
            )
        else:
            logical_stages = world_size * pp_virtual_stages
            pp_schedule = "interleaved" if pp_virtual_stages > 1 else "1f1b"
            pp_delay_wgrad = False
            rationale.append(
                "Use pipeline parallelism with memopt level 1 when capacity is the priority; prefer the more stable interleaved schedule by default when multiple virtual stages are available."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="pp",
                memopt_level=1,
                pp_microbatches=recommended_pipeline_microbatches(
                    batch_size, logical_stages
                ),
                pp_memopt_stage_budget_ratio=0.5,
                pp_schedule=pp_schedule,
                pp_virtual_stages=pp_virtual_stages,
                pp_delay_wgrad=pp_delay_wgrad,
                rationale=tuple(rationale),
            )

    rationale.append(
        "Pipeline APIs are unavailable, so capacity preference falls back to the strongest memory-oriented strategy."
    )
    fallback = recommend_snn_distributed_strategy(
        model=model,
        world_size=world_size,
        prefer="memory",
        batch_size=batch_size,
        backend=backend,
        zero_redundancy_optimizer_available=zero_available,
        pipelining_available=False,
        fsdp2_available=fsdp_available,
        tensor_parallel_available=tp_available,
    )
    return SNNDistributedRecommendation(
        prefer=prefer,
        model=model,
        world_size=world_size,
        mode=fallback.mode,
        optimizer_sharding=fallback.optimizer_sharding,
        memopt_level=fallback.memopt_level,
        mesh_shape=fallback.mesh_shape,
        tp_mesh_dim=fallback.tp_mesh_dim,
        dp_mesh_dim=fallback.dp_mesh_dim,
        pp_microbatches=fallback.pp_microbatches,
        pp_memopt_stage_budget_ratio=fallback.pp_memopt_stage_budget_ratio,
        pp_schedule=fallback.pp_schedule,
        pp_virtual_stages=fallback.pp_virtual_stages,
        pp_layout=fallback.pp_layout,
        pp_delay_wgrad=fallback.pp_delay_wgrad,
        rationale=tuple(rationale + list(fallback.rationale)),
    )
