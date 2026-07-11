import torch.distributed as dist  # noqa: F401

from spikingjelly.activation_based.distributed.analysis import (
    SNNDistributedAnalysis,
    analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.config import SNNDistributedConfig
from spikingjelly.activation_based.distributed.data_parallel import (
    materialize_dtensor_output,  # noqa: F401
    unwrap_parallel_module,  # noqa: F401
)
from spikingjelly.activation_based.distributed.execution import (
    configure_snn_distributed,
)
from spikingjelly.activation_based.distributed.fsdp import (
    FSDP2_AVAILABLE,  # noqa: F401
    _build_fsdp_mp_policy,  # noqa: F401
    apply_snn_fsdp2,  # noqa: F401
    fully_shard_snn_module,  # noqa: F401
)
from spikingjelly.activation_based.distributed.mesh import (
    _resolve_dp_group_from_mesh,  # noqa: F401
    _resolve_mesh_dim_group,  # noqa: F401
    _resolve_mesh_submesh,  # noqa: F401
    build_device_mesh,  # noqa: F401
    ensure_distributed_initialized,  # noqa: F401
    resolve_data_parallel_partition,  # noqa: F401
    resolve_tensor_parallel_group_size,  # noqa: F401
)
from spikingjelly.activation_based.distributed.planner import (
    SNN_DISTRIBUTED_PREFERENCES,  # noqa: F401
    SNNDistributedRecommendation,
    recommend_snn_distributed_strategy,
    recommended_pipeline_microbatches,
)
from spikingjelly.activation_based.distributed.optimizer import (
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,  # noqa: F401
    build_snn_optimizer,  # noqa: F401
)
from spikingjelly.activation_based.distributed.pipeline.memopt import (
    apply_pipeline_stage_memopt,  # noqa: F401
    recommend_pipeline_memopt_stages,  # noqa: F401
)
from spikingjelly.activation_based.distributed.pipeline.partition import (
    parse_pipeline_layout,
    resolve_pipeline_schedule_kind,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    PIPELINING_AVAILABLE,  # noqa: F401
    SNNPipelineRuntime,
    _MicrobatchResetStage,  # noqa: F401
    _PipelineSequentialModule,  # noqa: F401
    _build_snn_pipeline_runtime,  # noqa: F401
    _make_pipeline_outputs_contiguous,  # noqa: F401
    _reset_module_states,  # noqa: F401
    snn_sequence_cross_entropy,  # noqa: F401
)
from spikingjelly.activation_based.distributed.pipeline.cifar10dvs_vgg import (
    _CIFAR10DVSVGGPipelineStage,  # noqa: F401
    _build_cifar10dvs_vgg_pipeline_module,  # noqa: F401
    configure_cifar10dvs_vgg_pipeline,  # noqa: F401
)
from spikingjelly.activation_based.distributed.pipeline.spikformer import (
    _SpikformerPipelineStage,  # noqa: F401
    _build_spikformer_pipeline_module,  # noqa: F401
    configure_spikformer_pipeline,  # noqa: F401
)
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.cifar10dvs_vgg import (
    parallelize_snn_conv_blocks,  # noqa: F401
)
from spikingjelly.activation_based.distributed.tensor_parallel.spikformer import (
    parallelize_spikformer_blocks,  # noqa: F401
    parallelize_spikformer_patch_stem,  # noqa: F401
)
from spikingjelly.activation_based.distributed.tensor_parallel.debug import (
    enable_tp_communication_debug,
    get_tp_communication_debug_stats,
    reset_tp_communication_debug_stats,
)
from spikingjelly.activation_based.distributed.tensor_parallel.linear import (
    TENSOR_PARALLEL_AVAILABLE,  # noqa: F401
    _is_colwise_local_style,  # noqa: F401
    _iter_named_modules_under_roots,  # noqa: F401
    _make_colwise_parallel,  # noqa: F401
    _normalize_parallel_style,  # noqa: F401
    _replace_module_by_name,  # noqa: F401
    auto_build_tensor_parallel_plan,
    parallelize_snn_module,
    wrap_tp_memory_modules,  # noqa: F401
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    make_tensor_shard_memory_module,
)

try:
    from torch.distributed._tensor import DeviceMesh, init_device_mesh

    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

    DTENSOR_AVAILABLE = True
except ImportError:
    DeviceMesh = None
    init_device_mesh = None
    DTensor = None
    DTENSOR_AVAILABLE = False

__all__ = [
    "SNNDistributedConfig",
    "SNNPipelineRuntime",
    "SNNDistributedAnalysis",
    "SNNDistributedRecommendation",
    "make_tensor_shard_memory_module",
    "ChannelShardConv2d",
    "ChannelShardConv1d",
    "ChannelShardBatchNorm2d",
    "ChannelShardBatchNorm1d",
    "parse_pipeline_layout",
    "resolve_pipeline_schedule_kind",
    "recommended_pipeline_microbatches",
    "recommend_snn_distributed_strategy",
    "analyze_snn_distributed_capability",
    "auto_build_tensor_parallel_plan",
    "parallelize_snn_module",
    "configure_snn_distributed",
    "enable_tp_communication_debug",
    "reset_tp_communication_debug_stats",
    "get_tp_communication_debug_stats",
]
