"""
**API Language** - :ref:`中文 <distributed-cn>` | :ref:`English <distributed-en>`

----

.. _distributed-cn:

* **中文**

分布式训练支持模块，包含张量并行和数据并行工具。


----

.. _distributed-en:

* **English**

Distributed training support module with tensor and data parallelism utilities.
"""

from .api import analyze, apply, plan
from .dtensor import (
    DTENSOR_AVAILABLE,
    FSDP2_AVAILABLE,
    PIPELINING_AVAILABLE,
    SNN_DISTRIBUTED_PREFERENCES,
    TENSOR_PARALLEL_AVAILABLE,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    SNNDistributedAnalysis,
    TensorShardMemoryModule,
    apply_pipeline_stage_memopt,
    build_device_mesh,
    build_snn_optimizer,
    enable_tp_communication_debug,
    ensure_distributed_initialized,
    get_tp_communication_debug_stats,
    recommend_pipeline_memopt_stages,
    recommend_snn_distributed_strategy,
    recommended_pipeline_microbatches,
    reset_tp_communication_debug_stats,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
    unwrap_parallel_module,
)
from .planner import DistributedFeatureSet, SNNDistributedPlan
from .runtime import SNNDistributedRuntime
from .topology import SNNDistributedTopology

__all__ = [
    "DTENSOR_AVAILABLE",
    "DistributedFeatureSet",
    "FSDP2_AVAILABLE",
    "PIPELINING_AVAILABLE",
    "SNNDistributedPlan",
    "SNN_DISTRIBUTED_PREFERENCES",
    "TENSOR_PARALLEL_AVAILABLE",
    "SNNDistributedAnalysis",
    "SNNDistributedRuntime",
    "SNNDistributedTopology",
    "TensorShardMemoryModule",
    "ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE",
    "analyze",
    "apply",
    "apply_pipeline_stage_memopt",
    "build_snn_optimizer",
    "build_device_mesh",
    "enable_tp_communication_debug",
    "ensure_distributed_initialized",
    "get_tp_communication_debug_stats",
    "plan",
    "recommended_pipeline_microbatches",
    "recommend_snn_distributed_strategy",
    "recommend_pipeline_memopt_stages",
    "reset_tp_communication_debug_stats",
    "resolve_data_parallel_partition",
    "resolve_tensor_parallel_group_size",
    "unwrap_parallel_module",
]
