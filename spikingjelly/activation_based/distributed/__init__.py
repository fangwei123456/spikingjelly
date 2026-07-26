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
from .analysis import SNNDistributedAnalysis
from .config import SNNDistributedConfig
from .data_parallel import materialize_dtensor_output, unwrap_parallel_module
from .execution import configure_snn_distributed
from .fsdp import FSDP2_AVAILABLE
from .mesh import (
    DTENSOR_AVAILABLE,
    build_device_mesh,
    ensure_distributed_initialized,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
)
from .optimizer import ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE, build_snn_optimizer
from .pipeline import (
    apply_pipeline_stage_memopt,
    recommend_pipeline_memopt_stages,
)
from .pipeline.runtime import PIPELINING_AVAILABLE
from .planner import (
    DistributedFeatureSet,
    SNN_DISTRIBUTED_PREFERENCES,
    SNNDistributedPlan,
    TensorParallelStyle,
    recommend_snn_distributed_strategy,
    recommended_pipeline_microbatches,
)
from .runtime import SNNDistributedRuntime
from .tensor_parallel import (
    TENSOR_PARALLEL_AVAILABLE,
    enable_tp_communication_debug,
    get_tp_communication_debug_stats,
    make_tensor_shard_memory_module,
    reset_tp_communication_debug_stats,
)
from .topology import SNNDistributedTopology

__all__ = [
    "DTENSOR_AVAILABLE",
    "DistributedFeatureSet",
    "FSDP2_AVAILABLE",
    "PIPELINING_AVAILABLE",
    "SNNDistributedPlan",
    "SNNDistributedConfig",
    "SNN_DISTRIBUTED_PREFERENCES",
    "TENSOR_PARALLEL_AVAILABLE",
    "SNNDistributedAnalysis",
    "SNNDistributedRuntime",
    "SNNDistributedTopology",
    "TensorParallelStyle",
    "make_tensor_shard_memory_module",
    "ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE",
    "analyze",
    "apply",
    "apply_pipeline_stage_memopt",
    "build_snn_optimizer",
    "build_device_mesh",
    "configure_snn_distributed",
    "enable_tp_communication_debug",
    "ensure_distributed_initialized",
    "get_tp_communication_debug_stats",
    "plan",
    "materialize_dtensor_output",
    "recommended_pipeline_microbatches",
    "recommend_snn_distributed_strategy",
    "recommend_pipeline_memopt_stages",
    "reset_tp_communication_debug_stats",
    "resolve_data_parallel_partition",
    "resolve_tensor_parallel_group_size",
    "unwrap_parallel_module",
]
