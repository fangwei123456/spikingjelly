"""
**API Language:**
:ref:`中文 <distributed-cn>` | :ref:`English <distributed-en>`

----

.. _distributed-cn:

* **中文**

分布式训练支持模块，包含张量并行和数据并行工具。

:return: None
:rtype: None

----

.. _distributed-en:

* **English**

Distributed training support module with tensor and data parallelism utilities.

:return: None
:rtype: None
"""

from .dtensor import (
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_cifar10dvs_vgg_pipeline,
    configure_spikformer_distributed,
    configure_spikformer_fsdp2,
    configure_spikformer_pipeline,
    configure_snn_distributed,
    DTENSOR_AVAILABLE,
    FSDP2_AVAILABLE,
    PIPELINING_AVAILABLE,
    SNN_DISTRIBUTED_PREFERENCES,
    TENSOR_PARALLEL_AVAILABLE,
    SNNDistributedAnalysis,
    SNNDistributedConfig,
    SNNDistributedRecommendation,
    SNNPipelineRuntime,
    TensorShardMemoryModule,
    analyze_snn_distributed_capability,
    apply_pipeline_stage_memopt,
    apply_snn_fsdp2,
    auto_build_tensor_parallel_plan,
    build_snn_optimizer,
    build_device_mesh,
    ensure_distributed_initialized,
    materialize_dtensor_output,
    parallelize_snn_conv_blocks,
    parallelize_snn_module,
    prepare_snn_data_parallel,
    recommended_pipeline_microbatches,
    recommend_snn_distributed_strategy,
    recommend_pipeline_memopt_stages,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
    unwrap_parallel_module,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
)

__all__ = [
    "DTENSOR_AVAILABLE",
    "FSDP2_AVAILABLE",
    "PIPELINING_AVAILABLE",
    "SNN_DISTRIBUTED_PREFERENCES",
    "TENSOR_PARALLEL_AVAILABLE",
    "SNNDistributedAnalysis",
    "SNNDistributedConfig",
    "SNNDistributedRecommendation",
    "SNNPipelineRuntime",
    "TensorShardMemoryModule",
    "ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE",
    "analyze_snn_distributed_capability",
    "apply_pipeline_stage_memopt",
    "apply_snn_fsdp2",
    "auto_build_tensor_parallel_plan",
    "build_snn_optimizer",
    "build_device_mesh",
    "configure_cifar10dvs_vgg_distributed",
    "configure_cifar10dvs_vgg_fsdp2",
    "configure_cifar10dvs_vgg_pipeline",
    "configure_spikformer_distributed",
    "configure_spikformer_fsdp2",
    "configure_spikformer_pipeline",
    "configure_snn_distributed",
    "ensure_distributed_initialized",
    "materialize_dtensor_output",
    "parallelize_snn_conv_blocks",
    "parallelize_snn_module",
    "prepare_snn_data_parallel",
    "recommended_pipeline_microbatches",
    "recommend_snn_distributed_strategy",
    "recommend_pipeline_memopt_stages",
    "resolve_data_parallel_partition",
    "resolve_tensor_parallel_group_size",
    "unwrap_parallel_module",
]
