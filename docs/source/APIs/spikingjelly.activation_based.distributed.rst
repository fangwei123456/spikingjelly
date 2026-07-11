spikingjelly.activation\_based.distributed package
===========================================================

本子包提供基于 ``torch.distributed``、``DTensor``、tensor parallel 与 FSDP2 的实验性分布式训练工具，面向 ``spikingjelly.activation_based`` 的多步 SNN。

----

This package provides experimental distributed-training helpers for multi-step SNNs in ``spikingjelly.activation_based`` based on ``torch.distributed``, ``DTensor``, tensor parallelism, and FSDP2.

Distributed Helpers
+++++++++++++++++++++++++++++

.. list-table::

    * - :func:`analyze <spikingjelly.activation_based.distributed.analyze>`
      - Analyze an SNN model and find stateful modules and tensor-parallel candidates.
    * - :func:`plan <spikingjelly.activation_based.distributed.plan>`
      - Build a structured distributed plan from analysis, topology, objective, and backend.
    * - :func:`apply <spikingjelly.activation_based.distributed.apply>`
      - Apply a structured plan and return ``SNNDistributedRuntime``.
    * - :class:`SNNDistributedConfig <spikingjelly.activation_based.distributed.dtensor.SNNDistributedConfig>`
      - Low-level compatibility configuration for manual DTensor-ready SNN distribution.
    * - :class:`SNNDistributedAnalysis <spikingjelly.activation_based.distributed.SNNDistributedAnalysis>`
      - Capability analysis for stateful modules and tensor-parallel candidates.
    * - :func:`ensure_distributed_initialized <spikingjelly.activation_based.distributed.ensure_distributed_initialized>`
      - Initialize ``torch.distributed`` when needed.
    * - :func:`build_device_mesh <spikingjelly.activation_based.distributed.build_device_mesh>`
      - Build a ``DeviceMesh`` for tensor/data parallelism.
    * - :func:`configure_snn_distributed <spikingjelly.activation_based.distributed.dtensor.configure_snn_distributed>`
      - Low-level compatibility entry for manual DTensor-ready SNN distribution.
    * - :func:`materialize_dtensor_output <spikingjelly.activation_based.distributed.dtensor.materialize_dtensor_output>`
      - Convert a ``DTensor`` output back to a regular tensor when needed.

.. automodule:: spikingjelly.activation_based.distributed
    :members:
    :undoc-members:
    :show-inheritance:
