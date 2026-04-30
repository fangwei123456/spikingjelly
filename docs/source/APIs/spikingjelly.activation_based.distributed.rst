spikingjelly.activation\_based.distributed package
===========================================================

本子包提供基于 ``torch.distributed``、``DTensor``、tensor parallel 与 FSDP2 的实验性分布式训练工具，面向 ``spikingjelly.activation_based`` 的多步 SNN。

----

This package provides experimental distributed-training helpers for multi-step SNNs in ``spikingjelly.activation_based`` based on ``torch.distributed``, ``DTensor``, tensor parallelism, and FSDP2.

Distributed Helpers
+++++++++++++++++++++++++++++

.. list-table::

    * - :class:`SNNDistributedConfig <spikingjelly.activation_based.distributed.SNNDistributedConfig>`
      - High-level configuration for DTensor-ready SNN distribution.
    * - :class:`SNNDistributedAnalysis <spikingjelly.activation_based.distributed.SNNDistributedAnalysis>`
      - Capability analysis for stateful modules and tensor-parallel candidates.
    * - :func:`ensure_distributed_initialized <spikingjelly.activation_based.distributed.ensure_distributed_initialized>`
      - Initialize ``torch.distributed`` when needed.
    * - :func:`build_device_mesh <spikingjelly.activation_based.distributed.build_device_mesh>`
      - Build a ``DeviceMesh`` for tensor/data parallelism.
    * - :func:`configure_snn_distributed <spikingjelly.activation_based.distributed.configure_snn_distributed>`
      - The main low-level entry for DTensor-ready SNN distribution.
    * - :func:`configure_cifar10dvs_vgg_distributed <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_distributed>`
      - Convenience helper for ``CIFAR10DVSVGG`` with DP / TP.
    * - :func:`configure_cifar10dvs_vgg_fsdp2 <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_fsdp2>`
      - Convenience helper for ``CIFAR10DVSVGG`` with FSDP2 / FSDP2+TP.
    * - :func:`materialize_dtensor_output <spikingjelly.activation_based.distributed.materialize_dtensor_output>`
      - Convert a ``DTensor`` output back to a regular tensor when needed.

.. automodule:: spikingjelly.activation_based.distributed
    :members:
    :undoc-members:
    :show-inheritance:
