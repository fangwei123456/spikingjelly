spikingjelly.activation_based.memopt package
=======================================================

本子包提供了减少 ``spikingjelly.activation_based`` 模型训练显存开销的工具。详情请参阅我们在 ICLR 2026 上发表的论文 `Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression <https://openreview.net/forum?id=nrBJ0Uvj7c>`_ 以及 `源代码仓库 <https://github.com/AllenYolk/snn-gradient-checkpointing>`_  。

----

This package provides tools for reducing training memory consumption of ``spikingjelly.activation_based`` models. See our ICLR 2026 paper `Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression <https://openreview.net/forum?id=nrBJ0Uvj7c>`_ and `source code repository <https://github.com/AllenYolk/snn-gradient-checkpointing>`_ for details.

Optimization Pipeline
+++++++++++++++++++++++++++++++++++

基于梯度检查点和脉冲压缩的深度SNN训练显存自动优化工具。

----

Automatic memory optimization pipeline for deep SNN training based on gradient checkpointing and spike compression.

.. list-table::

    * - :func:`memory_optimization <spikingjelly.activation_based.memopt.pipeline.memory_optimization>`
      - **The main API**. Perform memory optimization on a model.
    * - :func:`resolve_device <spikingjelly.activation_based.memopt.pipeline.resolve_device>`
      - Get the device of the current process.
    * - :func:`apply_gc <spikingjelly.activation_based.memopt.pipeline.apply_gc>`
      - Apply GC to a submodule.
    * - :func:`get_module_and_parent <spikingjelly.activation_based.memopt.pipeline.get_module_and_parent>`
      - Get a module and its parent module given its path.

.. toctree::
    :hidden:

    spikingjelly.activation_based.memopt.pipeline

Gradient Checkpointing Tools
+++++++++++++++++++++++++++++++++++

用于实现带输入压缩的梯度检查点 (GC) 的工具。

----

Tools for implementing gradient checkpointing (GC) with input compression.

.. list-table::

    * - :func:`in_gc_1st_forward <spikingjelly.activation_based.memopt.checkpointing.in_gc_1st_forward>`
      - Whether in the first forward pass of GC.
    * - :func:`query_autocast <spikingjelly.activation_based.memopt.checkpointing.query_autocast>`
      - Query autocast information.
    * - :func:`input_compressed_gc <spikingjelly.activation_based.memopt.checkpointing.input_compressed_gc>`
      - Wrap a function with GC and input compression.
    * - :class:`GCContainer <spikingjelly.activation_based.memopt.checkpointing.GCContainer>`
      - Module container representing a GC segment.
    * - :class:`TCGCContainer <spikingjelly.activation_based.memopt.checkpointing.TCGCContainer>`
      - Module container representing a temporally chunked GC segment.

.. toctree::
    :hidden:

    spikingjelly.activation_based.memopt.checkpointing

Spike Compressors
+++++++++++++++++++++++++++++++++++

将浮点数表示的脉冲张量转换为更紧凑的表示形式的压缩器。

----

Compressors that convert spike tensors represented in floating-point numbers into more compact representations.

.. list-table::

    * - :class:`BaseSpikeCompressor <spikingjelly.activation_based.memopt.compress.BaseSpikeCompressor>`
      - Base class for spike compressors.
    * - :class:`NullSpikeCompressor <spikingjelly.activation_based.memopt.compress.NullSpikeCompressor>`
      - Do not perform any compression/decompression.
    * - :class:`BooleanSpikeCompressor <spikingjelly.activation_based.memopt.compress.BooleanSpikeCompressor>`
      - Convert spike tensors to/from boolean tensors.
    * - :class:`Uint8SpikeCompressor <spikingjelly.activation_based.memopt.compress.Uint8SpikeCompressor>`
      - Convert spike tensors to/from ``uint8`` tensors.
    * - :class:`BitSpikeCompressor <spikingjelly.activation_based.memopt.compress.BitSpikeCompressor>`
      - Converts spike tensors to/from bit representations.
    * - :class:`SparseSpikeCompressor <spikingjelly.activation_based.memopt.compress.SparseSpikeCompressor>`
      - Convert spike tensors to/from sparse representations.

.. toctree::
    :hidden:

    spikingjelly.activation_based.memopt.compress
