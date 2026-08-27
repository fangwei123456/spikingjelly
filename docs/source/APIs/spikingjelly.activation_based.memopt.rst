spikingjelly.activation_based.memopt package
============================================

用于自定义梯度检查点结构和应用论文预设的训练显存优化工具。

Training-memory tools for custom checkpoint structures and the paper preset.

Public API
++++++++++

.. list-table::

    * - :func:`checkpoint <spikingjelly.activation_based.memopt.checkpointing.checkpoint>`
      - Checkpoint one callable and optionally compress its first tensor input.
    * - :func:`checkpoint_module <spikingjelly.activation_based.memopt.checkpointing.checkpoint_module>`
      - Wrap one module transparently, with optional temporal chunks.
    * - :func:`optimize_memory <spikingjelly.activation_based.memopt.pipeline.optimize_memory>`
      - Apply the high-level progressive checkpoint preset.

.. toctree::
    :hidden:

    pipeline <spikingjelly.activation_based.memopt.pipeline>

.. toctree::
    :hidden:

    checkpointing <spikingjelly.activation_based.memopt.checkpointing>

Spike Compressors
+++++++++++++++++++++++++++++++++++

将浮点数表示的脉冲张量转换为更紧凑的表示形式的压缩器。

----

Compressors that convert spike tensors represented in floating-point numbers into more compact representations.

.. list-table::

    * - :class:`SpikeCompressor <spikingjelly.activation_based.memopt.compress.SpikeCompressor>`
      - Structural protocol for stateless compressors.
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

    compress <spikingjelly.activation_based.memopt.compress>
