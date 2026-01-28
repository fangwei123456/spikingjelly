spikingjelly.activation_based.layer package
=============================================================

.. note::

   **API稳定性说明**

   SpikingJelly ``0.0.0.1.0`` 对 ``layer`` 模块的内部实现进行了重构：原来的 ``layer.py`` 文件已被拆分并重组为 ``layer/`` 包，以提升代码的模块化程度和可维护性。

   **该改动不会影响对外公开的 API。** 我们强烈建议用户仍然通过 ``layer`` 这一顶层命名空间来访问相关功能，而不是从具体的内部子模块中进行导入。 ``layer`` 层级下的导入路径被视为稳定的公共接口；更深层的子模块仅作为内部实现细节，未来可能发生变化。

   .. code:: python

      from spikingjelly.activation_based.layer import SeqToANNContainer # 推荐 ✅
      from spikingjelly.activation_based.layer.container import SeqToANNContainer # 不推荐 ❌

   **API Stability Notice**

   We have refactored the internal implementation of the ``layer`` module. The original ``layer.py`` file has been reorganized into a package ( ``layer/`` ) for better modularity and maintainability.

   **This change does not affect the public API.** Users are strongly encouraged to continue accessing layers directly from the ``layer`` namespace, rather than importing from specific internal submodules. Import paths under ``layer`` are considered part of the stable public API, while deeper submodule paths are treated as implementation details and may change in future releases.

   .. code:: python

      from spikingjelly.activation_based.layer import SeqToANNContainer # recommended ✅
      from spikingjelly.activation_based.layer.container import SeqToANNContainer # not recommended ❌

Containers
++++++++++

SpikingJelly 的 **容器** 封装了常见的网络结构，并支持单步（s）和多步（m）步进模式。

----

SpikingJelly's **containers** wrap common network architectures and support both single-step (s) and multi-step (m) step modes.

.. list-table::

   * - :class:`MultiStepContainer <spikingjelly.activation_based.layer.container.MultiStepContainer>`
     - Container for multi-step forward pass.
   * - :class:`SeqToANNContainer <spikingjelly.activation_based.layer.container.SeqToANNContainer>`
     - Container that merges the time and batch dimensions for forward pass.
   * - :class:`TLastMultiStepContainer <spikingjelly.activation_based.layer.container.TLastMultiStepContainer>`
     - Multi-step container whose time dimension is placed at the last.
   * - :class:`TLastSeqToANNContainer <spikingjelly.activation_based.layer.container.TLastSeqToANNContainer>`
     - SeqToANNContainer whose time dimension is placed at the last.
   * - :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`
     - Container that sets the step mode for all contained modules.
   * - :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`
     - Container for element-wise recurrent connections.
   * - :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
     - Container for linear recurrent connections.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.container

Wrappers for Stateless Layers
+++++++++++++++++++++++++++++

SpikingJelly 的 **无状态层包装器** 封装了 PyTorch 的标准层，并支持脉冲神经网络的时间步进模式。

----

SpikingJelly's **stateless layer wrappers** wrap PyTorch's standard layers and support the step mode of Spiking Neural Networks.

Convolutional Layers
--------------------

.. list-table::

   * - :class:`Conv1d <spikingjelly.activation_based.layer.stateless_wrapper.Conv1d>`
     - 1D convolution layer with step mode support.
   * - :class:`Conv2d <spikingjelly.activation_based.layer.stateless_wrapper.Conv2d>`
     - 2D convolution layer with step mode support.
   * - :class:`Conv3d <spikingjelly.activation_based.layer.stateless_wrapper.Conv3d>`
     - 3D convolution layer with step mode support.

Convolutional Transpose Layers
-------------------------------

.. list-table::

   * - :class:`ConvTranspose1d <spikingjelly.activation_based.layer.stateless_wrapper.ConvTranspose1d>`
     - 1D transposed convolution layer with step mode support.
   * - :class:`ConvTranspose2d <spikingjelly.activation_based.layer.stateless_wrapper.ConvTranspose2d>`
     - 2D transposed convolution layer with step mode support.
   * - :class:`ConvTranspose3d <spikingjelly.activation_based.layer.stateless_wrapper.ConvTranspose3d>`
     - 3D transposed convolution layer with step mode support.

Upsampling Layers
-----------------

.. list-table::

   * - :class:`Upsample <spikingjelly.activation_based.layer.stateless_wrapper.Upsample>`
     - Upsampling layer with configurable interpolation methods.

Pooling Layers
--------------

.. list-table::

   * - :class:`MaxPool1d <spikingjelly.activation_based.layer.stateless_wrapper.MaxPool1d>`
     - 1D max pooling layer with step mode support.
   * - :class:`MaxPool2d <spikingjelly.activation_based.layer.stateless_wrapper.MaxPool2d>`
     - 2D max pooling layer with step mode support.
   * - :class:`MaxPool3d <spikingjelly.activation_based.layer.stateless_wrapper.MaxPool3d>`
     - 3D max pooling layer with step mode support.
   * - :class:`AvgPool1d <spikingjelly.activation_based.layer.stateless_wrapper.AvgPool1d>`
     - 1D average pooling layer with step mode support.
   * - :class:`AvgPool2d <spikingjelly.activation_based.layer.stateless_wrapper.AvgPool2d>`
     - 2D average pooling layer with step mode support.
   * - :class:`AvgPool3d <spikingjelly.activation_based.layer.stateless_wrapper.AvgPool3d>`
     - 3D average pooling layer with step mode support.
   * - :class:`AdaptiveAvgPool1d <spikingjelly.activation_based.layer.stateless_wrapper.AdaptiveAvgPool1d>`
     - 1D adaptive average pooling layer with step mode support.
   * - :class:`AdaptiveAvgPool2d <spikingjelly.activation_based.layer.stateless_wrapper.AdaptiveAvgPool2d>`
     - 2D adaptive average pooling layer with step mode support.
   * - :class:`AdaptiveAvgPool3d <spikingjelly.activation_based.layer.stateless_wrapper.AdaptiveAvgPool3d>`
     - 3D adaptive average pooling layer with step mode support.

Linear Layers
-------------

.. list-table::

   * - :class:`Linear <spikingjelly.activation_based.layer.stateless_wrapper.Linear>`
     - Linear transformation layer with step mode support.
   * - :class:`Flatten <spikingjelly.activation_based.layer.stateless_wrapper.Flatten>`
     - Flatten layer with step mode support.

Layers with Weight Standardization
---------------------------------------

.. list-table::

   * - :class:`WSConv2d <spikingjelly.activation_based.layer.stateless_wrapper.WSConv2d>`
     - Weight Standardization 2D convolution layer with step mode support.
   * - :class:`WSLinear <spikingjelly.activation_based.layer.stateless_wrapper.WSLinear>`
     - Weight Standardization linear layer with step mode support.

Group Normalization Layers
--------------------------

.. list-table::

   * - :class:`GroupNorm <spikingjelly.activation_based.layer.stateless_wrapper.GroupNorm>`
     - Group normalization layer with step mode support.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.stateless_wrapper

Attention Layers
++++++++++++++++

SpikingJelly 的 **注意力层** 提供了用于深度脉冲神经网络的注意力机制实现，包括用于卷积 SNN 的注意力层和用于脉冲 Transformers 的注意力层。

----

SpikingJelly's **attention layers** provide attention mechanisms for deep Spiking Neural Networks, including attention layers for convolutional SNNs and Spiking Transformers.

Attention for Convolutional SNNs
----------------------------------------

.. list-table::

   * - :class:`TemporalWiseAttention <spikingjelly.activation_based.layer.attention.TemporalWiseAttention>`
     - Temporal-wise attention.
   * - :class:`MultiDimensionalAttention <spikingjelly.activation_based.layer.attention.MultiDimensionalAttention>`
     - Multi-dimensional attention.

Attention for Spiking Transformers
------------------------------------------

.. list-table::

   * - :class:`SpikingSelfAttention <spikingjelly.activation_based.layer.attention.SpikingSelfAttention>`
     - Spiking self-attention for Spikformer.
   * - :class:`QKAttention <spikingjelly.activation_based.layer.attention.QKAttention>`
     - Query-Key attention for QKFormer.
   * - :class:`TokenQKAttention <spikingjelly.activation_based.layer.attention.TokenQKAttention>`
     - Token-wise Query-Key attention for QKFormer.
   * - :class:`ChannelQKAttention <spikingjelly.activation_based.layer.attention.ChannelQKAttention>`
     - Channel-wise Query-Key attention for QKFormer.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.attention

Batch Normalization Variants
++++++++++++++++++++++++++++

SpikingJelly 提供了多种适用于深度 SNN 的 **批归一化层变体** 。

----

SpikingJelly provides multiple **batch normalization variants** that are optimized for deep SNNs.

Standard Batch Normalization
-----------------------------

.. list-table::

   * - :class:`BatchNorm1d <spikingjelly.activation_based.layer.bn.BatchNorm1d>`
     - 1D batch normalization layer with step mode support.
   * - :class:`BatchNorm2d <spikingjelly.activation_based.layer.bn.BatchNorm2d>`
     - 2D batch normalization layer with step mode support.
   * - :class:`BatchNorm3d <spikingjelly.activation_based.layer.bn.BatchNorm3d>`
     - 3D batch normalization layer with step mode support.

NeuNorm
-------

.. list-table::

   * - :class:`NeuNorm <spikingjelly.activation_based.layer.bn.NeuNorm>`
     - Neural Normalization layer.

Threshold-Dependent Batch Normalization
----------------------------------------

.. list-table::

   * - :class:`ThresholdDependentBatchNorm1d <spikingjelly.activation_based.layer.bn.ThresholdDependentBatchNorm1d>`
     - 1D threshold-dependent batch normalization layer.
   * - :class:`ThresholdDependentBatchNorm2d <spikingjelly.activation_based.layer.bn.ThresholdDependentBatchNorm2d>`
     - 2D threshold-dependent batch normalization layer.
   * - :class:`ThresholdDependentBatchNorm3d <spikingjelly.activation_based.layer.bn.ThresholdDependentBatchNorm3d>`
     - 3D threshold-dependent batch normalization layer.

Temporal Effective Batch Normalization
--------------------------------------

.. list-table::

   * - :class:`TemporalEffectiveBatchNorm1d <spikingjelly.activation_based.layer.bn.TemporalEffectiveBatchNorm1d>`
     - 1D temporal effective batch normalization layer.
   * - :class:`TemporalEffectiveBatchNorm2d <spikingjelly.activation_based.layer.bn.TemporalEffectiveBatchNorm2d>`
     - 2D temporal effective batch normalization layer.
   * - :class:`TemporalEffectiveBatchNorm3d <spikingjelly.activation_based.layer.bn.TemporalEffectiveBatchNorm3d>`
     - 3D temporal effective batch normalization layer.

Batch Normalization Through Time
------------------------------------

.. list-table::

   * - :class:`BatchNormThroughTime1d <spikingjelly.activation_based.layer.bn.BatchNormThroughTime1d>`
     - 1D batch normalization through time.
   * - :class:`BatchNormThroughTime2d <spikingjelly.activation_based.layer.bn.BatchNormThroughTime2d>`
     - 2D batch normalization through time.
   * - :class:`BatchNormThroughTime3d <spikingjelly.activation_based.layer.bn.BatchNormThroughTime3d>`
     - 3D batch normalization through time.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.bn

Dropout Variants
++++++++++++++++

SpikingJelly 提供了适用于 SNN 的 **Dropout 实现**，支持步进模式。

----

SpikingJelly provides **dropout implementations** suitable for SNNs with step mode support.


.. list-table::

   * - :class:`Dropout <spikingjelly.activation_based.layer.dropout.Dropout>`
     - Dropout layer with step mode support.
   * - :class:`Dropout2d <spikingjelly.activation_based.layer.dropout.Dropout2d>`
     - 2D dropout layer with step mode support.
   * - :class:`DropConnectLinear <spikingjelly.activation_based.layer.dropout.DropConnectLinear>`
     - DropConnect linear layer with step mode support.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.dropout

Online Learning Modules
+++++++++++++++++++++++

SpikingJelly 的 **在线学习模块** 提供了用于在线学习的辅助类和操作。

----

SpikingJelly's **online learning modules** provide auxiliary classes and operations for online learning.

.. list-table::

   * - :class:`ReplaceforGrad <spikingjelly.activation_based.layer.online_learning.ReplaceforGrad>`
     - AutoGrad function for gradient replacement.
   * - :class:`GradwithTrace <spikingjelly.activation_based.layer.online_learning.GradwithTrace>`
     - Computing gradients using neuron traces.
   * - :class:`SpikeTraceOp <spikingjelly.activation_based.layer.online_learning.SpikeTraceOp>`
     - Operation for spike and trace handling in online training.
   * - :class:`OTTTSequential <spikingjelly.activation_based.layer.online_learning.OTTTSequential>`
     - Sequential container for online training through time.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.online_learning

Miscellaneous
+++++++++++++

SpikingJelly 的 **杂项模块** 提供了辅助层和其他实用工具。

----

SpikingJelly's **miscellaneous module** provides auxiliary layers and other utilities.

.. list-table::

   * - :class:`SynapseFilter <spikingjelly.activation_based.layer.misc.SynapseFilter>`
     - Synapse filter layer with exponential decay.
   * - :class:`PrintShapeModule <spikingjelly.activation_based.layer.misc.PrintShapeModule>`
     - Module for printing input shapes during forward pass.
   * - :class:`VotingLayer <spikingjelly.activation_based.layer.misc.VotingLayer>`
     - Voting layer for multi-class classification.
   * - :class:`Delay <spikingjelly.activation_based.layer.misc.Delay>`
     - Temporal delay operation.

.. toctree::
   :hidden:

   spikingjelly.activation_based.layer.misc
