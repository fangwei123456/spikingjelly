spikingjelly.activation_based.functional package
==================================================

.. note::

   **API稳定性说明**

   SpikingJelly ``0.0.0.1.0`` 对 ``functional`` 模块的内部实现进行了重构：原来的 ``functional.py`` 文件已被拆分并重组为 ``functional/`` 包，以提升代码的模块化程度和可维护性。

   **该改动不会影响对外公开的 API。** 我们强烈建议用户仍然通过 ``functional`` 这一顶层命名空间来访问相关功能，而不是从具体的内部子模块中进行导入。 ``functional`` 层级下的导入路径被视为稳定的公共接口；更深层的子模块仅作为内部实现细节，未来可能发生变化。

   .. code:: python

      from spikingjelly.activation_based.functional import reset_net # 推荐 ✅
      from spikingjelly.activation_based.functional.net_config import reset_net # 不推荐 ❌

   **API Stability Notice**

   We have refactored the internal implementation of the ``functional`` module. The original ``functional.py`` file has been reorganized into a package ( ``functional/`` ) for better modularity and maintainability.

   **This change does not affect the public API.** Users are strongly encouraged to continue accessing layers directly from the ``functional`` namespace, rather than importing from specific internal submodules. Import paths under ``functional`` are considered part of the stable public API, while deeper submodule paths are treated as implementation details and may change in future releases.

   .. code:: python

      from spikingjelly.activation_based.functional import reset_net # recommended ✅
      from spikingjelly.activation_based.functional.net_config import reset_net # not recommended ❌

Network Configuration Functions
+++++++++++++++++++++++++++++++++++

这些函数帮助用户统一设置网络中每个 **子模块的配置** ，如步进模式、后端等。

----

These functions help users set **configurations for each submodule** in a network, such as step mode and backend.

.. list-table::

   * - :func:`reset_net <spikingjelly.activation_based.functional.net_config.reset_net>`
     - Reset the state of a network.
   * - :func:`set_step_mode <spikingjelly.activation_based.functional.net_config.set_step_mode>`
     - Set the step mode for a network.
   * - :func:`set_backend <spikingjelly.activation_based.functional.net_config.set_backend>`
     - Set the computational backend for a network.
   * - :func:`detach_net <spikingjelly.activation_based.functional.net_config.detach_net>`
     - Detach the network's parameters from the computation graph.

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.net_config

Forward Functions
++++++++++++++++++++++++++

SpikingJelly 的 **前向传播函数** 实现了 SNN 的多步前向传播逻辑。

----

SpikingJelly's **forward functions** provide multi-step forward propagation logic for SNNs.

.. list-table::

   * - :func:`multi_step_forward <spikingjelly.activation_based.functional.forward.multi_step_forward>`
     - Forward pass for stateful modules in multi-step mode.
   * - :func:`t_last_multi_step_forward <spikingjelly.activation_based.functional.forward.t_last_multi_step_forward>`
     - Multi-step forward. The time dimension is placed at the last.
   * - :func:`chunk_multi_step_forward <spikingjelly.activation_based.functional.forward.chunk_multi_step_forward>`
     - Multi-step forward pass with chunked processing.
   * - :func:`seq_to_ann_forward <spikingjelly.activation_based.functional.forward.seq_to_ann_forward>`
     - Forward pass for stateless modules in multi-step mode.
   * - :func:`t_last_seq_to_ann_forward <spikingjelly.activation_based.functional.forward.t_last_seq_to_ann_forward>`
     - Seq-to-ann forward. The time dimension is placed at the last.

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.forward

Loss Functions
+++++++++++++++

适用于 SNN 的 **损失函数** 实现。

----

**Loss functions** suitable for SNNs.

.. list-table::

   * - :func:`kernel_dot_product <spikingjelly.activation_based.functional.loss.kernel_dot_product>`
     - Kernel dot product implementation.
   * - :func:`spike_similar_loss <spikingjelly.activation_based.functional.loss.spike_similar_loss>`
     - Spike similarity loss.
   * - :func:`temporal_efficient_training_cross_entropy <spikingjelly.activation_based.functional.loss.temporal_efficient_training_cross_entropy>`
     - TET loss.

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.loss

Conv-BN Fusion Functions
++++++++++++++++++++++++++++++++

SpikingJelly 提供了将 **卷积层和批归一化层融合** 的工具。这些函数可以计算融合后的权重和偏置，加速卷积 SNN 的推理，使硬件部署成为可能。

----

SpikingJelly provides tools for **fusing convolution with batch normalization**. These functions can compute the fused weights and biases, accelerating the inference process of Convolutional SNNs, and make hardware deployment possible.

.. list-table::

   * - :func:`fused_conv2d_weight_of_convbn2d <spikingjelly.activation_based.functional.conv_bn_fuse.fused_conv2d_weight_of_convbn2d>`
     - Compute fused weight for Conv2d-BN2d fusion.
   * - :func:`fused_conv2d_bias_of_convbn2d <spikingjelly.activation_based.functional.conv_bn_fuse.fused_conv2d_bias_of_convbn2d>`
     - Compute fused bias for Conv2d-BN2d fusion.
   * - :func:`scale_fused_conv2d_weight_of_convbn2d <spikingjelly.activation_based.functional.conv_bn_fuse.scale_fused_conv2d_weight_of_convbn2d>`
     - Scale the fused weight for Conv2d-BN2d fusion.
   * - :func:`scale_fused_conv2d_bias_of_convbn2d <spikingjelly.activation_based.functional.conv_bn_fuse.scale_fused_conv2d_bias_of_convbn2d>`
     - Scale the fused bias for Conv2d-BN2d fusion.
   * - :func:`fuse_convbn2d <spikingjelly.activation_based.functional.conv_bn_fuse.fuse_convbn2d>`
     - Fuse Conv2d-BN2d.

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.conv_bn_fuse

Online Learning Pipelines
+++++++++++++++++++++++++++++++

**在线学习** 的辅助函数。

----

Auxiliary functions for **online learning** .

.. list-table::

   * - :func:`fptt_online_training_init_w_ra <spikingjelly.activation_based.functional.online_learning.fptt_online_training_init_w_ra>`
     - Initialize weight for FPTT.
   * - :func:`fptt_online_training <spikingjelly.activation_based.functional.online_learning.fptt_online_training>`
     - Online training with FPTT.
   * - :func:`ottt_online_training <spikingjelly.activation_based.functional.online_learning.ottt_online_training>`
     - Online training with OTTT or SLTT.

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.online_learning

Miscellaneous
+++++++++++++++++++++

其他辅助 **工具函数** 。

----

Other auxiliary **tool functions** .

.. list-table::

   * - :func:`set_threshold_margin <spikingjelly.activation_based.functional.misc.set_threshold_margin>`
     - Set the threshold margin for classification layers.
   * - :func:`redundant_one_hot <spikingjelly.activation_based.functional.misc.redundant_one_hot>`
     - Convert labels to redundant one-hot encoding.
   * - :func:`first_spike_index <spikingjelly.activation_based.functional.misc.first_spike_index>`
     - Find the index of the first spike in a spike train.
   * - :func:`kaiming_normal_conv_linear_weight <spikingjelly.activation_based.functional.misc.kaiming_normal_conv_linear_weight>`
     - Initialize weights with Kaiming Normal initialization.
   * - :func:`delay <spikingjelly.activation_based.functional.misc.delay>`
     - ``y[t] = x[t - delay_steps]`` .

.. toctree::
   :hidden:

   spikingjelly.activation_based.functional.misc
