spikingjelly.activation_based.functional package
==================================================

.. warning::

  卷积-批归一化融合函数已弃用。用户可以使用 `PyTorch的fuse_conv_bn_eval <https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html>`_ 来实现相同的功能。

  Functions for conv-bn fusion have been deprecated. Use `PyTorch's fuse_conv_bn_eval <https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html>`_ to achieve the same functionality.

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

   net_config <spikingjelly.activation_based.functional.net_config>

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

   forward <spikingjelly.activation_based.functional.forward>

Neuron State Updates
++++++++++++++++++++

这些函数显式接收并返回神经元状态。``*_step`` 表示一个时间步的完整更新；
``*_multi_step`` 表示具有独立实现的时间序列更新，而不是对 ``*_step`` 的 Python
循环包装。backend 仍由 ``MemoryModule`` 选择，因此 backend 专用函数在名称中
标出 ``cupy`` 或 ``triton``。

----

These functions receive and return neuron states explicitly. ``*_step`` denotes
one complete time-step update. ``*_multi_step`` denotes an independently implemented
sequence update, not a Python-loop wrapper around ``*_step``. Backend selection
remains a ``MemoryModule`` responsibility, so backend-specific functions identify
``cupy`` or ``triton`` in their names.

.. list-table::

   * - :func:`if_step <spikingjelly.activation_based.functional.neuron.if_step>`
     - One IF state update.
   * - :func:`qif_step <spikingjelly.activation_based.functional.neuron.qif_step>`
     - One QIF state update.
   * - :func:`eif_step <spikingjelly.activation_based.functional.neuron.eif_step>`
     - One EIF state update.
   * - :func:`lif_step <spikingjelly.activation_based.functional.neuron.lif_step>`
     - One LIF state update.
   * - :func:`plif_step <spikingjelly.activation_based.functional.neuron.plif_step>`
     - One ParametricLIF state update.
   * - :func:`izhikevich_step <spikingjelly.activation_based.functional.neuron.izhikevich_step>`
     - One Izhikevich voltage and adaptation-current update.
   * - :func:`klif_step <spikingjelly.activation_based.functional.neuron.klif_step>`
     - One KLIF state update.
   * - :func:`cuba_lif_step <spikingjelly.activation_based.functional.neuron.cuba_lif_step>`
     - One current-based LIF state update.
   * - :func:`lava_cuba_lif_step <spikingjelly.activation_based.functional.neuron.lava_cuba_lif_step>`
     - One Lava-compatible quantized CUBA-LIF state update.
   * - :func:`activation_aware_if_step <spikingjelly.activation_based.functional.neuron.activation_aware_if_step>`
     - One ActivationAwareIF state update.
   * - :func:`sliding_psn_step <spikingjelly.activation_based.functional.neuron.sliding_psn_step>`
     - One SlidingPSN queue update.
   * - :func:`gated_lif_step <spikingjelly.activation_based.functional.neuron.gated_lif_step>`
     - One GatedLIF state update.
   * - :func:`stbif_step <spikingjelly.activation_based.functional.neuron.stbif_step>`
     - One SpikeZIP STBIF state update.
   * - :func:`if_step_cupy <spikingjelly.activation_based.functional.neuron.if_step_cupy>`
     - One IF update with caller-selected CuPy kernels.
   * - :func:`lif_step_cupy <spikingjelly.activation_based.functional.neuron.lif_step_cupy>`
     - One LIF update with caller-selected CuPy kernels.
   * - :func:`if_multi_step_cupy <spikingjelly.activation_based.functional.neuron.if_multi_step_cupy>`
     - IF sequence update with CuPy.
   * - :func:`lif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.lif_multi_step_cupy>`
     - LIF sequence update with CuPy.
   * - :func:`plif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.plif_multi_step_cupy>`
     - ParametricLIF sequence update with CuPy.
   * - :func:`qif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.qif_multi_step_cupy>`
     - QIF sequence update with CuPy.
   * - :func:`eif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.eif_multi_step_cupy>`
     - EIF sequence update with CuPy.
   * - :func:`izhikevich_multi_step_cupy <spikingjelly.activation_based.functional.neuron.izhikevich_multi_step_cupy>`
     - Izhikevich sequence update with CuPy.
   * - :func:`if_multi_step_triton <spikingjelly.activation_based.functional.neuron.if_multi_step_triton>`
     - IF sequence update with Triton.
   * - :func:`lif_multi_step_triton <spikingjelly.activation_based.functional.neuron.lif_multi_step_triton>`
     - LIF sequence update with Triton.
   * - :func:`plif_multi_step_triton <spikingjelly.activation_based.functional.neuron.plif_multi_step_triton>`
     - ParametricLIF sequence update with Triton.
   * - :func:`activation_aware_if_multi_step_triton <spikingjelly.activation_based.functional.neuron.activation_aware_if_multi_step_triton>`
     - ActivationAwareIF sequence update with Triton.
.. toctree::
   :hidden:

   neuron <spikingjelly.activation_based.functional.neuron>

Stateful Layer Updates
++++++++++++++++++++++

这些函数显式接收并返回 stateful layer 的局部状态，不读取 ``MemoryModule`` 的隐式
memory。

----

These functions receive and return local state for stateful layers explicitly.
They do not read implicit ``MemoryModule`` memory.

.. list-table::

   * - :func:`delay_step <spikingjelly.activation_based.functional.layer.delay_step>`
     - One Delay queue update.
   * - :func:`synapse_filter_step <spikingjelly.activation_based.functional.layer.synapse_filter_step>`
     - One SynapseFilter state update.

.. toctree::
   :hidden:

   layer <spikingjelly.activation_based.functional.layer>

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

   loss <spikingjelly.activation_based.functional.loss>

Learning State Transition Functions
+++++++++++++++++++++++++++++++++++

这些函数显式接收 STDP/mSTDP/mSTDP-ET 的 trace、eligibility、reward 和 raw
权重 tensor，不读取 monitor、``MemoryModule`` memory，也不负责 ``step_mode``、
``training/eval`` 或梯度写入。

----

These functions receive STDP/mSTDP/mSTDP-ET traces, eligibility, reward, and
raw weight tensors explicitly. They do not read monitors or ``MemoryModule``
memory, and do not manage ``step_mode``, ``training/eval``, or gradient writes.

.. list-table::

   * - :func:`stdp_linear_step <spikingjelly.activation_based.functional.learning.stdp_linear_step>`
     - Tensor-only linear STDP single-step update.
   * - :func:`stdp_conv1d_step <spikingjelly.activation_based.functional.learning.stdp_conv1d_step>`
     - Tensor-only Conv1d STDP single-step update.
   * - :func:`stdp_conv2d_step <spikingjelly.activation_based.functional.learning.stdp_conv2d_step>`
     - Tensor-only Conv2d STDP single-step update.
   * - :func:`mstdp_linear_step <spikingjelly.activation_based.functional.learning.mstdp_linear_step>`
     - Tensor-only linear mSTDP eligibility update.
   * - :func:`mstdpet_linear_step <spikingjelly.activation_based.functional.learning.mstdpet_linear_step>`
     - Tensor-only linear mSTDP-ET eligibility update.
   * - :func:`mstdpet_reward_step <spikingjelly.activation_based.functional.learning.mstdpet_reward_step>`
     - Eligibility-trace decay and reward modulation for mSTDP-ET.

.. toctree::
   :hidden:

   learning <spikingjelly.activation_based.functional.learning>

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

   online_learning <spikingjelly.activation_based.functional.online_learning>

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

   misc <spikingjelly.activation_based.functional.misc>
