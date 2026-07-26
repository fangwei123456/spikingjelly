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

Neuron State Transition Functions
++++++++++++++++++++++++++++++++++++++++

这些函数显式接收并返回神经元状态，不读取 ``MemoryModule`` 的隐式 memory，
也不负责 ``training/eval`` 或 backend dispatch。

----

These functions receive and return neuron states explicitly. They do not read
implicit ``MemoryModule`` memory and do not handle ``training/eval`` or backend
dispatch.

.. list-table::

   * - :func:`neuron_fire <spikingjelly.activation_based.functional.neuron.neuron_fire>`
     - Compute spikes from membrane voltage.
   * - :func:`hard_reset <spikingjelly.activation_based.functional.neuron.hard_reset>`
     - Apply hard reset to membrane voltage.
   * - :func:`soft_reset <spikingjelly.activation_based.functional.neuron.soft_reset>`
     - Apply soft reset to membrane voltage.
   * - :func:`if_single_step <spikingjelly.activation_based.functional.neuron.if_single_step>`
     - Single-step IF state transition.
   * - :func:`if_multi_step <spikingjelly.activation_based.functional.neuron.if_multi_step>`
     - Multi-step IF state transition.
   * - :func:`if_multi_step_inductor <spikingjelly.activation_based.functional.neuron.if_multi_step_inductor>`
     - Multi-step IF state transition for the Inductor backend.
   * - :func:`if_single_step_cupy <spikingjelly.activation_based.functional.neuron.if_single_step_cupy>`
     - Single-step IF state transition for caller-selected CuPy kernels.
   * - :func:`if_multi_step_cupy <spikingjelly.activation_based.functional.neuron.if_multi_step_cupy>`
     - Multi-step IF state transition for the CuPy backend.
   * - :func:`lif_single_step <spikingjelly.activation_based.functional.neuron.lif_single_step>`
     - Single-step LIF state transition.
   * - :func:`lif_multi_step <spikingjelly.activation_based.functional.neuron.lif_multi_step>`
     - Multi-step LIF state transition.
   * - :func:`lif_multi_step_inductor <spikingjelly.activation_based.functional.neuron.lif_multi_step_inductor>`
     - Multi-step LIF state transition for the Inductor backend.
   * - :func:`lif_single_step_cupy <spikingjelly.activation_based.functional.neuron.lif_single_step_cupy>`
     - Single-step LIF state transition for caller-selected CuPy kernels.
   * - :func:`lif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.lif_multi_step_cupy>`
     - Multi-step LIF state transition for the CuPy backend.
   * - :func:`lif_single_step_with_pre_spike_mean <spikingjelly.activation_based.functional.neuron.lif_single_step_with_pre_spike_mean>`
     - Single-step LIF transition with pre-spike membrane mean observation.
   * - :func:`lif_multi_step_with_pre_spike_mean <spikingjelly.activation_based.functional.neuron.lif_multi_step_with_pre_spike_mean>`
     - Multi-step LIF transition with optional pre-spike membrane mean sequence.
   * - :func:`plif_single_step <spikingjelly.activation_based.functional.neuron.plif_single_step>`
     - Single-step ParametricLIF state transition.
   * - :func:`plif_multi_step <spikingjelly.activation_based.functional.neuron.plif_multi_step>`
     - Multi-step ParametricLIF state transition.
   * - :func:`plif_multi_step_inductor <spikingjelly.activation_based.functional.neuron.plif_multi_step_inductor>`
     - Multi-step ParametricLIF state transition for the Inductor backend.
   * - :func:`plif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.plif_multi_step_cupy>`
     - Multi-step ParametricLIF state transition for the CuPy backend.
   * - :func:`qif_charge <spikingjelly.activation_based.functional.neuron.qif_charge>`
     - QIF charge equation.
   * - :func:`eif_charge <spikingjelly.activation_based.functional.neuron.eif_charge>`
     - EIF charge equation.
   * - :func:`qif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.qif_multi_step_cupy>`
     - Multi-step QIF state transition for the CuPy backend.
   * - :func:`eif_multi_step_cupy <spikingjelly.activation_based.functional.neuron.eif_multi_step_cupy>`
     - Multi-step EIF state transition for the CuPy backend.
   * - :func:`adaptive_current_update <spikingjelly.activation_based.functional.neuron.adaptive_current_update>`
     - Adaptation-current update equation.
   * - :func:`adaptive_reset <spikingjelly.activation_based.functional.neuron.adaptive_reset>`
     - Reset equation for neurons with adaptation current.
   * - :func:`izhikevich_charge <spikingjelly.activation_based.functional.neuron.izhikevich_charge>`
     - Izhikevich charge equation.
   * - :func:`izhikevich_multi_step_cupy <spikingjelly.activation_based.functional.neuron.izhikevich_multi_step_cupy>`
     - Multi-step Izhikevich state transition for the CuPy backend.
   * - :func:`klif_charge <spikingjelly.activation_based.functional.neuron.klif_charge>`
     - KLIF charge equation.
   * - :func:`klif_reset <spikingjelly.activation_based.functional.neuron.klif_reset>`
     - KLIF reset equation.
   * - :func:`cuba_lif_charge <spikingjelly.activation_based.functional.neuron.cuba_lif_charge>`
     - CUBA-LIF current and membrane charge equation.
   * - :func:`lava_cuba_lif_charge <spikingjelly.activation_based.functional.neuron.lava_cuba_lif_charge>`
     - Lava-compatible quantized CUBA-LIF charge equation.
   * - :func:`lava_cuba_lif_single_step <spikingjelly.activation_based.functional.neuron.lava_cuba_lif_single_step>`
     - Single-step Lava-compatible CUBA-LIF state transition.
   * - :func:`lava_cuba_lif_multi_step <spikingjelly.activation_based.functional.neuron.lava_cuba_lif_multi_step>`
     - Multi-step Lava-compatible CUBA-LIF state transition.
   * - :func:`liaf_output <spikingjelly.activation_based.functional.neuron.liaf_output>`
     - LIAF analog output equation.
   * - :func:`mpbn_fire <spikingjelly.activation_based.functional.neuron.mpbn_fire>`
     - MPBN firing and residual-normalization path.
   * - :func:`online_lif_charge <spikingjelly.activation_based.functional.neuron.online_lif_charge>`
     - LIF charge equation for OTTT/SLTT training paths with detached previous voltage.
   * - :func:`ottt_trace_update <spikingjelly.activation_based.functional.neuron.ottt_trace_update>`
     - OTTT trace update under ``torch.no_grad()``.
   * - :func:`activation_aware_if_single_step <spikingjelly.activation_based.functional.neuron.activation_aware_if_single_step>`
     - Single-step ActivationAwareIF state transition.
   * - :func:`activation_aware_if_multi_step <spikingjelly.activation_based.functional.neuron.activation_aware_if_multi_step>`
     - Multi-step ActivationAwareIF state transition.
   * - :func:`activation_aware_if_multi_step_triton <spikingjelly.activation_based.functional.neuron.activation_aware_if_multi_step_triton>`
     - Multi-step ActivationAwareIF state transition for the Triton backend.
   * - :func:`masked_psn_advance_queue <spikingjelly.activation_based.functional.neuron.masked_psn_advance_queue>`
     - Single-step MaskedPSN queue advancement.
   * - :func:`masked_psn_single_step_from_queue <spikingjelly.activation_based.functional.neuron.masked_psn_single_step_from_queue>`
     - Single-step MaskedPSN spike computation from an already advanced queue.
   * - :func:`sliding_psn_single_step <spikingjelly.activation_based.functional.neuron.sliding_psn_single_step>`
     - Single-step SlidingPSN queue state transition.
   * - :func:`gated_lif_multi_step <spikingjelly.activation_based.functional.neuron.gated_lif_multi_step>`
     - Multi-step GatedLIF state transition.
   * - :func:`stbif_single_step <spikingjelly.activation_based.functional.neuron.stbif_single_step>`
     - Single-step SpikeZIP STBIF state transition.
   * - :func:`stbif_multi_step_torch <spikingjelly.activation_based.functional.neuron.stbif_multi_step_torch>`
     - Torch multi-step SpikeZIP STBIF state transition.

.. toctree::
   :hidden:

   neuron <spikingjelly.activation_based.functional.neuron>

Stateful Layer State Transition Functions
+++++++++++++++++++++++++++++++++++++++++

这些函数显式接收并返回 stateful layer 的局部状态，不读取 ``MemoryModule`` 的隐式
memory。

----

These functions receive and return local state for stateful layers explicitly.
They do not read implicit ``MemoryModule`` memory.

.. list-table::

   * - :func:`delay_single_step <spikingjelly.activation_based.functional.layer.delay_single_step>`
     - Single-step Delay queue state transition.
   * - :func:`element_wise_recurrent_single_step <spikingjelly.activation_based.functional.layer.element_wise_recurrent_single_step>`
     - Single-step ElementWiseRecurrentContainer state transition.
   * - :func:`linear_recurrent_single_step <spikingjelly.activation_based.functional.layer.linear_recurrent_single_step>`
     - Single-step LinearRecurrentContainer state transition.
   * - :func:`batch_norm_through_time_single_step <spikingjelly.activation_based.functional.layer.batch_norm_through_time_single_step>`
     - Single-step BatchNormThroughTime time-state transition.
   * - :func:`neunorm_single_step <spikingjelly.activation_based.functional.layer.neunorm_single_step>`
     - Single-step NeuNorm state transition.
   * - :func:`synapse_filter_single_step <spikingjelly.activation_based.functional.layer.synapse_filter_single_step>`
     - Single-step SynapseFilter state transition.

.. toctree::
   :hidden:

   layer <spikingjelly.activation_based.functional.layer>

Encoder State Transition Functions
+++++++++++++++++++++++++++++++++++

这些函数显式接收并返回 encoder 的局部状态，不读取 ``MemoryModule`` 的隐式 memory。

----

These functions receive and return local encoder state explicitly. They do not
read implicit ``MemoryModule`` memory.

.. list-table::

   * - :func:`latency_encode <spikingjelly.activation_based.functional.encoding.latency_encode>`
     - Stateless LatencyEncoder spike-sequence generation.
   * - :func:`stateful_encoder_single_step <spikingjelly.activation_based.functional.encoding.stateful_encoder_single_step>`
     - Single-step StatefulEncoder time-state transition.
   * - :func:`weighted_phase_encode <spikingjelly.activation_based.functional.encoding.weighted_phase_encode>`
     - Stateless WeightedPhaseEncoder spike-sequence generation.

.. toctree::
   :hidden:

   encoding <spikingjelly.activation_based.functional.encoding>

ANN-to-SNN Functional Helpers
+++++++++++++++++++++++++++++

这些函数实现 STA 和 SpikeZIP 中具有独立状态转移语义的叶子运算。TD operator
本身是围绕可替换 ``ann_forward`` 的状态适配器，不为其 ANN 数值路径提供冗余的
函数式包装。

----

These functions implement leaf operations with independent state-transition
semantics in STA and SpikeZIP. TD operators are state adapters around a
replaceable ``ann_forward`` and therefore do not expose redundant functional
wrappers for their ANN numeric paths.

.. list-table::

   * - :func:`spikezip_matmul_delta <spikingjelly.activation_based.functional.ann2snn.spikezip_matmul_delta>`
     - SpikeZIP attention single-step matmul delta.
   * - :func:`spikezip_matmul_sequence_delta <spikingjelly.activation_based.functional.ann2snn.spikezip_matmul_sequence_delta>`
     - SpikeZIP attention multi-step matmul delta.
   * - :func:`spikezip_embedding_single_step <spikingjelly.activation_based.functional.ann2snn.spikezip_embedding_single_step>`
     - SpikeZIP embedding single-step time-state transition.
   * - :func:`spikezip_embedding_multi_step <spikingjelly.activation_based.functional.ann2snn.spikezip_embedding_multi_step>`
     - SpikeZIP embedding multi-step time-state transition.
   * - :func:`spikezip_release_bias_single_step <spikingjelly.activation_based.functional.ann2snn.spikezip_release_bias_single_step>`
     - SpikeZIP bias single-step release.
   * - :func:`spikezip_release_bias_multi_step <spikingjelly.activation_based.functional.ann2snn.spikezip_release_bias_multi_step>`
     - SpikeZIP bias multi-step release.
   * - :func:`sta_spike_encoder_single_step <spikingjelly.activation_based.functional.ann2snn.sta_spike_encoder_single_step>`
     - Single-step STA spike encoder residual-state transition.
   * - :func:`sta_constant_single_step <spikingjelly.activation_based.functional.ann2snn.sta_constant_single_step>`
     - Single-step STA constant time-state transition.
   * - :func:`sta_constant_multi_step <spikingjelly.activation_based.functional.ann2snn.sta_constant_multi_step>`
     - Multi-step STA constant time-state transition.

.. toctree::
   :hidden:

   ann2snn <spikingjelly.activation_based.functional.ann2snn>

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

   * - :func:`stdp_linear_single_step <spikingjelly.activation_based.functional.learning.stdp_linear_single_step>`
     - Tensor-only linear STDP single-step update.
   * - :func:`stdp_conv1d_single_step <spikingjelly.activation_based.functional.learning.stdp_conv1d_single_step>`
     - Tensor-only Conv1d STDP single-step update.
   * - :func:`stdp_conv2d_single_step <spikingjelly.activation_based.functional.learning.stdp_conv2d_single_step>`
     - Tensor-only Conv2d STDP single-step update.
   * - :func:`stdp_multi_step <spikingjelly.activation_based.functional.learning.stdp_multi_step>`
     - Multi-step STDP loop over a selected single-step tensor rule.
   * - :func:`mstdp_linear_single_step <spikingjelly.activation_based.functional.learning.mstdp_linear_single_step>`
     - Tensor-only linear mSTDP eligibility update.
   * - :func:`mstdpet_linear_single_step <spikingjelly.activation_based.functional.learning.mstdpet_linear_single_step>`
     - Tensor-only linear mSTDP-ET eligibility update.
   * - :func:`mstdp_reward_delta <spikingjelly.activation_based.functional.learning.mstdp_reward_delta>`
     - Reward modulation for mSTDP eligibility.
   * - :func:`mstdpet_reward_delta <spikingjelly.activation_based.functional.learning.mstdpet_reward_delta>`
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
