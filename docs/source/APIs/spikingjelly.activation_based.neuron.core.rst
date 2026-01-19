Core Neuron Modules
===============================================

SpikingJelly 的 **核心神经元模块** 提供了规范神经元抽象。
这些神经元是 SNN 领域中被广泛接受和使用的模型，旨在作为研究与实际应用中的基础构建单元。

纳入该类别的主要标准包括：

- 神经元在概念上具有通用性，不依赖于某一特定论文、任务或训练策略。
- 该神经元可以被推荐用于下游项目中的通用建模需求。

除非明确需要某种特定的研究型行为，否则在构建新模型时，建议用户优先使用核心神经元模块。

----

SpikingJelly's **core neuron modules** provide canonical and stable neuron abstractions.
These neurons represent widely accepted models in SNN (SNN) literature and are designed to serve as
fundamental building blocks for both research and practical applications.

The main criteria for inclusion in this category are:

- The neuron is conceptually general and not tied to a specific paper, task, or training strategy.
- The neuron can be recommended for general use in downstream projects.

Users are encouraged to preferentially use core neuron modules when building new models, unless a specific research-oriented behavior is explicitly required.

Base Classes
---------------------------

.. automodule:: spikingjelly.activation_based.neuron.base_node
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: store_v_seq, extra_repr, v_float_to_tensor, jit_hard_reset, jit_soft_reset, jit_neuronal_adaptation

Integrate-and-fire (IF) Neurons
------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.integrate_and_fire
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, jit_eval_single_step_forward_hard_reset, jit_eval_single_step_forward_soft_reset, jit_eval_multi_step_forward_hard_reset, jit_eval_multi_step_forward_hard_reset_with_v_seq, jit_eval_multi_step_forward_soft_reset, jit_eval_multi_step_forward_soft_reset_with_v_seq

Leaky Integrate-and-fire (LIF) Neurons
------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.lif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, neuronal_charge_decay_input_reset0, neuronal_charge_decay_input, neuronal_charge_no_decay_input_reset0, neuronal_charge_no_decay_input, jit_eval_single_step_forward_hard_reset_decay_input, jit_eval_single_step_forward_hard_reset_no_decay_input, jit_eval_single_step_forward_soft_reset_decay_input, jit_eval_single_step_forward_soft_reset_no_decay_input, jit_eval_multi_step_forward_hard_reset_decay_input, jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq, jit_eval_multi_step_forward_hard_reset_no_decay_input, jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq, jit_eval_multi_step_forward_soft_reset_decay_input, jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq, jit_eval_multi_step_forward_soft_reset_no_decay_input, jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq,

Parametric Leaky Integrate-and-fire (PLIF) Neurons
----------------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.plif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr

Parallel Spiking Neuron Family
--------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.psn
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr

FlexSN
-------------

.. automodule:: spikingjelly.activation_based.neuron.flexsn
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr, store_state_seqs