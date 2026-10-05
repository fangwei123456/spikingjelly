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
   :exclude-members: store_v_seq, extra_repr, apply_hard_reset, apply_soft_reset, jit_neuronal_adaptation

Integrate-and-fire (IF) Neurons
------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.integrate_and_fire
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends

Leaky Integrate-and-fire (LIF) Neurons
------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.lif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends

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

**中文：实现组织**

``FlexSN`` 负责用户 ``core``、状态、静态输入和 ``reset()``。
``neuron.flexsn_trace`` 捕获推理/训练 FX 图，``neuron.flexsn_hop`` 负责 Dynamo
捕获。后端 ``spikingjelly._ops.flexsn`` 接收图，完成 Triton 代码生成、算子注册、
启动与反向；HOP 的执行也位于该包。后端不依赖神经元实例。
现有 ``backend="torch" / "hop" / "triton"`` 选择保持不变，不引入自动后端选择。

**English: implementation layout**

``FlexSN`` owns the user ``core``, state, static inputs and ``reset()``.
``neuron.flexsn_trace`` captures inference/training FX graphs, while
``neuron.flexsn_hop`` handles Dynamo capture. The backend in
``spikingjelly._ops.flexsn`` consumes graphs and owns Triton code generation,
operator registration, launches and backward execution, as well as HOP execution.
It does not depend on neuron instances. The existing
``backend="torch" / "hop" / "triton"`` selection remains explicit.


.. automodule:: spikingjelly.activation_based.neuron.flexsn
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr, store_state_seqs
