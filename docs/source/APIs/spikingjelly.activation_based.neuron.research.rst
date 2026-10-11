Research-specific Neuron Modules
===============================================

**特定研究的神经元** 模块包含为特定研究目的或实验场景提出的神经元模型。
这些神经元通常源自具体论文，其适用范围和使用频率相比 :doc:`核心神经元 <./spikingjelly.activation_based.neuron.core>` 而言更加有限。

这类神经元通常具有以下特征：

- 由某一具体论文提出。
- 专用于特定训练方法或任务，依赖于这些场景所引入的假设。

特定研究的神经元旨在支持研究复现与实验探索。虽然它们同样受到完整支持，但并不被视为核心抽象的一部分。用户在使用时应充分理解其设计初衷、适用范围及潜在局限性。

----

**Research-specific neuron modules** contain neuron models that are proposed for particular research purposes or experimental settings.
These neurons often originate from specific papers and are not so widely used as :doc:`./spikingjelly.activation_based.neuron.core`.

Typical characteristics of research-specific neurons include:

- Introduced by a specific paper.
- Dependence on assumptions introduced by a particular learning method or task setting.

These modules are provided to facilitate reproducibility and experimentation.
While fully supported, they are not considered part of the core abstraction.
Users should adopt them with an understanding of their intended scope and limitations.

Adaptive Neurons
--------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.adapt
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Nonlinear Integrate-and-fire Neurons
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.nonlinear_if
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Resonate-and-Fire Neurons
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.resonate
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

LIF Variants
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.lif_variants
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Neurons with Membrane Potential Batch Normalization
----------------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.mpbn
   :members:
   :undoc-members:
   :show-inheritance:

Differentiation on Spike Representation (DSR)
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.dsr
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Neurons for Online Learning
-----------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.online_learning
   :members:
   :undoc-members:
   :show-inheritance:

Few-Spike / LAS Neuron Primitives
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.few_spike
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Signed Ternary Neurons
--------------------------------------------------

:class:`~spikingjelly.activation_based.neuron.TernaryIFNode` 和
:class:`~spikingjelly.activation_based.neuron.TernaryLIFNode` 使用正负两个阈值，
分别产生 ``1`` 和 ``-1`` 事件，未达到阈值时输出 ``0``。IF 积累输入，
LIF 在积累过程中加入电位泄漏；两者均支持替代梯度训练、硬重置和有符号软重置。

----

:class:`~spikingjelly.activation_based.neuron.TernaryIFNode` and
:class:`~spikingjelly.activation_based.neuron.TernaryLIFNode` emit ``1`` or ``-1``
at the positive or negative threshold, and ``0`` otherwise. IF accumulates input;
LIF adds voltage leakage. Both support surrogate-gradient training, hard reset
and signed soft reset.

.. _signed-ternary-compatibility:

.. admonition:: 工具兼容性 / Tool compatibility

   有符号事件适用于稠密 PyTorch 层，但二值脉冲压缩、默认 SynOps/能耗估计
   和 NIR/Lava 导出不适用。使用 ``memopt.optimize_memory`` 时设置
   ``compress=False``，使用 ``memopt.checkpoint_module`` 时设置
   ``compressor=None``。用 ``functional.first_spike_index`` 检测任一极性的
   首次事件时，先取绝对值，并遵循其时间维布局。

   Signed events work with dense PyTorch layers. Binary spike compression,
   default SynOps/energy estimates and NIR/Lava export do not support them.
   Set ``compress=False`` in ``memopt.optimize_memory`` and ``compressor=None``
   in ``memopt.checkpoint_module``. To find the first event of either polarity
   with ``functional.first_spike_index``, take absolute values first and follow
   its time-axis layout.

.. automodule:: spikingjelly.activation_based.neuron.ternary
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

SpikeZIP Neuron Primitives
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.spikezip
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Integer-Valued Training Neurons
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.ilif
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Neurons with Inter-layer Connection
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.inter_layer_connection
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members:

Neurons with Input or Output Noise
---------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.noisy
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr
