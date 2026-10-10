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

三值神经元用于普通稠密 PyTorch 训练，输出包含负事件。二值脉冲压缩器和默认
SynOps/能耗模型不适用于这些输出。使用 ``memopt.optimize_memory`` 时设置
``compress=False``，使用 ``memopt.checkpoint_module`` 时保留 ``compressor=None``。
检测任一极性的首次事件时，先取绝对值，再遵循 ``functional.first_spike_index``
要求的时间维布局。

----

Ternary neurons support ordinary dense PyTorch training and emit signed events.
Binary spike compressors and default SynOps/energy models do not apply to these
outputs. Set ``compress=False`` for ``memopt.optimize_memory`` and keep
``compressor=None`` for ``memopt.checkpoint_module``. To detect the first event of
either polarity, take absolute values and follow the time-axis layout required
by ``functional.first_spike_index``.

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
