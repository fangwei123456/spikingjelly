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
   :exclude-members: supported_backends, extra_repr, w_float_to_tensor

Nonlinear Integrate-and-fire Neurons
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.nonlinear_if
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr

LIF Variants
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.lif_variants
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr

Neurons with Membrane Potential Batch Normalization
----------------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.mpbn
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends

Differentiation on Spike Representation (DSR)
--------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.dsr
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr

Neurons for Online Learning
-----------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.neuron.online_learning
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends

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
   :exclude-members: