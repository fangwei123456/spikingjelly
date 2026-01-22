spikingjelly.activation_based.neuron module
===============================================

.. note::

   **API稳定性说明**

   SpikingJelly `0.0.0.1.0` 对 ``neuron`` 模块的内部实现进行了重构：原来的 ``neuron.py`` 文件已被拆分并重组为 ``neuron/`` 包，以提升代码的模块化程度和可维护性。

   **该改动不会影响对外公开的 API。** 我们强烈建议用户仍然通过 ``neuron`` 这一顶层命名空间来访问相关功能，而不是从具体的内部子模块中进行导入。 ``neuron`` 层级下的导入路径被视为稳定的公共接口；更深层的子模块仅作为内部实现细节，未来可能发生变化。

   .. code:: python

      from spikingjelly.activation_based.neuron import LIFNode # 推荐 ✅
      from spikingjelly.activation_based.neuron.lif import LIFNode # 不推荐 ❌

   **API Stability Notice**

   We have refactored the internal implementation of the ``neuron`` module. The original ``neuron.py`` file has been reorganized into a package ( ``neuron/`` ) for better modularity and maintainability.

   **This change does not affect the public API.** Users are strongly encouraged to continue accessing layers directly from the ``neuron`` namespace, rather than importing from specific internal submodules. Import paths under ``neuron`` are considered part of the stable public API, while deeper submodule paths are treated as implementation details and may change in future releases.

   .. code:: python

      from spikingjelly.activation_based.neuron import LIFNode # recommended ✅
      from spikingjelly.activation_based.neuron.lif import LIFNode # not recommended ❌

Core Neuron Modules
++++++++++++++++++++++++++++++++++

SpikingJelly 的 **核心神经元模块** 提供了规范神经元抽象。
这些神经元是 SNN 领域中被广泛接受和使用的模型，旨在作为研究与实际应用中的基础构建单元。

----

SpikingJelly's **core neuron modules** provide canonical and stable neuron abstractions.
These neurons represent widely accepted models in SNN (SNN) literature and are designed to serve as
fundamental building blocks for both research and practical applications.

.. toctree::
   :hidden:

   spikingjelly.activation_based.neuron.core

Base Classes
---------------------------

.. list-table::

   * - :class:`SimpleBaseNode <spikingjelly.activation_based.neuron.base_node.SimpleBaseNode>`
     - Simplified base class for neurons.
   * - :class:`BaseNode <spikingjelly.activation_based.neuron.base_node.BaseNode>`
     - Base class for neurons.
   * - :class:`NonSpikingBaseNode <spikingjelly.activation_based.neuron.base_node.NonSpikingBaseNode>`
     - Base class for neurons that do not emit spikes.

Integrate-and-fire (IF) Neurons
------------------------------------

.. list-table::

   * - :class:`SimpleIFNode <spikingjelly.activation_based.neuron.integrate_and_fire.SimpleIFNode>`
     - Simplified IF neuron.
   * - :class:`IFNode <spikingjelly.activation_based.neuron.integrate_and_fire.IFNode>`
     - IF neuron.
   * - :class:`NonSpikingIFNode <spikingjelly.activation_based.neuron.integrate_and_fire.NonSpikingIFNode>`
     - IF variant that does not emit spikes.

Leaky Integrate-and-fire (LIF) Neurons
------------------------------------------------

.. list-table::

   * - :class:`SimpleLIFNode <spikingjelly.activation_based.neuron.lif.SimpleLIFNode>`
     - Simplified LIF neuon.
   * - :class:`LIFNode <spikingjelly.activation_based.neuron.lif.LIFNode>`
     - LIF neuron.
   * - :class:`NonSpikingLIFNode <spikingjelly.activation_based.neuron.lif.NonSpikingLIFNode>`
     - LIF variant that does not emit spikes.

Parametric Leaky Integrate-and-fire (PLIF) Neurons
----------------------------------------------------------

.. list-table::

   * - :class:`ParametricLIFNode <spikingjelly.activation_based.neuron.plif.ParametricLIFNode>`
     - Parametric LIF neuron.

Parallel Spiking Neuron Family
--------------------------------------------

.. list-table::

   * - :class:`PSN <spikingjelly.activation_based.neuron.psn.PSN>`
     - Parallel Spiking Neuron (PSN).
   * - :class:`MaskedPSN <spikingjelly.activation_based.neuron.psn.MaskedPSN>`
     - Masked PSN.
   * - :class:`SlidingPSN <spikingjelly.activation_based.neuron.psn.SlidingPSN>`
     - Sliding PSN.

FlexSN
-------------

.. list-table::

   * - :class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>`
     - High-level FlexSN neuron module for automatic Triton kernel generation.
   * - :class:`FlexSNKernel <spikingjelly.activation_based.neuron.flexsn.FlexSNKernel>`
     - Low-level callable interface for the generated Triton neuron kernel.

Research-specific Neuron Modules
++++++++++++++++++++++++++++++++++++++++

**特定研究的神经元** 模块包含为特定研究目的或实验场景提出的神经元模型。
这些神经元通常源自具体论文，其适用范围和使用频率相比 :doc:`核心神经元 <./spikingjelly.activation_based.neuron.core>` 而言更加有限。

----

**Research-specific neuron modules** contain neuron models that are proposed for particular research purposes or experimental settings.
These neurons often originate from specific papers and are not so widely used as :doc:`./spikingjelly.activation_based.neuron.core`.

.. toctree::
   :hidden:

   spikingjelly.activation_based.neuron.research

Adaptive Neurons
--------------------------------------------

.. list-table::

   * - :class:`AdaptBaseNode <spikingjelly.activation_based.neuron.adapt.AdaptBaseNode>`
     - Base class for neurons with adaptation.
   * - :class:`IzhikevichNode <spikingjelly.activation_based.neuron.adapt.IzhikevichNode>`
     - Izhikevich neuron model.

Nonlinear Integrate-and-fire Neurons
--------------------------------------------------

.. list-table::

   * - :class:`QIFNode <spikingjelly.activation_based.neuron.nonlinear_if.QIFNode>`
     - Quadratic Integrate-and-Fire (QIF) neuron.
   * - :class:`EIFNode <spikingjelly.activation_based.neuron.nonlinear_if.EIFNode>`
     - Exponential Integrate-and-Fire (EIF) neuron.

LIF Variants
--------------------------------------------------

.. list-table::

   * - :class:`GatedLIFNode <spikingjelly.activation_based.neuron.lif_variants.GatedLIFNode>`
     - Gated LIF (GLIF) neuron.
   * - :class:`KLIFNode <spikingjelly.activation_based.neuron.lif_variants.KLIFNode>`
     - KLIF neuron.
   * - :class:`CUBALIFNode <spikingjelly.activation_based.neuron.lif_variants.CUBALIFNode>`
     - CUrrent-BAsed LIF neuron.
   * - :class:`LIAFNode <spikingjelly.activation_based.neuron.lif_variants.LIAFNode>`
     - Leaky integrate and analog fire neuron.

Neurons with Membrane Potential Batch Normalization
----------------------------------------------------------

.. list-table::

   * - :class:`MPBNBaseNode <spikingjelly.activation_based.neuron.mpbn.MPBNBaseNode>`
     - Base class for neurons with membrane potential batch normalization (MPBN).
   * - :class:`MPBNLIFNode <spikingjelly.activation_based.neuron.mpbn.MPBNLIFNode>`
     - LIF neuron with MPBN

Differentiation on Spike Representation (DSR)
--------------------------------------------------

.. list-table::

   * - :class:`DSRIFNode <spikingjelly.activation_based.neuron.dsr.DSRIFNode>`
     - IF neuron with DSR.
   * - :class:`DSRLIFNode <spikingjelly.activation_based.neuron.dsr.DSRLIFNode>`
     - LIF neuron with DSR.

Neurons for Online Learning
-----------------------------------------------------------------

.. list-table::

   * - :class:`OTTTLIFNode <spikingjelly.activation_based.neuron.online_learning.OTTTLIFNode>`
     - LIF neuron for online training through time (OTTT).
   * - :class:`SLTTLIFNode <spikingjelly.activation_based.neuron.online_learning.SLTTLIFNode>`
     - LIF neuron for spatial learning through time (SLTT).

Neurons with Inter-layer Connection
--------------------------------------------------

.. list-table::

   * - :class:`ILCBaseNode <spikingjelly.activation_based.neuron.inter_layer_connection.ILCBaseNode>`
     - Base class for neurons with inter-layer connection.
   * - :class:`ILCIFNode <spikingjelly.activation_based.neuron.inter_layer_connection.ILCIFNode>`
     - IF neuron layer with inter-layer connection.
   * - :class:`ILCLIFNode <spikingjelly.activation_based.neuron.inter_layer_connection.ILCLIFNode>`
     - LIF neuron layer with inter-layer connection.
   * - :class:`ILCCUBALIFNode <spikingjelly.activation_based.neuron.inter_layer_connection.ILCCUBALIFNode>`
     - CUrrent-BAsed LIF neuron layer with inter-layer connection.

Neurons with Input or Output Noise
---------------------------------------------

.. list-table::

   * - :func:`powerlaw_psd_gaussian <spikingjelly.activation_based.neuron.noisy.powerlaw_psd_gaussian>`
     - Generate Gaussian noise with a power spectrum.
   * - :class:`NoisyBaseNode <spikingjelly.activation_based.neuron.noisy.NoisyBaseNode>`
     - Base class for neurons with noisy input / output.
   * - :class:`NoisyCUBALIFNode <spikingjelly.activation_based.neuron.noisy.NoisyCUBALIFNode>`
     - CUrrent-BAsed LIF neuron with noisy input / output.
   * - :class:`NoisyILCBaseNode <spikingjelly.activation_based.neuron.noisy.NoisyILCBaseNode>`
     - Base class for neurons with noisy input / output and inter-layer connection.
   * - :class:`NoisyILCCUBALIFNode <spikingjelly.activation_based.neuron.noisy.NoisyILCCUBALIFNode>`
     - CUrrent-BAsed LIF neuron layer with noisy input / output and inter-layer connection.
   * - :class:`NoisyNonSpikingBaseNode <spikingjelly.activation_based.neuron.noisy.NoisyNonSpikingBaseNode>`
     - Base class for neurons with noisy input that does not emit spikes.
   * - :class:`NoisyNonSpikingIFNode <spikingjelly.activation_based.neuron.noisy.NoisyNonSpikingIFNode>`
     - IF neuron with noisy input that does not emit spikes.
