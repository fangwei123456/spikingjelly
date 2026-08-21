spikingjelly.activation_based.nir_exchange package
=====================================================

.. admonition:: Quote
   :class: tip

   `Neuromorphic intermediate representation (NIR) <https://neuroir.org/docs/index.html>`_ 是一组计算原语，在不同的神经形态框架和技术栈之间通用。目前，NIR 被多个模拟器和硬件平台支持，使用户能够在这些平台之间无缝迁移。

   `Neuromorphic intermediate representation (NIR) <https://neuroir.org/docs/index.html>`_ is a set of computational primitives, shared across different neuromorphic frameworks and technology stacks. NIR is currently **supported by multiple simulators and hardware platforms**, allowing users to seamlessly move between any of these platforms.

.. note::

   本页面的所有函数都可通过 ``spikingjelly.activation_based.nir_exchange`` 命名空间直接访问。

   The functions are available in the ``spikingjelly.activation_based.nir_exchange`` namespace.

Supported Modules
--------------------------

**Supported SpikingJelly / PyTorch Modules:**

* ``torch.nn.Linear``, :class:`layer.Linear <spikingjelly.activation_based.layer.Linear>`
* ``torch.nn.Conv1d``, :class:`layer.Conv1d <spikingjelly.activation_based.layer.Conv1d>`
* ``torch.nn.Conv2d``, :class:`layer.Conv2d <spikingjelly.activation_based.layer.Conv2d>`
* ``torch.nn.AvgPool2d``, :class:`layer.AvgPool2d <spikingjelly.activation_based.layer.AvgPool2d>`
* ``torch.nn.Flatten``, :class:`layer.Flatten <spikingjelly.activation_based.layer.Flatten>`
* :class:`IFNode <spikingjelly.activation_based.neuron.IFNode>`
* :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` and :class:`ParametricLIFNode <spikingjelly.activation_based.neuron.ParametricLIFNode>`
* :class:`CUBALIFNode <spikingjelly.activation_based.neuron.CUBALIFNode>`

**Supported NIR Nodes:**

* ``nir.Linear``, ``nir.Affine``
* ``nir.Conv1d``
* ``nir.Conv2d``
* ``nir.AvgPool2d``
* ``nir.Flatten``
* ``nir.IF``
* ``nir.LIF``
* ``nir.CubaLIF``

.. warning::

   转换仅支持 hard reset 和非分组卷积。NIR 神经元参数必须在所有神经元上均匀，
   并符合 SpikingJelly 使用的离散化关系。

   Conversion supports only hard reset and ungrouped convolutions. NIR neuron
   parameters must be uniform and match the discretization used by SpikingJelly.

SpikingJelly to NIR
-----------------------------

.. automodule:: spikingjelly.activation_based.nir_exchange.to_nir
   :members:
   :undoc-members:
   :show-inheritance:

NIR to SpikingJelly
-----------------------------

.. automodule:: spikingjelly.activation_based.nir_exchange.from_nir
   :members:
   :undoc-members:
   :show-inheritance:
