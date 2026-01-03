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
* ``torch.nn.Conv2d``, :class:`layer.Conv2d <spikingjelly.activation_based.layer.Conv2d>`
* ``torch.nn.AvgPool2d``, :class:`layer.AvgPool2d <spikingjelly.activation_based.layer.AvgPool2d>`
* ``torch.nn.Flatten``, :class:`layer.Flatten <spikingjelly.activation_based.layer.Flatten>`
* :class:`IFNode <spikingjelly.activation_based.neuron.IFNode>`
* :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` and :class:`ParametricLIFNode <spikingjelly.activation_based.neuron.ParametricLIFNode>`

**Supported NIR Nodes:**

* ``nir.Linear``, ``nir.Affine``
* ``nir.Conv2d``
* ``nir.AvgPool2d``
* ``nir.Flatten``
* ``nir.IF``
* ``nir.LIF``

.. note::

   我们将在后续更新中逐渐完善对其他模块的支持。

   We will add support for more modules in future updates.

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