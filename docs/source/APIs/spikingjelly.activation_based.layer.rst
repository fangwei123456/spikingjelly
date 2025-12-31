spikingjelly.activation_based.layer package
==============================================

.. note::

   **API稳定性说明**

   SpikingJelly `0.0.0.1.0` 对 ``layer`` 模块的内部实现进行了重构：原来的 ``layer.py`` 文件已被拆分并重组为 ``layer/`` 包，以提升代码的模块化程度和可维护性。

   **该改动不会影响对外公开的 API。** 我们强烈建议用户仍然通过 ``layer`` 这一顶层命名空间来访问相关功能，而不是从具体的内部子模块中进行导入。``layer`` 层级下的导入路径被视为稳定的公共接口；更深层的子模块仅作为内部实现细节，未来可能发生变化。

   .. code:: python

      from spikingjelly.activation_based.layer import SeqToANNContainer # 推荐 ✅
      from spikingjelly.activation_based.layer.container import SeqToANNContainer # 不推荐 ❌

   **API Stability Notice**

   We have refactored the internal implementation of the ``layer`` module. The original ``layer.py`` file has been reorganized into a package ( ``layer/`` ) for better modularity and maintainability.

   **This change does not affect the public API.** Users are strongly encouraged to continue accessing layers directly from the ``layer`` namespace, rather than importing from specific internal submodules. Import paths under ``layer`` are considered part of the stable public API, while deeper submodule paths are treated as implementation details and may change in future releases.

   .. code:: python

      from spikingjelly.activation_based.layer import SeqToANNContainer # recommended ✅
      from spikingjelly.activation_based.layer.container import SeqToANNContainer # not recommended ❌

Containers
-----------------

.. automodule:: spikingjelly.activation_based.layer.container
   :members:
   :undoc-members:
   :show-inheritance:

Wrappers for Stateless Layers
---------------------------------

.. automodule:: spikingjelly.activation_based.layer.stateless_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Attention Layers
-----------------------

.. automodule:: spikingjelly.activation_based.layer.attention
   :members:
   :undoc-members:
   :show-inheritance:

Batch Normalization Variants
-------------------------------------

.. automodule:: spikingjelly.activation_based.layer.bn
   :members:
   :undoc-members:
   :show-inheritance:

Dropout Variants
-------------------------------------

.. automodule:: spikingjelly.activation_based.layer.dropout
   :members:
   :undoc-members:
   :show-inheritance:

OTTT Modules
-------------------------

.. automodule:: spikingjelly.activation_based.layer.ottt
   :members:
   :undoc-members:
   :show-inheritance:

Miscellaneous
-------------------------

.. automodule:: spikingjelly.activation_based.layer.misc
   :members:
   :undoc-members:
   :show-inheritance:
