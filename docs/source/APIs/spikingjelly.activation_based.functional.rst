spikingjelly.activation_based.functional package
==================================================

.. note::

   **API稳定性说明**

   SpikingJelly `0.0.0.1.0` 对 ``functional`` 模块的内部实现进行了重构：原来的 ``functional.py`` 文件已被拆分并重组为 ``functional/`` 包，以提升代码的模块化程度和可维护性。

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
------------------------------------------------

本模块包含若干辅助函数，用于统一设置网络中每个子模块的相关配置。

This module contains helper functions to set configurations for each submodule in a network.

.. automodule:: spikingjelly.activation_based.functional.net_config
   :members:
   :undoc-members:
   :show-inheritance:

Forward Functions
----------------------------

.. automodule:: spikingjelly.activation_based.functional.forward
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
----------------------------

.. automodule:: spikingjelly.activation_based.functional.loss
   :members:
   :undoc-members:
   :show-inheritance:

Functions for Conv-BN Fusion
---------------------------------

.. automodule:: spikingjelly.activation_based.functional.conv_bn_fuse
   :members:
   :undoc-members:
   :show-inheritance:

Online Learning Pipelines
----------------------------

.. automodule:: spikingjelly.activation_based.functional.online_learning
   :members:
   :undoc-members:
   :show-inheritance:

Miscellaneous
----------------------------

.. automodule:: spikingjelly.activation_based.functional.misc
   :members:
   :undoc-members:
   :show-inheritance: