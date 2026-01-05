spikingjelly.datasets package
==================================

.. note::

   **API 改动说明**

   SpikingJelly ``0.0.0.1.0`` 对 ``datasets`` 包进行了结构性重构，以提升代码的模块化程度和可维护性。主要改动为：

   * ``datasets/__init__.py`` 被拆分为 ``base.py``、 ``utils.py`` 以及 ``transform.py`` ；
   * 新的 ``datasets/__init__.py`` 仅作为 API 门面（facade），负责引入并暴露各子模块中的公开对象；
   * ``datasets/to_rep_x.py`` 及其中内容已被移除。

   使用建议：

   * 数据集类：推荐从 ``spikingjelly.datasets`` 命名空间直接导入；
   * 工具函数和数据增强：推荐通过 ``spikingjelly.datasets.utils`` 和 ``spikingjelly.datasets.transform`` 子模块访问；

   .. code:: python

      # 推荐 ✅
      from spikingjelly.datasets import DVS128Gesture
      from spikingjelly.datasets.utils import create_sub_dataset

      # 旧的导入方式，不推荐 ❌
      from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
      from spikingjelly.datasets import create_sub_dataset

   为保证兼容性，以上旧的导入方式在当前版本中仍然可用，但不再推荐在新代码中使用。

   **API Change Notice**

   In SpikingJelly ``0.0.0.1.0``, the ``datasets`` package has undergone a structural refactor to improve modularity and maintainability. The main changes include:

   * ``datasets/__init__.py`` has been split into ``base.py``, ``utils.py``, and ``transform.py``;
   * the new ``datasets/__init__.py`` now serves solely as an API facade, responsible for importing and re-exporting public objects from its submodules;
   * ``datasets/to_rep_x.py`` and its contents have been removed.

   Usage recommendations:

   * Dataset classes: import them directly from the ``spikingjelly.datasets`` namespace;
   * Utility functions and data augmentation: access them via the ``spikingjelly.datasets.utils`` and ``spikingjelly.datasets.transform`` submodules;

   .. code:: python

      # Recommended ✅
      from spikingjelly.datasets import DVS128Gesture
      from spikingjelly.datasets.utils import create_sub_dataset

      # Legacy import patterns, not recommended ❌
      from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
      from spikingjelly.datasets import create_sub_dataset

   To preserve compatibility, the legacy import patterns shown above remain supported in the current version, but their use is discouraged.

Datasets
-------------

.. toctree::
   :maxdepth: 2

   spikingjelly.datasets.asl_dvs
   spikingjelly.datasets.bullying10k
   spikingjelly.datasets.cifar10_dvs
   spikingjelly.datasets.dvs128_gesture
   spikingjelly.datasets.dvs_lip
   spikingjelly.datasets.es_imagenet
   spikingjelly.datasets.hardvs
   spikingjelly.datasets.n_caltech101
   spikingjelly.datasets.n_mnist
   spikingjelly.datasets.nav_gesture
   spikingjelly.datasets.shd
   spikingjelly.datasets.speechcommands

Dataset Base Class
------------------------

.. toctree::
   :maxdepth: 2

   spikingjelly.datasets.base

Dataset Utilities
-----------------------

.. toctree::
   :maxdepth: 2

   spikingjelly.datasets.utils

Data Augmentations
-----------------------

.. toctree::
   :maxdepth: 2

   spikingjelly.datasets.transform
