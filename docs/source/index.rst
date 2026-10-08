**Language:**
:ref:`中文 <index>` | :ref:`English <index_en>`

.. _index:

欢迎来到惊蜇(SpikingJelly)的文档
###################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ ，使用脉冲神经网络(Spiking Neural Network, SNN)进行深度学习的框架。

版本说明
----------------
自 ``0.0.0.0.14`` 版本开始，包括 ``clock_driven`` 和 ``event_driven`` 在内的模块被重命名了，请参考教程 :doc:`./tutorials/cn/migrate_from_legacy`。

V2 版本更新记录见 :doc:`./changelog`。

不同版本文档的地址（其中 `latest` 是开发版）：

- `zero <https://spikingjelly.readthedocs.io/zh_CN/zero/>`__

- `0.0.0.0.4 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/>`__

- `0.0.0.0.6 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/>`__

- `0.0.0.0.8 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.8/>`__

- `0.0.0.0.10 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.10/>`__

- `0.0.0.0.12 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/>`__

- `0.0.0.0.14 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/>`__

- `latest <https://spikingjelly.readthedocs.io/zh_CN/latest/>`__

安装
----------------

SpikingJelly是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装SpikingJelly。最新版要求 Python >= 3.11 和 ``torch>=2.6.0``；已验证环境包括 ``torch==2.7.1``，最低要求不等于所有版本及设备均已验收。

从 SpikingJelly V2 起，发布版本采用兼容 PEP 440 的语义化版本号。V2 之前使用历史遗留的 ``0.0.0.0.X`` 版本方案，其中奇数 ``X`` 对应 GitHub/OpenI 上的开发版，偶数 ``X`` 对应 PyPI 稳定版。

**从 PyPI 安装最新的稳定版本：**

.. code-block:: bash

    uv pip install spikingjelly

**从源代码安装最新的开发版：**

通过 `GitHub <https://github.com/fangwei123456/spikingjelly>`_：

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    uv pip install .

通过 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ ：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    uv pip install .

**可选依赖**

常规 PyPI wheel 是纯 Python 包，CPU 使用 Torch，不需要 GPU 可选依赖。
NVIDIA CUDA 输入会自动采用兼容的实现；没有原生扩展时，可安装 Triton：

.. code-block:: bash

    uv pip install "spikingjelly[triton]"
    # 源码目录中的 editable 安装：
    uv pip install --editable ".[triton]"

不能假定任意平台的 CUDA Torch 安装都包含可用 Triton。其他 GPU 平台及编译后端
不属于本教程的 CUDA 加速保证范围。

原生 CUDA 扩展是可选的本地构建。先安装匹配的 CUDA 版 Torch、CUDA Toolkit
（含 nvcc）、C++ 编译器，以及 ``setuptools>=77.0.3`` 和 ninja，再从源码目录执行：

.. code-block:: bash

    SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .

缺少 CUDA 版 Torch 或工具链时，安装会提示并跳过原生扩展；工具链存在但实际编译失败时，
安装会报错。无可见 GPU 的构建应显式设置目标设备的 ``TORCH_CUDA_ARCH_LIST``。
运行时只加载原生二进制，不调用编译器；更换 Torch/CUDA 或目标 GPU 后，可能需要
重新构建。Triton 自身保留首次 JIT 和缓存。近期不发布预编译原生 CUDA wheel。

安装后从 :doc:`/tutorials/cn/neuron` 开始；执行、编译和诊断见
:doc:`/tutorials/cn/triton_backend`。

若想使用 ``nir_exchange`` 功能，请安装 `NIR <https://github.com/neuromorphs/NIR>`_ 和 `NIRTorch <https://github.com/neuromorphs/NIRTorch>`_ 。

.. code:: bash

    uv pip install "spikingjelly[nir]"

上手教程
----------------------

.. toctree::
    :maxdepth: 2

    /tutorials/cn/index

引用和出版物
-------------------------
如果您在自己的工作中用到了惊蜇(SpikingJelly)，您可以按照下列格式进行引用：

.. code-block::

    @article{
    doi:10.1126/sciadv.adi1480,
    author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
    title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
    journal = {Science Advances},
    volume = {9},
    number = {40},
    pages = {eadi1480},
    year = {2023},
    doi = {10.1126/sciadv.adi1480},
    URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
    eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
    abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}

使用惊蜇(SpikingJelly)的出版物可见于 :doc:`./publications` 。

许可证
-------------------------
SpikingJelly 的项目许可证为
`Apache-2.0 <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSE>`_。
第三方归属与许可条款汇总于
`LICENSES/NOTICE <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSES/NOTICE>`_。
适用范围与历史许可证见
`许可证指南 <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSES/README.md>`_。

项目信息
-------------------------
北京大学信息科学技术学院数字媒体所媒体学习组 `Multimedia Learning Group <https://pkuml.org/>`_ 和 `鹏城实验室 <https://www.pcl.ac.cn/>`_ 是SpikingJelly的主要负责机构。

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

SpikingJelly主要由以下开发者开发维护：

**2024.07~现在:** `黄一凡 <https://github.com/AllenYolk>`_, `薛鹏 <https://github.com/PengXue0812>`_

**2019.12~2024.06:** `方维 <https://github.com/fangwei123456>`_, `陈彦骐 <https://github.com/Yanqi-Chen>`_, `丁健豪 <https://github.com/DingJianhao>`_, `陈鼎 <https://github.com/lucifer2859>`_, `黄力炜 <https://github.com/Grasshlw>`_

全体贡献者名单可见于 `贡献者 <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_ 。

友情链接
-------------------------
* `脉冲神经网络相关博客 <https://www.cnblogs.com/lucifer1997/tag/SNN/>`_
* `脉冲强化学习相关博客 <https://www.cnblogs.com/lucifer1997/tag/SNN-RL/>`_
* `神经形态计算软件框架LAVA相关博客 <https://www.cnblogs.com/lucifer1997/p/16286303.html>`_

.. _index_en:

Welcome to SpikingJelly's documentation
############################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ is a deep learning framework for Spiking Neural Network (SNN) based on `PyTorch <https://pytorch.org/>`_.

Notification
----------------
From the version ``0.0.0.0.14``, modules including ``clock_driven`` and ``event_driven`` are renamed. \
Please refer to the tutorial :doc:`./tutorials/en/migrate_from_legacy`.

See :doc:`./changelog` for the V2 release changelog.

Docs for different versions (`latest` is the developing version):

- `zero <https://spikingjelly.readthedocs.io/zh_CN/zero/>`__

- `0.0.0.0.4 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/#index-en>`__

- `0.0.0.0.6 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/#index-en>`__

- `0.0.0.0.8 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.8/#index-en>`__

- `0.0.0.0.10 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.10/#index-en>`__

- `0.0.0.0.12 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/#index-en>`__

- `0.0.0.0.14 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/#index-en>`__

- `latest <https://spikingjelly.readthedocs.io/zh_CN/latest/#index-en>`__

Installation
----------------

Note that SpikingJelly is based on PyTorch. Please make sure that you have installed PyTorch before you install SpikingJelly. The latest version of SpikingJelly requires ``torch>=2.6.0`` and is tested on ``torch==2.7.1`` .

Starting from SpikingJelly V2, release versions use PEP 440 compatible SemVer-style version numbers. Before V2, SpikingJelly used the legacy ``0.0.0.0.X`` scheme: odd ``X`` tracked the development version on GitHub/OpenI, and even ``X`` tracked the stable PyPI release.

**Install the last stable version from PyPI:**

.. code-block:: bash

    uv pip install spikingjelly

**Install the latest developing version from the source codes:**

From `GitHub <https://github.com/fangwei123456/spikingjelly>`_:

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    uv pip install .

From `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    uv pip install .

**Optional Dependencies**

Python >= 3.11 and Torch >= 2.6 are required. Torch 2.7.1 is a verified
configuration; the minimum version is not a claim that every version/device was tested.
Regular PyPI wheels are pure Python. CPU execution needs no optional GPU package.
For NVIDIA CUDA execution without a native extension, install Triton:

.. code-block:: bash

    uv pip install "spikingjelly[triton]"
    # Editable installation from a source checkout:
    uv pip install --editable ".[triton]"

Do not assume every platform's CUDA Torch distribution includes usable Triton.
Other GPU platforms and compiler backends are outside this tutorial's CUDA guarantee.

Native CUDA extensions are optional local builds. Prepare matching CUDA-enabled
Torch, a CUDA Toolkit with nvcc, a C++ compiler, ``setuptools>=77.0.3`` and ninja,
then run from the source checkout:

.. code-block:: bash

    SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .

Missing CUDA-enabled Torch/toolchains produce a message and skip native extensions.
Actual compilation failures with a present toolchain fail installation. Builds
without a visible GPU must set ``TORCH_CUDA_ARCH_LIST`` for the target device.
Runtime loads native binaries without invoking a compiler. Changes to Torch/CUDA
or target GPUs may require rebuilding. Triton retains first-use JIT and caching.
Precompiled native CUDA wheels are not provided in the near-term release plan.

Start with :doc:`/tutorials/en/neuron`; see :doc:`/tutorials/en/triton_backend`
for execution, compilation and diagnostics.

To enable ``nir_exchange`` , install `NIR <https://github.com/neuromorphs/NIR>`_ and `NIRTorch <https://github.com/neuromorphs/NIRTorch>`_ .

.. code:: bash

    uv pip install "spikingjelly[nir]"

Tutorials
------------------------

.. toctree::
    :maxdepth: 2

    /tutorials/en/index

Citation
-------------------------

If you use SpikingJelly in your work, please cite it as follows:

.. code-block::

    @article{
    doi:10.1126/sciadv.adi1480,
    author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
    title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
    journal = {Science Advances},
    volume = {9},
    number = {40},
    pages = {eadi1480},
    year = {2023},
    doi = {10.1126/sciadv.adi1480},
    URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
    eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
    abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}


Publications using SpikingJelly are recorded in :doc:`./publications`.

License
-------------------------
SpikingJelly's project license is
`Apache-2.0 <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSE>`_.
Third-party attributions and license terms are collected in
`LICENSES/NOTICE <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSES/NOTICE>`_.
Scope and historical licenses are described in the
`license guide <https://github.com/fangwei123456/spikingjelly/blob/master/LICENSES/README.md>`_.

About
-------------------------
`Multimedia Learning Group, Institute of Digital Media (NELVT), Peking University <https://pkuml.org/>`_ and `Peng Cheng Laboratory <http://www.szpclab.com/>`_ are the main institutions behind the development of SpikingJelly.

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

SpikingJelly has been developed and maintained by multiple main developers over time.

**2024.07~Now:** `Yifan Huang <https://github.com/AllenYolk>`_, `Peng Xue <https://github.com/PengXue0812>`_

**2019.12~2024.06**: `Wei Fang <https://github.com/fangwei123456>`_, `Yanqi Chen <https://github.com/Yanqi-Chen>`_, `Jianhao Ding <https://github.com/DingJianhao>`_, `Ding Chen <https://github.com/lucifer2859>`_, `Liwei Huang <https://github.com/Grasshlw>`_

The list of contributors can be found at `contributors <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_.

.. toctree::
   :hidden:

   /APIs/spikingjelly
   changelog
   publications
   贡献指南 | Contributing <https://github.com/fangwei123456/spikingjelly/blob/master/CONTRIBUTING.md>
