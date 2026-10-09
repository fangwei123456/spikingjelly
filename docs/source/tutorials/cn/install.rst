安装指南
==========================

English version: :doc:`../en/install`

准备环境
--------------------------

V2 要求 Python >= 3.11、Torch >= 2.6。先创建环境，再按
`PyTorch 官方安装入口 <https://pytorch.org/get-started/locally/>`_ 为目标设备安装
Torch、torchvision 和 torchaudio：

.. code-block:: bash

    uv venv --python 3.11
    source .venv/bin/activate

本文按当前开发源码说明。PyPI 安装只包含已发布的改动；使用旧版本时，请切换到
对应版本的文档。最低版本要求不代表所有版本和设备都已经验收；已有验证包括
Torch 2.7.1，以及 Torch 2.11.0+cu128 / Triton 3.6.0 的 GPU 环境。

V2 使用兼容 PEP 440 的语义化版本号。此前的 ``0.0.0.0.X`` 是历史版本方案，
奇数 ``X`` 为开发版，偶数为 PyPI 稳定版。

选择安装方式
--------------------------

CPU 不需要 Triton 或原生 CUDA。NVIDIA CUDA 用户可按需要安装 Triton、手动
构建原生扩展，或同时准备两者。Triton 保持可选，不会由普通安装默认添加。

.. figure:: /_static/tutorials/install/installation.svg
    :alt: 安装决策树：准备 Python 和 Torch 后，CPU 普通安装；NVIDIA CUDA 可使用参考实现、可选 Triton 或手动构建原生 CUDA。
    :width: 100%

    两种加速实现可以同时安装，模块无需配置 backend。

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - 安装方式
      - 命令
    * - PyPI 发布版
      - ``uv pip install spikingjelly``
    * - PyPI 先行版
      - ``uv pip install --pre spikingjelly``
    * - 可选 Triton
      - ``uv pip install "spikingjelly[triton]"``
    * - 最新源码开发版
      - ``uv pip install git+https://github.com/fangwei123456/spikingjelly.git``

若开发版需要 Triton，可执行：

.. code-block:: bash

    uv pip install "spikingjelly[triton] @ git+https://github.com/fangwei123456/spikingjelly.git"

源码也可从 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ 获取。

已有源码 checkout 的开发者可执行 ``uv pip install --editable ".[triton]"``。
根目录 ``ops/`` 经安装映射为 ``spikingjelly._ops``，只设置 ``PYTHONPATH``
不能代替安装。开发环境约定见仓库 ``CONTRIBUTING.md``。

.. _install-native-cuda-cn:

手动构建原生 CUDA
--------------------------

常规 wheel 不含预编译原生动态库。当前唯一推荐的用户构建入口是下载 PyPI
源码包 sdist 并在本机编译；无需克隆仓库。以下命令需等待包含这些算子的
V2 版本发布到 PyPI，旧发行版不能据此构建当前算子。

先准备 CUDA 版 Torch、与其匹配的 CUDA Toolkit（含 ``nvcc``）和 C++ 编译器。
若工具链不在默认位置，设置 ``CUDA_HOME``；无可见 GPU 的构建需设置
``TORCH_CUDA_ARCH_LIST``，指定实际目标架构。

.. code-block:: bash

    uv pip install "setuptools>=77.0.3" ninja
    SJ_BUILD_NATIVE_CUDA=1 uv pip install \
      --no-build-isolation --no-binary spikingjelly \
      --reinstall-package spikingjelly --no-cache "spikingjelly>=2.0.0"

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - 选项
      - 作用
    * - ``SJ_BUILD_NATIVE_CUDA=1``
      - 本次源码构建尝试编译原生扩展；默认不编译。
    * - ``--no-binary spikingjelly``
      - 下载 SpikingJelly sdist；其他依赖仍可使用 wheel。
    * - ``--no-build-isolation``
      - 使用当前环境的 Torch 和构建依赖。默认隔离环境不包含我们的构建所需 Torch。
    * - ``--reinstall-package spikingjelly``
      - 即使已安装，也重新安装 SpikingJelly。
    * - ``--no-cache``
      - 不复用先前构建的 wheel，避免构建设置变化后仍使用旧产物。

该命令不会默认添加 Triton；需要时另行安装 ``spikingjelly[triton]``。
缺少 CUDA 版 Torch 或工具链时会提示并跳过原生扩展；工具链存在但编译失败时
安装会报错。安装成功本身不能证明扩展已构建，检查方法见下文。

原生扩展运行时只加载二进制，不调用编译器。当前加载器检查算子 ABI、完整 Torch
版本、CUDA 版本及目标 GPU 支持；更换环境后可能需要重建。Triton 保留自己的
首次 JIT 和缓存，不能假定所有平台的 CUDA Torch 都附带可用 Triton。

安装后如何执行
--------------------------

将模块和输入移到同一设备即可。普通注册神经元的选择流程如下；CUDA 首次使用
按设备及执行路径检查候选，随后复用绑定，不在线测速。

.. figure:: /_static/tutorials/install/execution.svg
    :alt: 普通注册神经元的执行决策树：CPU 使用 Torch；CUDA 先检查输入配置，再按 eager 或 Inductor 路径选择兼容实现。
    :width: 100%

    图中 CUDA 箭头表示候选顺序。CUDA Graph 沿用捕获函数已经选择的实现。

安装某种实现，不代表每个调用都使用它：

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - 功能
      - 启用条件
    * - 普通注册神经元
      - 受支持的输入、FP32 状态及内置替代梯度可进入融合路径。普通模块状态跟随输入 dtype，低精度状态或自定义 surrogate 可使用 Torch 参考公式。
    * - IF/LIF/PLIF 显式精度配置
      - 要求支持该组合的 CUDA Triton；不能以其他实现替换明确指定的数值策略。
    * - FlexSN CUDA 多步
      - 可编译的 core 使用 Triton；已知不支持的组合可使用 Torch/HOP。单步使用 Torch。
    * - 融合 IF/LIF-Linear、packed/sparse 投影
      - 原生扩展兼容时使用对应内核，缺失时使用 Torch 参考执行；Triton 不提供这些原生融合投影的性能特性。

输入精度、状态精度和递推精度的区别见 :doc:`./precision`；执行与编译示例见
:doc:`./triton_backend`，自定义动力学见 :doc:`./flexsn`。

确认安装与排查
--------------------------

先检查安装位置和 CUDA Torch：

.. code-block:: python

    import torch
    import spikingjelly

    print(spikingjelly.__file__)
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())

有可用 NVIDIA GPU 时，再查询普通神经元的绑定：

.. code-block:: python

    from spikingjelly.activation_based import functional

    device = torch.device("cuda:0")
    print(functional.neuron_implementation("lif", device, execution="eager"))
    print(functional.neuron_implementation("lif", device, execution="compile"))

查询初始化选择但不计算神经元输出、不推进状态。返回的 ``implementation`` 表示
绑定，``unavailable`` 解释更高优先级候选不可用的原因；低精度状态等配置仍可能
走参考路径。缺失依赖或已知不兼容可以检查下一候选，未知 JIT、kernel、OOM 或
梯度错误会直接报告。日志默认静默，启用方法与严格诊断环境变量见
:doc:`./triton_backend`。改变依赖、扩展或配置后重启进程。

其他可选依赖
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - 功能
      - 安装命令
    * - :doc:`./nir_exchange`
      - ``uv pip install "spikingjelly[nir]"``
    * - Lightning 集成
      - ``uv pip install "spikingjelly[lightning]"``
    * - Transformer Engine 精度功能
      - ``uv pip install "spikingjelly[fp8]"``；范围见 :doc:`./precision`，不等同于开启所有神经元 FP8 组合。

其他 extras 及版本约束见仓库 ``pyproject.toml``。当前包不依赖 CuPy；旧安装项
迁移见 :doc:`./migrate_from_legacy`。
