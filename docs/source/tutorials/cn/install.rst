安装指南
==========================

English version: :doc:`../en/install`

准备环境
--------------------------

SpikingJelly 2.0 需要 Python 3.11 及以上版本和 PyTorch 2.6 及以上版本。先创建虚拟环境，
再按 `PyTorch 官方安装页 <https://pytorch.org/get-started/locally/>`_ 为你的设备安装
Torch、torchvision 和 torchaudio：

.. code-block:: bash

    uv venv --python 3.11
    source .venv/bin/activate

这三个包请用官方页面给出的同一条命令一起安装。只装其中一部分时，安装器可能为了补齐
其余的包而换掉已装的 Torch。

本页按开发源码编写；使用 PyPI 发行版时，请阅读对应版本的文档。

安装 SpikingJelly
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - 安装方式
      - 命令
    * - PyPI 发布版
      - ``uv pip install spikingjelly``
    * - PyPI 先行版
      - ``uv pip install --pre spikingjelly``
    * - 发布版 + Triton
      - ``uv pip install "spikingjelly[triton]"``
    * - 最新源码开发版
      - ``uv pip install git+https://github.com/fangwei123456/spikingjelly.git``

只用 CPU 时，安装发布版即可。使用 NVIDIA GPU 时，建议安装带 Triton 的版本，CUDA 上的
IF、LIF、PLIF 等神经元会自动使用 Triton 内核。如果环境中已有与 Torch 匹配的 Triton
（3.2.0 及以上），无需重复安装，可以这样查看版本：

.. code-block:: bash

    python -c "import triton; print(triton.__version__)"

原生 CUDA 扩展需要手动构建，只有在使用融合投影等原生算子，或实测原生实现更快时才需要，
步骤见 :ref:`install-native-cuda-cn`。Triton 和原生扩展可以同时安装，模块无需配置执行后端。

.. figure:: /_static/tutorials/install/installation.svg
    :alt: 安装决策树：准备 Python 和 Torch 后，CPU 普通安装；NVIDIA CUDA 可使用参考实现、可选 Triton 或手动构建原生 CUDA。
    :width: 100%

    根据设备和需求选择安装方式。

开发版需要 Triton 时：

.. code-block:: bash

    uv pip install "spikingjelly[triton] @ git+https://github.com/fangwei123456/spikingjelly.git"

源码也可从 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ 获取。

已有源码工作副本的开发者可执行 ``uv pip install --editable ".[triton]"``。
根目录 ``ops/`` 经安装映射为 ``spikingjelly._ops``，只设置 ``PYTHONPATH``
不能代替安装。开发环境约定见仓库 ``CONTRIBUTING.md``。

安装完成后运行 ``uv pip check``，检查依赖版本是否一致。

.. _install-native-cuda-cn:

手动构建原生 CUDA
--------------------------

常规 wheel 不含预编译原生动态库。当前推荐手动源码构建，默认从 PyPI 下载
sdist 并在本机编译，无需克隆仓库。以下命令要求 SpikingJelly >= 2.0.0；
旧发行版不含这些算子。

先准备 CUDA 版 Torch、与其匹配的 CUDA Toolkit（含 ``nvcc``）和 C++ 编译器。
若工具链不在默认位置，设置 ``CUDA_HOME``；无可见 GPU 的构建需设置
``TORCH_CUDA_ARCH_LIST``，指定实际目标架构。

.. code-block:: bash

    uv pip install "setuptools>=77.0.3" ninja
    SJ_BUILD_NATIVE_CUDA=1 uv pip install \
      --no-build-isolation --no-binary spikingjelly \
      --reinstall-package spikingjelly --no-cache "spikingjelly>=2.0.0"

构建成功后，在可用的目标 NVIDIA GPU 上启动新进程，检查默认 eager 绑定：

.. code-block:: bash

    python -c \
      'import torch; from spikingjelly.activation_based import functional; print(functional.neuron_implementation("lif", torch.device("cuda:0")))'

兼容的原生扩展预期返回 ``implementation: cuda``；若为 ``triton`` 或 ``torch``，
检查 ``unavailable`` 中的原生加载原因。查询不运行神经元；状态精度等配置仍会影响
实际调用路径。完整检查见下文。

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - 选项
      - 作用
    * - ``SJ_BUILD_NATIVE_CUDA=1``
      - 本次源码构建要求编译原生扩展；默认不编译。
    * - ``--no-binary spikingjelly``
      - 下载 SpikingJelly sdist；其他依赖仍可使用 wheel。
    * - ``--no-build-isolation``
      - 使用当前环境的 Torch 和构建依赖。默认隔离环境不包含构建所需的 Torch。
    * - ``--reinstall-package spikingjelly``
      - 即使已安装，也重新安装 SpikingJelly。
    * - ``--no-cache``
      - 不复用先前构建的 wheel，避免构建设置变化后仍使用旧产物。

该命令不会默认添加 Triton；需要时另行安装 ``spikingjelly[triton]``。
显式要求构建时，缺 CUDA 版 Torch、CUDA Toolkit、编译器或必要架构配置会立即报错；
实际编译失败也会终止安装。未设置环境变量时仍构建纯 Python 包。
该检查只在源码构建中生效；安装已有 wheel 或跳过已有安装不会执行构建检查。
命令保留重安装及禁用缓存参数，因为 ``--no-binary`` 仍可能复用 uv 缓存的 wheel。

PyPI 下载不可用或需要开发源码时，可从 OpenI 获取源码工作副本，沿用同一手动构建：

.. code-block:: bash

    git clone https://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    uv pip install "setuptools>=77.0.3" ninja
    SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .

也可从 GitHub 克隆源码。uv 对命令行明确传入的本地目录重新构建安装，因此无需
sdist 的 ``--no-binary`` 等参数。可编辑安装中修改 ``.cu`` 或头文件后需要再次执行
原生构建；不会像 Python 源码那样即时生效。

原生扩展运行时只加载二进制，不调用编译器。当前加载器检查算子 ABI、完整 Torch
版本、CUDA 版本及目标 GPU 支持；更换环境后可能需要重建。Triton 保留自己的
首次即时编译和缓存，其宿主启动器需要可用的 C 编译器（例如 GCC）；精简运行时
镜像即使不构建原生 CUDA 扩展，也可能需要补装编译器。若 CUDA 版 Torch 未附带可用的 Triton，请单独安装。

安装后如何执行
--------------------------

将模块和输入移到同一设备即可。已注册的普通神经元的选择流程如下；CUDA 首次使用
按设备及执行路径检查候选，随后复用绑定，不在线测速。

.. figure:: /_static/tutorials/install/execution.svg
    :alt: 已注册的普通神经元的执行决策树：CPU 使用 Torch；CUDA 先检查输入配置，再按 eager 或 Inductor 路径选择兼容实现。
    :width: 100%

    图中 CUDA 箭头表示候选顺序。CUDA Graph 沿用捕获函数已经选择的实现。

安装某种实现，不代表每个调用都使用它：

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - 功能
      - 启用条件
    * - 已注册的普通神经元
      - 受支持的输入、FP32 状态及内置替代梯度可进入融合路径。普通模块状态跟随输入数据类型，低精度状态或自定义替代梯度可使用 Torch 参考公式。
    * - ComplementaryLIF（CLIF）
      - 同形状、同设备、同数据类型的 CUDA FP32/FP16/BF16 张量可使用可选 Triton 内核，保留所选放电函数和状态数据类型。自动模式在训练和推理中均可选择 Triton，也支持 ``torch.compile``；小批量可能更慢。CPU、自定义放电函数及其他不支持的输入使用 Torch 公式；显式请求 Triton 时，不支持的 CUDA 输入配置会报错。可设 ``SJ_CLIF_CUDA_IMPLEMENTATION=auto|triton|torch``；没有原生 CUDA CLIF 内核。
    * - IF/LIF/PLIF 显式精度配置
      - 要求支持该组合的 CUDA Triton；不能以其他实现替换明确指定的数值策略。
    * - FlexSN CUDA 多步
      - 可编译的单步函数使用 Triton；已知不支持的组合可使用 Torch/HOP。单步使用 Torch。
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

    import torch
    from spikingjelly.activation_based import functional

    device = torch.device("cuda:0")
    print(functional.neuron_implementation("lif", device, execution="eager"))
    print(functional.neuron_implementation("lif", device, execution="compile"))

查询初始化选择但不计算神经元输出、不推进状态。返回的 ``implementation`` 表示
绑定，``unavailable`` 解释更高优先级候选不可用的原因；低精度状态等配置仍可能
使用参考公式。缺失依赖或已知不兼容可以检查下一候选，未预期的即时编译、内核、内存不足或
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

其他可选依赖及版本约束见仓库 ``pyproject.toml``。CuPy 安装选项的迁移见 :doc:`./migrate_from_legacy`。
