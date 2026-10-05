Neuron State Updates
++++++++++++++++++++

``*_step`` 表示一次完整状态更新。``*_multi_step`` 仅表示具有独立序列实现的路径；
backend 专用路径在函数名中标明 backend。

----

``*_step`` denotes one complete state update. ``*_multi_step`` is reserved for an
independently implemented sequence path, with backend-specific paths naming the
backend explicitly.

Experimental registered kernels / 实验注册内核
----------------------------------------------

``*_registered`` 提供无 ``backend`` 参数的显式状态接口，算子代码位于根目录
``ops/``，安装为同一发行包内的 ``spikingjelly._ops``。覆盖现有 CuPy/Triton
神经元类型及其单步/序列入口，FlexSN 除外。生产节点和 backend 专用函数现在也从
``ops`` 加载实现，同时保留原来的精度、布局及自定义替代梯度契约。
这些生产契约由各神经元的 ``cupy_generated.py``、``cupy_single_step.py`` 和
``triton_precision.py`` 实现；``*_registered`` 继续提供显式 FP32 状态契约。
原 ``cuda_kernel`` / ``triton_kernel`` 实现暂时保留为独立对照，FlexSN 的前端追踪与后端执行已分离，后端位于 ``spikingjelly._ops.flexsn``。
``lava_cuba_lif_step`` 本来就是 Lava exchange 的 Torch 路径，且有独立的定点量化
及可学习衰减契约，不属于这次 CuPy/Triton 内核迁移。

The ``*_registered`` functions provide explicit-state interfaces without a
``backend`` argument. Kernel code lives in root ``ops/``, installed as
``spikingjelly._ops`` in the same distribution. They cover existing CuPy/Triton
neuron types and single-step/sequence entry points, excluding FlexSN. Existing
production nodes and backend-specific functions now load implementations from
``ops`` while retaining precision, layout, and custom-surrogate contracts through
each neuron's ``cupy_generated.py``, ``cupy_single_step.py``, and
``triton_precision.py``. The ``*_registered`` interfaces retain their explicit
FP32-state contract. Original ``cuda_kernel`` / ``triton_kernel`` implementations
remain independently callable references; FlexSN separates frontend capture from execution in ``spikingjelly._ops.flexsn``.
``lava_cuba_lif_step`` is already the Torch path of Lava exchange,
with a separate fixed-point quantization and learnable-decay contract, outside
this CuPy/Triton kernel migration.

.. list-table::
   :header-rows: 1

   * - Neuron / 神经元
     - CPU + native CUDA + Triton + CuPy
     - Gradient / 梯度
   * - IF, LIF, PLIF
     - FP32 / FP16 / BF16
     - Seven binary surrogates / 七种二值替代梯度
   * - QIF, EIF, Izhikevich
     - FP32 / FP16 / BF16
     - Seven binary surrogates / 七种二值替代梯度
   * - I-LIF
     - FP32 / FP16 / BF16
     - MultiLevelSpikeCount STE window / 脉冲计数 STE 窗口
   * - ActivationAwareIF, STBIF
     - FP32 / FP16 / BF16
     - Inference only / 仅推理

所有新入口要求 FP32 初态；膜电位、恢复状态、workspace 和跨时间梯度累积为 FP32。
输出脉冲和输入梯度跟随输入 dtype。Izhikevich 保留两个初态梯度、可选双状态轨迹，
以及原始 Torch 方程中未 detach 的恢复脉冲和硬重置 ``spike*v_reset`` 项。
I-LIF 使用 ties-to-even 的计数舍入和闭区间 STE 窗口。
ActivationAwareIF 支持标量或通道阈值/偏移；STBIF 保留残余、累积及当前输出状态。
这两个推理入口拒绝 ``requires_grad=True`` 的输入、状态及参数。
不支持 FP64、FP8、高阶梯度或自定义 surrogate 子类。

All new interfaces require FP32 initial states; voltages, recovery states,
workspaces, and temporal gradient accumulation use FP32. Output spikes and input
gradients follow the input dtype. Izhikevich retains both initial-state gradients,
optional dual-state traces, and the undetached recovery spike and hard-reset
``spike*v_reset`` terms of the original Torch equations. I-LIF uses ties-to-even
count rounding and an inclusive STE window. ActivationAwareIF supports scalar or
channel thresholds/offsets; STBIF retains residual, accumulated, and current-output
states. These two inference interfaces reject inputs, states, or parameters with
``requires_grad=True``. FP64, FP8, higher-order gradients, and custom surrogate
subclasses are unsupported.

.. code-block:: python

   import torch
   from spikingjelly.activation_based import functional, surrogate

   x = torch.randn(4, 8, requires_grad=True)
   v = torch.zeros(8, requires_grad=True)
   spikes, final, trace = functional.qif_multi_step_registered(
       x, v, surrogate_function=surrogate.ATan(), store_v_seq=True
   )
   (spikes.sum() + final.sum() + trace.sum()).backward()

CUDA 默认每设备选择原生 CUDA → Triton → CuPy 并缓存；反向绑定同一实现。
每种神经元分别拥有独立 ``ops`` 子包、注册 namespace、原生扩展和设备选择，
不使用共享的模型编号或参数数组。IF/LIF/PLIF 沿用各自环境变量，新增变量为
``SJ_QIF_CUDA_IMPLEMENTATION``、``SJ_EIF_CUDA_IMPLEMENTATION``、
``SJ_IZHIKEVICH_CUDA_IMPLEMENTATION``、``SJ_ILIF_CUDA_IMPLEMENTATION``、
``SJ_ACTIVATION_AWARE_IF_CUDA_IMPLEMENTATION``、``SJ_STBIF_CUDA_IMPLEMENTATION``。
所有配置在进程启动前设置，值为
``auto|cuda|triton|cupy``；强制实现不可用时报错，执行错误不触发回退。
使用 ``functional.registered_neuron_implementation(name, device)`` 查询实际选择。
编译或图捕获前须完成设备选择，并预热实际形状/dtype 的前向和所需反向。

CUDA selects and caches native CUDA → Triton → CuPy per device, with backward
bound to the same implementation. Each neuron owns an independent ``ops`` package,
registered namespace, native extension, and device selection; there is no shared
model ID or parameter array. IF/LIF/PLIF retain their existing settings. The added
variables are ``SJ_QIF_CUDA_IMPLEMENTATION``, ``SJ_EIF_CUDA_IMPLEMENTATION``,
``SJ_IZHIKEVICH_CUDA_IMPLEMENTATION``, ``SJ_ILIF_CUDA_IMPLEMENTATION``,
``SJ_ACTIVATION_AWARE_IF_CUDA_IMPLEMENTATION``, and
``SJ_STBIF_CUDA_IMPLEMENTATION``.
Set configurations before process startup to ``auto|cuda|triton|cupy``. Unavailable
forced implementations raise errors; execution errors do not trigger fallback.
Query the actual choice with
``functional.registered_neuron_implementation(name, device)``. Select the device
implementation and warm up actual shapes/dtypes and needed backward paths before
compilation or graph capture.

之前实验版本的分组变量 ``SJ_DYNAMICS_CUDA_IMPLEMENTATION`` 和
``SJ_INFERENCE_CUDA_IMPLEMENTATION`` 已移除，不再读取。使用过前者时，请分别
设置 QIF/EIF/Izhikevich/I-LIF 的变量；使用过后者时，请分别设置
ActivationAwareIF/STBIF 的变量。若希望各神经元使用同一实现，给对应变量设置相同值。

The previous experimental grouped variables ``SJ_DYNAMICS_CUDA_IMPLEMENTATION``
and ``SJ_INFERENCE_CUDA_IMPLEMENTATION`` are removed and no longer read. Replace
the former with the individual QIF/EIF/Izhikevich/I-LIF variables, and the latter
with ActivationAwareIF/STBIF variables. Set them to the same value when the same
implementation is desired for all affected neurons.

源码目录为 ``ops/{if_,lif,plif,qif,eif,izhikevich,ilif,activation_aware_if,stbif}``。
各目录有 ``cpu.py``、``cupy.py``、``triton.py``、``native.py``、``native.cu`` 和
``kernels.cuh``，训练模型还拥有 ``autograd.py``，推理模型拥有 ``validation.py``。
共享工具使用普通文件名，例如 ``selection.py``、``native_loader.py``、
``cupy_loader.py``、``surrogate.py`` 和 ``triton_surrogate.py``；内部对象用
下划线命名，包安装路径仍为 ``spikingjelly._ops``。

Source packages are ``ops/{if_,lif,plif,qif,eif,izhikevich,ilif,activation_aware_if,stbif}``.
Each has ``cpu.py``, ``cupy.py``, ``triton.py``, ``native.py``, ``native.cu``, and
``kernels.cuh``. Training models own ``autograd.py``; inference models own
``validation.py``. Shared tools use ordinary filenames such as ``selection.py``,
``native_loader.py``, ``cupy_loader.py``, ``surrogate.py``, and
``triton_surrogate.py``. Internal objects use underscore names; the installed
package remains ``spikingjelly._ops``.

默认安装仍为纯 Python，CPU 不依赖 GPU 可选包。CuPy/Triton 使用各自 JIT。
原生 CUDA 沿用 ``SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .``，
安装时构建新增扩展，运行时只加载，不调用编译器。不发布预编译 CUDA wheel。

Default installation remains pure Python, and CPU needs no optional GPU packages.
CuPy/Triton use their own JIT. Native CUDA retains the opt-in
``SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .`` source installation,
building the additional extensions at installation time and only loading them at
runtime. No precompiled CUDA wheels are published.

.. automodule:: spikingjelly.activation_based.functional.neuron
   :members:
   :undoc-members:
   :show-inheritance:
