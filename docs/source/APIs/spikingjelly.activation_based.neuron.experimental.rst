Experimental device-dispatched neurons
============================================================

Device dispatch / 设备分发
--------------------------

九种实验神经元使用各自统一的 ``torch.ops.sj_<neuron>.forward`` 算子，
训练型神经元另有统一的 ``backward``。PyTorch dispatcher 根据 Tensor 的 dispatch
key 自动选择 CPU 或 CUDA 实现；节点和函数入口不再用 Python 判断 CPU/CUDA。
CPU 直接执行 Torch 参考实现。CUDA 在每个设备首次使用时选择并缓存原生 CUDA、
Triton 或 CuPy，之后只查询绑定入口；执行错误不触发重新选择。

``import spikingjelly`` 不初始化 CUDA，也不强制加载 GPU 可选依赖。
环境变量仍在对应神经元算子模块导入时读取；选择以神经元和设备为单位，
不要求一个进程中的异构 GPU 使用同一实现。编译/图捕获前仍须预热。
BF16 编译能力取决于 PyTorch 和 GPU；例如 PyTorch 2.7 的 Inductor 在 TITAN RTX
上拒绝 BF16 编译，即使 eager 可以执行。
CPU 实现注册时和首次 CUDA 绑定时会把实现名和源码指纹追加到 PyTorch 的
``torch.compiler.config.cache_key_tag``，保留调用方已有标签；这会影响进程中后续的
编译缓存键。图捕获时会保留该指纹，避免更新 CPU 实现或切换 CUDA 实现后复用旧缓存。
未预热的 CUDA ``torch.compile`` 调用会明确报错；单独 FakeTensor 推导仍不加载 GPU 实现。
CPU/Triton 后端只提供普通实现函数，不另行注册 ``cpu_forward`` 或
``triton_forward`` 等内部算子。公共 ``forward`` / ``backward`` 使用 ``triton_op``；
Triton 在 eager 中直接启动 kernel，在编译展开中通过普通函数和 ``wrap_triton``
记录 kernel 调用。fake/autograd 只在公共 CPU/Triton 算子上注册一次。

The nine experimental neurons use one ``torch.ops.sj_<neuron>.forward`` operator
per family, plus a shared-device ``backward`` operator for trainable families.
PyTorch's dispatcher selects CPU or CUDA from Tensor dispatch keys; node and
functional entry points no longer branch on device type in Python. CPU executes
the Torch reference directly. CUDA selects and caches native CUDA, Triton or
CuPy on first use of each device; later execution only looks up the binding.
Execution errors never trigger provider reselection.

``import spikingjelly`` does not initialize CUDA or force GPU optional dependency
loading. Environment settings are still read when the corresponding neuron
operator module is imported. Selection is per neuron and device, allowing
heterogeneous GPUs to use different implementations. Warmup before compilation
or graph capture is still required. BF16 compilation depends on PyTorch and GPU support; for example, PyTorch 2.7
Inductor rejects BF16 compilation on TITAN RTX even when eager execution works.
CPU registration and first CUDA binding append implementation source fingerprints to PyTorch's
``torch.compiler.config.cache_key_tag``, preserving the caller's existing tag;
this affects subsequent compiler cache keys in the process. Capture retains the
fingerprint, preventing stale decompositions after CPU implementation updates or
CUDA provider changes. CUDA compilation without warmup raises a clear error; standalone
FakeTensor inference still does not load a GPU implementation.
CPU/Triton backends expose ordinary functions and do not register internal
``cpu_forward`` or ``triton_forward`` operators. The public ``forward`` / ``backward``
use ``triton_op``. Triton launches kernels directly in eager execution; compiler
decomposition follows plain functions and ``wrap_triton`` to record kernel calls.
Fake/autograd behavior is registered once on the public CPU/Triton operators.

Registration scope / 注册范围
------------------------------

加载各神经元 ops 子包时即注册 schema、CPU/CUDA 入口、fake 及适用的 autograd。
CUDA 入口始终存在；首次 CUDA 执行只填充实现缓存，不替换 dispatch slot。
``ops/dispatch.py`` 拥有注册对象，``ops/selection.py`` 拥有每种神经元按设备索引
保存的前后向选择；节点不保存第二份选择。原生 CUDA 仍通过它的 C++ 注册算子执行，
因此其 eager 路径还有一层算子调用。CuPy 保留内部 ``custom_op`` 作为编译器不能
展开的外部执行边界，eager 仍直接调用实现函数。这两处边界与已删除的 CPU/Triton
重复包装不同；该架构不承诺三种实现有相同的主机开销。

Each neuron ops subpackage registers its schema, CPU/CUDA entries, fake behavior
and applicable autograd when imported. The CUDA entry already exists before the
first CUDA execution, which only fills the implementation cache and never replaces
a dispatch slot. ``ops/dispatch.py`` owns registration lifetimes;
``ops/selection.py`` owns each family's forward/backward selection by device
index. Nodes do not hold a second selection. Native CUDA still calls its registered
C++ operator, adding an operator call in eager execution. CuPy retains an internal
``custom_op`` as an opaque external-execution boundary for compilation, while
eager calls its implementation function directly. These necessary boundaries
differ from the removed CPU/Triton duplicate registrations. This architecture
does not promise equal host overhead across providers.

.. list-table:: Fixed neuron operators / 固定神经元算子
   :header-rows: 1

   * - Families / 神经元
     - Installed packages / 安装后子包
     - Execution / 执行
   * - IF, LIF, PLIF
     - ``spikingjelly._ops.if_``, ``lif``, ``plif``
     - CPU + CUDA; forward/backward / 前后向
   * - QIF, EIF, Izhikevich
     - ``spikingjelly._ops.qif``, ``eif``, ``izhikevich``
     - CPU + CUDA; forward/backward / 前后向
   * - I-LIF
     - ``spikingjelly._ops.ilif``
     - CPU + CUDA; count STE / 计数 STE 前后向
   * - ActivationAwareIF, STBIF
     - ``spikingjelly._ops.activation_aware_if``, ``stbif``
     - CPU + CUDA; inference only / 仅推理

所有表中 CUDA 入口均使用原生 CUDA → Triton → CuPy 的候选顺序。
下面的既有路径保留各自契约，不应被理解成已经自动切换到表中算子：

* 生产神经元的显式 ``backend`` 接口及 ``cupy_generated``、
  ``cupy_single_step``、``triton_precision``：保留低精度状态舍入、FP8、替代梯度及
  training/eval 语义，不能静默替换成实验算子的 FP32 状态契约。
* FlexSN：``core``、状态数量及生成图决定 kernel，缓存必须关联生成图及其生命周期，
  不能仅按设备共享一个 kernel。Torch/HOP 是可追踪 scan，Triton 是动态生成的算子；
  当前没有等价的原生 CUDA/CuPy 生成器。前端保留 core/状态，后端位于 ``ops/flexsn``。
* ``ops/if_linear``、``ops/lif_linear``：现有公开接口仅支持 CUDA FP32/CuPy，
  并保留融合输出和反向重算契约；没有同 schema 的 CPU 或其他 GPU 实现。
  本轮不新增融合后端，也不伪造候选选择。
* 仅由 Torch 运算组成的其他神经元直接复用 PyTorch 分发；Lava 路径属于外部运行时，
  不属于本实验的 CPU/NVIDIA CUDA Tensor 算子。

Every listed CUDA entry uses the native CUDA → Triton → CuPy candidate order.
The following existing paths keep their own contracts and do not automatically
switch to the operators in the table:

* Production neurons' explicit ``backend`` interfaces and ``cupy_generated``,
  ``cupy_single_step`` and ``triton_precision`` retain their low-precision state
  rounding, FP8, surrogate and training/evaluation semantics. They cannot silently
  adopt the experimental FP32-state contract.
* FlexSN kernels depend on the user ``core``, state arity and generated graphs.
  Caches must follow graph identity and lifetime, rather than share one kernel
  solely by device. Torch/HOP execute traceable scans; Triton uses generated
  operators. Equivalent native CUDA/CuPy generators do not exist. The frontend
  retains core/state ownership and the backend lives in ``ops/flexsn``.
* ``ops/if_linear`` and ``ops/lif_linear`` expose CUDA FP32/CuPy-only interfaces
  with fused outputs and backward recomputation. No CPU or alternative GPU
  implementation has the same schema. This stage adds neither fused backends nor
  an artificial candidate selector.
* Other Torch-only neurons already use PyTorch device dispatch through their
  constituent operations. Lava integrates an external runtime outside this
  CPU/NVIDIA CUDA Tensor experiment.

Additional neuron wrappers / 新增神经元封装
---------------------------------------------

除 IF/LIF/PLIF 外，模块还提供 ``ExperimentalQIFNode``、``ExperimentalEIFNode``、
``ExperimentalIzhikevichNode``、``ExperimentalILIFNode``、
``ExperimentalActivationAwareIFNode`` 和 ``ExperimentalSTBIFNode``。
这些类使用独立的 ops 子包并自动选择实现，输入支持 FP32/FP16/BF16，状态为 FP32。
QIF/EIF/Izhikevich 使用七种二值替代梯度；I-LIF 使用 MultiLevelSpikeCount 的计数和
闭区间 STE；ActivationAwareIF/STBIF 仅推理。状态在连续调用间保留，reset() 清空，
形状变化时重建；Izhikevich 另保存恢复状态 w/w_seq，STBIF 保存 q/acc_q/cur_output。
这些封装不继承 MemoryModule，不接受 backend，也不改变生产节点的状态管理。

Beyond IF/LIF/PLIF, the module provides ``ExperimentalQIFNode``,
``ExperimentalEIFNode``, ``ExperimentalIzhikevichNode``, ``ExperimentalILIFNode``,
``ExperimentalActivationAwareIFNode``, and ``ExperimentalSTBIFNode``. Each uses
its own ops package and automatic implementation selection, with FP32/FP16/BF16
inputs and FP32 state. QIF/EIF/Izhikevich use seven binary surrogates; I-LIF uses
MultiLevelSpikeCount and its inclusive STE; ActivationAwareIF/STBIF are inference
only. State persists across calls, reset() clears it, and shape changes recreate
it. Izhikevich also keeps w/w_seq; STBIF keeps q/acc_q/cur_output. These wrappers
do not inherit MemoryModule or accept backend, and do not change production
nodes' state management.

Compatibility matrix / 兼容矩阵
-------------------------------

Applies to IF, LIF and PLIF / 适用于 IF、LIF 和 PLIF。

.. list-table:: Input dtype support / 输入 dtype 支持
   :header-rows: 1

   * - Input / 输入
     - CPU (Torch)
     - CUDA (native / 原生)
     - CUDA (Triton)
     - CUDA (CuPy)
   * - FP32
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持
   * - FP16
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持
   * - BF16
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持
     - Yes / 支持

Every entry supports all seven surrogates listed below; state arithmetic is FP32.
每一项均支持下面列出的七种替代梯度，状态运算使用 FP32。

中文
----

``ExperimentalIFNode``、``ExperimentalLIFNode`` 和 ``ExperimentalParametricLIFNode``
是独立的实验性模块，不替换现有神经元，也不继承 ``MemoryModule``。
通过下面的路径显式导入；不接受 ``backend`` 或 ``step_mode``。
支持 CPU/NVIDIA CUDA、FP32/FP16/BF16、非空多步序列和七种替代梯度：
Sigmoid、ATan、PiecewiseQuadratic、PiecewiseExp、SoftSign、SuperSpike、Erf。
支持 autocast 输入；不支持 FP64、FP8 或高阶梯度。IF 不含泄漏项；LIF 使用固定 ``tau``；PLIF
使用共享的可学习零维 FP32 参数 ``w``，初始值为 ``-log(init_tau - 1)``，
每次计算使用 ``sigmoid(w)`` 作为 ``1/tau``，不缓存派生时间常数。

.. code-block:: python

   import torch
   from spikingjelly.activation_based.neuron.experimental import ExperimentalLIFNode

   node = ExperimentalLIFNode(store_v_seq=True)
   spikes = node(torch.randn(8, 2, 32, requires_grad=True))
   (spikes.sum() + node.v.sum()).backward()
   node.reset()

PLIF 的参数支持 FP32/FP16/BF16，必须与输入同 device；使用 ``.to(device)`` 移动模块。
``reset()`` 只清空神经元状态，不改变 ``w``。优化器和 ``state_dict`` 正常管理该参数：

.. code-block:: python

   import torch
   from spikingjelly.activation_based.neuron.experimental import ExperimentalParametricLIFNode

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   node = ExperimentalParametricLIFNode(init_tau=3.0).to(device)
   optimizer = torch.optim.SGD(node.parameters(), lr=0.01)
   spikes = node(torch.ones(4, 8, device=device))
   (spikes.sum() + node.v.sum()).backward()
   optimizer.step()
   optimizer.zero_grad()
   node.reset()

``v`` 和 ``v_seq`` 是非持久 buffer，不进入 ``state_dict``。连续调用保留膜电位及
计算图；``reset()`` 清空状态。输入形状改变时重新初始化；形状相同时将状态转到
输入的 device 并保持 FP32。``v_seq`` 仅记录最近一次调用。训练与 eval 模式始终输出
硬脉冲，且启用梯度时都采用所选替代梯度；这与生产级神经元的 eval
语义不同，不应将这些模块视为可直接替换的实现。

默认 ``store_v_seq=False`` 时只分配并返回最终重置电位；不会写出完整重置后
电位轨迹。反向所需的充电电位 workspace 仍按时间序列保存。
函数式算子额外接受 ``store_v_seq``，默认 True 保持原有完整轨迹返回值；
False 时第二个输出与初态同形状，支持最终状态及初态梯度。
该算子 schema 变化需要重新构建已有原生 CUDA 扩展，加载器会拒绝旧 schema。

CPU 使用 Torch 运算。CUDA 在首次使用每个设备时按原生 CUDA、Triton、CuPy 的
顺序检查可用性并缓存选择，不在每次 forward 中重新遍历候选。可在启动进程前设置
对应环境变量为 ``SJ_IF_CUDA_IMPLEMENTATION``、``SJ_LIF_CUDA_IMPLEMENTATION``
和 ``SJ_PLIF_CUDA_IMPLEMENTATION``，取值均为 ``auto|cuda|triton|cupy``，默认 ``auto``。
每种算子独立选择实现。
强制指定的实现不可用时明确报错，不回退。进程运行中不支持修改该配置。

.. code-block:: python

   import torch
   from spikingjelly._ops.lif import get_cuda_implementation
   from spikingjelly.activation_based.neuron.experimental import ExperimentalLIFNode

   # 解析并注册实现；这一步不执行 kernel。
   print(get_cuda_implementation(torch.device("cuda", 0)))
   node = ExperimentalLIFNode()
   x = torch.randn(8, 2, 32, device="cuda:0", requires_grad=True)
   node(x).sum().backward()
   node.reset()
   x.grad = None
   torch.cuda.synchronize()

返回字典的 ``implementation`` 表示选中实现，``unavailable`` 记录未选用候选的
不可用原因。该函数仅选择、加载和注册实现，不执行 Triton/CuPy 的 kernel JIT。
在编译或 CUDA Graph 捕获前，应使用实际工作负载的设备、形状和 dtype，在捕获区域
外预热代表性的前向与反向，如上例所示；随后清空状态和梯度。
选定实现运行时报错不会触发静默回退。

默认安装无需 CUDA 编译器，也不编译原生扩展。Triton/CuPy 按需使用各自的 JIT，
需要安装对应可选依赖。若要构建原生 CUDA 实现，请先安装 PyTorch 和匹配的 CUDA
开发工具链，在仓库根目录运行：

.. code-block:: bash

   uv pip install 'setuptools>=77.0.3' ninja
   SJ_BUILD_NATIVE_CUDA=1 uv pip install --no-build-isolation .

原生扩展在安装阶段编译，在使用时加载；没有隐藏的运行时 CUDA 源码编译。

构建由 ``pyproject.toml`` 中的 ``setuptools.build_meta`` 驱动，遵循 PEP 517/660。
元数据、包目录映射和头文件规则均在 TOML 中；``ops/`` 自动安装为
``spikingjelly._ops``。``setup.py`` 仅保留可选 CUDA 的工具链检查、
``CUDAExtension`` 配置及构建兼容信息，不作为命令行入口。
使用 ``uv build`` 构建发布产物：默认先构建 sdist，再从它构建 wheel，避免复用
工作区中的陈旧构建输出。默认隔离构建只需要 setuptools，不会安装 Torch 或 CUDA 工具链。


精度与替代梯度契约
~~~~~~~~~~~~~~~~~~

所有实现采用同一精度策略：脉冲和输入梯度跟随输入 dtype；初始状态、膜电位输出、
充电 workspace、替代梯度计算和跨时间梯度累积均为 FP32。函数式接口要求传入 FP32
初态；节点自动创建 FP32 状态。PLIF 在 FP32 中计算 ``sigmoid(w)`` 和梯度归约，
最后将梯度转换到 w dtype。autocast 训练建议保留 FP32 模型参数。
算子不主动转换输入 dtype：autocast 中由上游 Linear/Conv 等决定实际输入 dtype，
直接传入 FP32 时仍返回 FP32 脉冲。
这不等同于生产级低精度后端每步舍入膜电位的语义，不能保证两者脉冲逐位一致。

构造时可传入 ``surrogate_function=surrogate.ATan(alpha=2.0)`` 等对象。
只支持列出的精确类型、``spiking=True`` 以及固定的有限正 alpha；构造时读取
类型和 alpha，随后修改原替代梯度对象无效。省略时使用 ``Sigmoid(alpha)``。
原生 CUDA/CuPy 通过模板、Triton 通过 constexpr 专门化替代梯度，时间循环内没有
动态选择分支。原生扩展 ABI 已更新为 3，已有本地扩展需要重新构建。

.. code-block:: python

   import torch
   from spikingjelly.activation_based import surrogate
   from spikingjelly.activation_based.neuron.experimental import ExperimentalLIFNode

   node = ExperimentalLIFNode(surrogate_function=surrogate.ATan(alpha=2.0))
   x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
   spikes = node(x)
   assert spikes.dtype == x.dtype and node.v.dtype == torch.float32
   (spikes.float().sum() + node.v.sum()).backward()
   node.reset()

English
-------

``ExperimentalIFNode``, ``ExperimentalLIFNode`` and ``ExperimentalParametricLIFNode``
are independent experimental modules, not replacements for existing neurons or
``MemoryModule`` subclasses. Import them explicitly as shown above; there is no
``backend`` or ``step_mode`` argument. They support CPU/NVIDIA CUDA,
FP32/FP16/BF16 nonempty sequences and seven surrogates: Sigmoid, ATan,
PiecewiseQuadratic, PiecewiseExp, SoftSign, SuperSpike and Erf. Autocast inputs
are supported; FP64, FP8 and higher-order gradients are unsupported. IF has no leak; LIF uses fixed ``tau``;
PLIF uses one shared learnable scalar FP32 parameter ``w``, initialized to
``-log(init_tau - 1)``. Each call uses ``sigmoid(w)`` as ``1/tau`` without caching
a derived time constant. PLIF's FP32/FP16/BF16 parameter must share the input device;
move the module with ``.to(device)``. Optimizers and ``state_dict`` manage ``w``
normally, and ``reset()`` preserves it. The PLIF example above performs one update.

``v`` and ``v_seq`` are nonpersistent buffers excluded from ``state_dict``.
Successive calls preserve voltage and its autograd graph; ``reset()`` clears both
states. A changed input shape reinitializes voltage; otherwise state follows the
input device while staying FP32. ``v_seq`` records only the latest call. Both training and
evaluation emit hard spikes and use the selected surrogate gradients when gradients
are enabled. This differs from production neuron evaluation semantics, so
these modules are not drop-in replacements.

With the default ``store_v_seq=False``, only the final reset voltage is allocated
and returned; the full reset-voltage trace is not written. The charged-voltage
workspace needed for backward still spans the sequence. Functional operators
also accept ``store_v_seq``: True (the default) preserves their full-trace return,
while False makes the second output match the initial state's shape, retaining
final-state and initial-state gradients. Rebuild existing native CUDA extensions
after this schema change; the loader rejects the old schema.

CPU uses Torch operations. On first use of each CUDA device, availability is
checked in native CUDA, Triton, CuPy order and the result is cached. Forward calls
do not repeat candidate selection. Set
``SJ_IF_CUDA_IMPLEMENTATION``, ``SJ_LIF_CUDA_IMPLEMENTATION``, or
``SJ_PLIF_CUDA_IMPLEMENTATION`` to ``auto|cuda|triton|cupy`` before starting the
process; the default is ``auto``. Selection is independent for each operator.
A forced unavailable implementation raises an error
without fallback. Changing the setting within a running process is unsupported.
``get_cuda_implementation`` only selects, loads, and registers an implementation;
it does not run Triton/CuPy kernel JIT compilation. Its result contains the selected
``implementation`` and ``unavailable`` candidate reasons. Before ``torch.compile``
or CUDA Graph capture, warm up representative forward and backward calls outside
capture, using the workload's device, shape, and dtype as shown above, then clear
state and gradients. Execution failures do not silently select another implementation.

The default installation needs no CUDA compiler and does not build the native
extension. Triton/CuPy use their own JIT and require the corresponding optional
dependencies. To build native CUDA, install PyTorch and a matching CUDA developer
toolchain, then install the build dependencies and run the installation commands
above from the repository root. Build isolation is disabled, so these dependencies
must already exist in the environment.
Native CUDA compiles at installation time and loads at runtime; there is no
implicit runtime compilation of its CUDA source.

The PEP 517/660 build is driven by ``setuptools.build_meta`` in ``pyproject.toml``.
TOML owns metadata, package-directory mappings and header-file rules; ``ops/``
automatically installs as ``spikingjelly._ops``. ``setup.py`` only provides the
optional CUDA toolchain checks, ``CUDAExtension`` configuration and compatibility
metadata; it is not a command-line build entry point. Use ``uv build`` for release
artifacts: it builds an sdist and then a wheel from that sdist by default, avoiding
stale workspace build outputs. Default isolated builds require only setuptools,
without installing Torch or a CUDA toolchain.


Precision and surrogate contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All implementations use the same precision policy: spikes and input gradients
follow the input dtype. Initial state, voltage outputs, charged workspace,
surrogate arithmetic and temporal-gradient accumulation are FP32. Functional
operators require FP32 initial state; nodes create it automatically. PLIF computes
``sigmoid(w)`` and gradient reduction in FP32, then casts the gradient to w dtype.
Retain FP32 model parameters for autocast training. Operators do not implicitly
cast inputs: upstream Linear/Conv operations determine their dtype under autocast;
direct FP32 inputs still produce FP32 spikes. This differs from production
low-precision backends that round voltage at every step; identical spikes are not
guaranteed between those policies.

Pass an object such as ``surrogate_function=surrogate.ATan(alpha=2.0)`` at
construction. Only the listed exact types with ``spiking=True`` and fixed finite
positive alpha are supported. Construction snapshots type and alpha; subsequent
changes to the original surrogate object have no effect. Omitting the object uses
``Sigmoid(alpha)``. Native CUDA/CuPy templates and Triton constexpr specialize the
surrogate, avoiding dynamic selection inside the time loop. The native extension
ABI is now 3; rebuild existing local extensions. The BF16 example above also runs
on CPU and demonstrates FP32 state with input-dtype spikes and gradients.

Reference
---------

.. automodule:: spikingjelly.activation_based.neuron.experimental
   :members:
   :undoc-members:
   :show-inheritance:

Experimental operator interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: spikingjelly._ops.lif.lif

.. autofunction:: spikingjelly._ops.lif.get_cuda_implementation

.. autofunction:: spikingjelly._ops.if_.if_multi_step

.. autofunction:: spikingjelly._ops.if_.get_cuda_implementation

.. autofunction:: spikingjelly._ops.plif.plif

.. autofunction:: spikingjelly._ops.plif.get_cuda_implementation
