算子计数与能耗估计
=========================================

本教程作者： `黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/op_counter`

本教程介绍 ``spikingjelly.activation_based.op_counter`` 模块。

该模块服务于两个紧密相关的目标：

1. 统计模型侧的运行时代价，例如 FLOPs、访存、SynOps、MACs 和 ACs；
2. 在这些计数器之上构建更高层的能耗估计器。

本教程中的示例都刻意保持得较小，以便在 CPU 上也能在几秒内运行完成。

当你对自己的模型做 profiling 时，要始终使用有代表性的输入形状和有代表性的脉冲稀疏度。
``op_counter`` 是 runtime-driven 的，因此输入一变，计数结果和能耗估计也可能随之变化。

概述
++++++++++++++++++++++++

什么是 ``op_counter``？
-------------------------

``op_counter`` 是一个基于 PyTorch dispatch 和模块追踪的运行时 profiling 工具集。
它并不只从静态层形状估算计数，而是观察一次真实执行，在给定输入下记录模型实际发生了什么。

这对 SNN 尤其有用，因为很多量都依赖运行时活动：

* 二值脉冲和稠密激活不能按同一种方式理解；
* 同一层在不同输入稀疏度下可能表现不同；
* 有些能耗模型只需要前向统计，而另一些还需要反向和优化器阶段。

为什么 counter mode 很重要？
------------------------------

计数器不会修改模型，它们只会在 context manager 内生效。
这种设计让 profiling 逻辑保持显式：

* 在 context 外，模型行为完全正常；
* 在 context 内，受支持的算子会被拦截并计数；
* 多个计数器可以在同一次执行中同时工作。

核心入口是
:class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`。
它会把运行时算子调用分发给一个或多个计数器，并把结果按模块作用域和全局汇总保存下来。

基础计数工作流
++++++++++++++++++++++++

使用 ``DispatchCounterMode``
------------------------------

基本工作流如下：

1. 实例化一个或多个计数器；
2. 在 ``DispatchCounterMode`` 内执行一次真实前向或前向加反向；
3. 用 ``get_counts()`` 读取按作用域划分的计数，或用 ``get_total()`` 读取全局总数。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, op_counter

    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        neuron.IFNode(),
        nn.Linear(16, 4, bias=False),
    )
    x = (torch.rand(2, 8) > 0.5).float()

    flop_counter = op_counter.FlopCounter()
    mem_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode(
        [flop_counter, mem_counter],
        verbose=False,
        strict=True,
    ):
        _ = model(x)

    print("FLOPs:", flop_counter.get_total())
    print("Memory access (bytes):", mem_counter.get_total())
    print("Global FLOP record:", flop_counter.get_counts()["Global"])

对于第一次在小型受支持模型上上手，``strict=True`` 是更稳妥的默认值，因为它可以避免 silent under-counting。
只有当你明确希望在探索 unsupported 算子时先拿到一份 partial report，才建议使用 ``strict=False``。

虽然这个第一个例子已经使用了带 ``IFNode`` 的 SNN 风格模块，但它此处仍然只聚焦于通用的 counter workflow。
像 ``SynOps`` 这类真正依赖脉冲语义的指标，会在下文单独解释。

可用的计数器
----------------

最常用的计数器包括：

* :class:`FlopCounter <spikingjelly.activation_based.op_counter.flop.FlopCounter>`：
  统计浮点操作数，适合做 ANN 风格的计算强度分析。
* :class:`MemoryAccessCounter <spikingjelly.activation_based.op_counter.memory_access.MemoryAccessCounter>`：
  按字节统计运行时访存流量。
* :class:`SynOpCounter <spikingjelly.activation_based.op_counter.synop.SynOpCounter>`：
  统计脉冲驱动的突触加法操作。稠密浮点输入不会贡献 SynOps。
* :class:`MACCounter <spikingjelly.activation_based.op_counter.mac.MACCounter>`：
  统计乘加操作。
* :class:`ACCounter <spikingjelly.activation_based.op_counter.ac.ACCounter>`：
  统计未被建模为 MAC 的加法类算术工作。

这些计数器是互补的，而不是相互替代的。例如，某个脉冲驱动的线性层可能有非零 SynOps 和 ACs，但 MACs 为零。

``SynOpCounter`` 还需要额外提醒一点：只有当相关层真正接收到二值脉冲输入时，它才有意义。
如果同一层接收到的是稠密浮点激活，那么 SynOp 计数为 0 是完全正常的。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False)
    spike_x = (torch.rand(2, 8) > 0.5).float()

    synop_counter = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([synop_counter], strict=True):
        _ = model(spike_x)

    print("SynOps:", synop_counter.get_total())

Roofline 分析示例
-------------------

下面的示例复现了一次训练步 roofline 分析所需的基本量：FLOPs、访存和 arithmetic intensity。
如果你只关心推理 roofline，把 ``backward()`` 去掉即可。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
    )
    for p in model.parameters():
        p.requires_grad_(True)
    x = torch.rand(1, 2, 16, 16)

    flop_counter = op_counter.FlopCounter()
    mem_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([flop_counter, mem_counter], strict=True):
        y = model(x)
        y.sum().backward()

    flops = flop_counter.get_total()
    mem_bytes = mem_counter.get_total()
    intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")

    print("total FLOPs:", flops)
    print("total memory access (bytes):", mem_bytes)
    print("arithmetic intensity (FLOPs/byte):", intensity)

这个例子不会替你直接画 roofline 图。它给出了工作负载测得的点，之后你可以再结合硬件峰值 FLOPs 和带宽，把这个点放到 roofline 图上。

高层次能耗模型
++++++++++++++++++++++++

模型概览与适用边界
--------------------

``op_counter`` 当前暴露了四种高层能耗估计器：

* ``estimate_compute_energy``：仅计算 MAC/AC 的能耗；
* ``estimate_lemaire_energy``：Lemaire 风格解析式前向推理能耗；
* ``estimate_neuromc_runtime_energy``：运行时 NeuroMC 风格能耗；
* ``estimate_spikesim_event_energy``：运行时 SpikeSim 风格 Conv2d 能耗。

它们回答的不是同一个问题。各自的目标和边界如下：

.. list-table::
    :header-rows: 1

    * - 估计器
      - 主要用途
      - 覆盖范围
      - 主要边界
    * - ``estimate_compute_energy``
      - 归一化的计算能耗比较
      - 仅 MAC 和 AC 能耗
      - 不包含访存、寻址、神经元状态驻留和硬件映射
    * - ``estimate_lemaire_energy``
      - 与 Lemaire 公式对齐的前向 SNN 推理估计
      - ops、寻址、运行时尺寸的访存、神经元状态访存
      - 仅前向推理；是解析式估计，不是硬件仿真
    * - ``estimate_neuromc_runtime_energy``
      - 更完整的前向、反向、优化器运行时能耗
      - NeuroMC 风格映射下的计算和访存
      - 只有在受支持 fragment 集合内才能解释为 exact
    * - ``estimate_spikesim_event_energy``
      - SpikeSim 风格卷积加速器估计
      - 带有 SpikeSim 系数的 Conv2d stage 能耗
      - 只适用于受支持的 Conv2d 推理 stage，不是通用完整模型能耗估计器

最重要的一条是：不要把不同估计器给出的绝对值当成共享同一硬件假设的数字来直接比较。
每个估计器都有自己的成本口径和建模范围。

仅计算 MAC/AC 的能耗模型
--------------------------

``estimate_compute_energy`` 是最简单的高层估计器。
它执行一次真实前向传播，然后用一个很小的成本表，把运行时 MAC 和 AC 计数换算成能耗。

它适合用于归一化比较，例如：

* 在同一成本口径下比较两个结构；
* 在算术层面比较脉冲驱动执行和稠密执行；
* 报告 Horowitz 风格的 FP32、FP16 或 INT8 计算成本。

它的边界同样必须明确：

* 不建模访存能耗；
* 不建模寻址或路由开销；
* 不尝试复现某个具体加速器；
* ``SynOps`` 和 ``FLOPs`` 只是辅助统计，不会直接进入总能耗。

默认使用 Horowitz 2014 的 FP32 口径。如果你想做 FP16 或 INT8 比较，需要显式传入对应 preset。

Lemaire 解析式推理能耗
------------------------

``estimate_lemaire_energy`` 是一个仅用于推理的解析式估计器，与 Lemaire 风格的 SNN 能耗文献对齐。
与纯静态公式不同，当前实现仍会运行一次真实前向传播，以收集运行时计数和访存字节数。

它包含：

* 突触操作计数；
* MAC 和 AC 类工作；
* 寻址计数；
* 神经元状态读写和状态算术；
* 基于运行时字节流量与 buffer 尺寸的分段访存能耗估计。

它的边界是：

* 仅前向推理；
* 属于解析式估计，而不是 cycle-accurate 的硬件仿真；
* 访存成本来自被建模的 buffer 尺寸，而不是宿主机真实 cache 行为；
* 对某些不受支持的稀疏情况，可能会带 warning 地回退到稠密 lower bound 的访存统计。

当你需要一个比 compute-only MAC/AC 能耗更丰富的前向 SNN 推理估计，但又不需要反向和优化器建模时，它是更合适的选择。

NeuroMC 运行时能耗
--------------------

``estimate_neuromc_runtime_energy`` 是这个模块里目标最完整的估计器。
它对真实执行片段做 profiling，再把这些片段映射到 NeuroMC 风格的计算和访存公式上。

它支持几个层次的使用方式：

* 仅前向推理；
* 一次完整训练步的 ``forward -> backward -> optimizer``；
* 通过
  :class:`NeuroMCEnergyProfiler <spikingjelly.activation_based.op_counter.neuromc.core.NeuroMCEnergyProfiler>`
  做手工分阶段 profiling。

它的优势包括：

* 支持 ``forward``、``backward``、``optimizer`` 这样的 stage 级报告；
* 在受支持 fragment 上兼容 ANN、SNN 和混合执行路径；
* 能显式处理脉冲生成、BatchNorm、优化器等不同 process category。

它的边界包括：

* 只有在受支持 fragment 集合内，报告才应被解释为 exact；
* 不支持的算子不应被理解为“完整覆盖下的精确能耗”；
* 在手工 profiling 时，stage 命名本身会携带诸如时间/批次复用等语义；
* 它仍然是基于硬件模型的估计，而不是从真实芯片上测得的功耗。

如果你需要训练阶段能耗，或者需要在线学习场景下的 stage breakdown，那么应优先从它开始。

SpikeSim 事件能耗
-------------------

``estimate_spikesim_event_energy`` 面向的是一个更窄的问题：
在 SpikeSim 风格加速器模型下，受支持的 Conv2d 推理 stage 消耗了多少能耗？

默认情况下，它会保留已发布 SpikeSim 实现中的 dense PE-cycle 能耗路径，同时用运行时 profiling 自动发现真实发生的 Conv2d stage 和 shape。

它的边界很严格：

* 只适用于受支持的 Conv2d 前向推理 stage；
* 不是通用的完整模型能耗估计器；
* 默认 ``activity_mode="dense"`` 时，运行时脉冲稀疏度不会降低能耗；
* 当 ``require_if_lif_neurons=True`` 时，模型应保持在 IF/LIF 风格神经元假设之内；
* 非 Conv2d 的工作不在它的主要能耗路径中。

如果你明确关心的是 SpikeSim 风格卷积加速器比较，就用它。
否则，多数情况下应优先考虑 ``estimate_lemaire_energy`` 或 ``estimate_neuromc_runtime_energy``。

推理能耗估计示例
++++++++++++++++++++++++

Compute-Only 示例
-------------------

在使用面向推理的能耗估计器之前，先调用 ``model.eval()``。
如果你想把反向或优化器阶段也纳入进来，应切换到 ``estimate_neuromc_runtime_energy``。

下面的例子使用最简单的能耗模型估计一次前向推理能耗。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False).eval()
    x = torch.rand(2, 8)

    report = op_counter.estimate_compute_energy(model, x)

    print("total energy (pJ):", report.energy_total_pj)
    print("MAC energy (pJ):", report.energy_mac_pj)
    print("AC energy (pJ):", report.energy_ac_pj)
    print("counts:", report.counts)

如果你想切换到另一套成本口径：

.. code-block:: python

    cfg = op_counter.ComputeEnergyConfig(
        cost_config=op_counter.ComputeEnergyCostConfig.fp16()
    )
    report_fp16 = op_counter.estimate_compute_energy(model, x, config=cfg)
    print("FP16-regime energy (pJ):", report_fp16.energy_total_pj)

这个例子刻意保持简单，因为当前验收目标只要求提供推理能耗估计示例。
如果你需要更细致的前向 SNN 推理建模，可以把入口替换为 ``estimate_lemaire_energy``。

如果你希望得到一个更丰富的仅前向推理估计，并且把访存、寻址和神经元状态效应也纳入进去，
可以切换到 Lemaire 风格估计器：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    model_snn = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        neuron.IFNode(),
        nn.Linear(16, 4, bias=False),
    ).eval()
    spike_x = (torch.rand(2, 8) > 0.5).float()

    lemaire_report = op_counter.estimate_lemaire_energy(model_snn, spike_x)
    print("Lemaire total (pJ):", lemaire_report.total_pj)
    print("Lemaire breakdown:", lemaire_report.breakdown_pj)

实践建议
+++++++++++++++++

如何选择计数器或能耗模型
--------------------------

让工具去匹配你的问题：

* 如果你需要 FLOPs、访存或 SynOps，直接使用基础计数器；
* 如果你需要 roofline 输入，组合 ``FlopCounter`` 和 ``MemoryAccessCounter``；
* 如果你需要一个简单、归一化的算术能耗比较，使用 ``estimate_compute_energy``；
* 如果你需要包含访存和神经元状态效应的前向 SNN 推理能耗，使用 ``estimate_lemaire_energy``；
* 如果你需要训练阶段 breakdown 或优化器能耗，使用 ``estimate_neuromc_runtime_energy``；
* 如果你需要 SpikeSim 风格的 Conv2d 加速器能耗，使用 ``estimate_spikesim_event_energy``。

在汇报结果时，始终要说明：

* 你使用了哪个估计器；
* 这次运行是否只包含前向，还是还包含反向/优化器；
* 采用了什么成本口径或硬件假设；
* 输入类型和稀疏条件是什么。

总结
++++++++++++++++++++++++

``op_counter`` 不只是一个单独的计数器，而是一套从底层运行时计数走向高层能耗估计的 profiling 框架。

对大多数工作流来说，一个实用的推进顺序是：

1. 先用直接计数器理解运行时行为；
2. 再用 FLOP 和访存计数做 roofline 风格分析；
3. 然后选择与目标问题匹配的能耗估计器；
4. 最后在各自的建模边界内解释结果，而不是把它们当成普适真值。
