算子计数与能耗估计
=========================================

本教程作者： `黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/op_counter`

本教程介绍 ``spikingjelly.activation_based.op_counter``。它统计一次真实执行中的
FLOPs、访存、SynOps、MACs 和 ACs，并基于这些计数估算能耗。结果取决于输入形状、
脉冲稀疏度和 ``train``/``eval`` 模式，因此应使用能代表目标场景的配置。

概述
++++++++++++++++++++++++

运行时统计方式
--------------

``op_counter`` 通过三种上下文管理器观察运行时调用：

* ``DispatchCounterMode`` 拦截 ATen 算子；
* ``FunctionCounterMode`` 拦截 ``torch.*`` 函数；
* ``ModuleCounterMode`` 记录实际执行的 ``nn.Module`` 前向和反向事件。

它们只在上下文内生效，不修改模型。多个计数器可以统计同一次执行。与静态形状
分析相比，这种方式能区分二值脉冲和稠密激活，也能反映输入稀疏度及执行阶段的
变化。

基础计数工作流
++++++++++++++++++++++++

使用 ``DispatchCounterMode``
------------------------------

1. 实例化一个或多个计数器；
2. 在 ``DispatchCounterMode`` 内执行一次真实前向或前向加反向；
3. 用 ``get_counts()`` 读取按作用域划分的计数，或用 ``get_total()`` 读取全局总数。

``train()`` 和 ``eval()`` 都可以统计。若模型包含 dropout 或 batch normalization，
应选择与目标场景一致的模式。

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
        strict=False,
    ):
        _ = model(x)

    print("FLOPs:", flop_counter.get_total())
    print("Memory access (bytes):", mem_counter.get_total())
    print("Global FLOP record:", flop_counter.get_counts()["Global"])

包级 logger 会以 ``DEBUG`` 级别输出逐算子记录。该日志在大型模型上开销较高，
只应在排查计数问题时启用：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

上述算子分发示例使用 ``strict=False``，未支持的辅助算子会被跳过。确认计数器覆盖目标
路径后，可改用 ``strict=True`` 让未支持的算子立即报错。

使用 ``ModuleCounterMode``
----------------------------

模块计数器的规则键为 ``("forward" | "backward", module_type)``。
``ModuleCounterMode`` 管理钩子、作用域和异常清理，但不会自动重置计数器。作用域
以 ``Global`` 开始，随后是根模块类型和完整的子模块路径。

.. code-block:: python

    memory_counter = op_counter.NeuromorphicMemoryAccessCounter()
    with op_counter.ModuleCounterMode(
        [memory_counter], model=model, strict=True
    ):
        _ = model(x)

    print(memory_counter.get_counts()["Global"])

三种模式都可在上下文结束后调用 ``mode.get_unsupported(counter)``，查看非严格
模式跳过的目标。

可用的计数器
----------------

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

这些计数器统计不同含义的工作量。例如，脉冲驱动的线性层可能产生 SynOps 和
ACs，但不产生 MACs。``SynOpCounter`` 只把二值脉冲输入计为 SynOps；稠密浮点输入
得到 0。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False)
    spike_x = (torch.rand(2, 8) > 0.5).float()

    synop_counter = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([synop_counter], strict=False):
        _ = model(spike_x)

    print("SynOps:", synop_counter.get_total())

Roofline 分析示例
-------------------

下面的示例统计一个训练步的 FLOPs、访存和算术强度。推理分析可去掉
``backward()``。

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
        nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
    )
    x = torch.rand(1, 2, 16, 16)

    flop_counter = op_counter.FlopCounter()
    mem_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([flop_counter, mem_counter], strict=False):
        y = model(x)
        y.sum().backward()

    flops = flop_counter.get_total()
    mem_bytes = mem_counter.get_total()
    intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")

    print("total FLOPs:", flops)
    print("total memory access (bytes):", mem_bytes)
    print("arithmetic intensity (FLOPs/byte):", intensity)

该结果是工作负载在 roofline 图上的测量点，还需配合硬件峰值 FLOPs 和带宽。
计数采用理想化口径：每个 MAC 计 2 FLOPs，逻辑输入读取一次，逻辑输出写入一次；
不建模 tiling、cache、fusion、bank conflict 或真实 DRAM 流量。

高层能耗模型
++++++++++++++++++++++++

模型概览
--------

``op_counter`` 目前提供四种高层能耗估计器：

* ``estimate_simple_energy``：运行时 MAC/AC/访存的简单能耗；
* ``estimate_lemaire_energy``：Lemaire 风格解析式前向推理能耗；
* ``estimate_neuromc_runtime_energy``：运行时 NeuroMC 风格能耗；
* ``estimate_spikesim_energy``：运行时 SpikeSim 风格 Conv2d 能耗。

.. list-table::
    :header-rows: 1

    * - 估计器
      - 主要用途
      - 覆盖范围
      - 主要边界
    * - ``estimate_simple_energy``
      - 归一化的运行时能耗比较
      - MAC、AC、权重/bias 读取和持久神经元状态读写
      - 不包含信号流、FIFO、路由、寻址和硬件映射
    * - ``estimate_lemaire_energy``
      - 与 Lemaire 公式对齐的前向 SNN 推理估计
      - ops、寻址、运行时尺寸的访存、神经元状态访存
      - 仅前向推理；是解析式估计，不是硬件仿真
    * - ``estimate_neuromc_runtime_energy``
      - 前向、反向和优化器阶段的运行时能耗
      - NeuroMC 风格映射下的计算和访存
      - 仅覆盖受支持的执行片段和阶段语义
    * - ``estimate_spikesim_energy``
      - SpikeSim 风格卷积加速器估计
      - 带有 SpikeSim 系数的 Conv2d 阶段能耗
      - 只适用于受支持的 Conv2d 推理阶段，不是通用完整模型能耗估计器

四个估计器使用不同的成本口径和硬件假设，绝对值不能交叉比较。

每份报告的 ``model_info`` 给出稳定模型 ID、来源、工艺节点、精度、适用范围和
fidelity；``config``（NeuroMC 为 ``memory_config``）保存实际成本配置。
``paper``/``reference-code`` 表示论文或作者脚本复刻，``source-aligned`` 表示采用
作者常量和表格但保留本项目的运行时映射，``spikingjelly-defined`` 表示公式由
SpikingJelly 明确定义。汇报结果时，应注明估计器、执行阶段、成本配置、输入类型
和稀疏度。

简单运行时能耗模型
--------------------

``estimate_simple_energy`` 执行一次前向传播，并按下式换算运行时计数：
``MAC * E_MAC + AC * E_AC + bytes * E_memory``。

主要假设如下：

* ``NeuromorphicMemoryAccessCounter`` 独立统计实际使用的权重和偏置，
  以及持久神经元状态每时间步的一次读取和一次写回；
* 输入电流和输出脉冲被视为片上信号流，不计为访存；
* 不建模 FIFO、寻址、路由、缓存复用或具体硬件映射；
* ``SynOps`` 是 AC 的辅助子集，不会重复收费；
* 默认使用 Horowitz 2014 FP32 算术成本和 STEP Table 9 的 ``24.96 pJ/byte``
  访存单价；访存流量公式由 SpikingJelly 定义；
* FP16/INT8 预设只更换算术单价，不量化模型。

Lemaire 解析式推理能耗
------------------------

``estimate_lemaire_energy`` 执行一次前向传播，再把突触操作、MAC/AC、寻址、
神经元状态和逐层 SRAM 访问代入 Lemaire 解析式。主要限制如下：

* 仅前向推理；
* 是解析式估计，不是周期精确的硬件仿真；
* 运算和访存固定使用论文的 32-bit 口径，不随宿主张量数据类型改变；
* 参数、FIFO 和膜电位访问先按各层本地 SRAM 容量计价，再汇总能耗；
* 二元输入使用 SNN 事件公式，稀疏但非二元的输入仍使用 FNN 稠密公式；
* 分组卷积和深度卷积按每组输出通道计算脉冲扇出；
* 神经元只支持论文范围内的 IF/LIF，其他 ``BaseNode`` 默认直接拒绝；
* SNN FIFO 默认容纳 1000 个消息；可通过 ``snn_fifo_capacity_elements`` 覆盖；
* 默认 ``strict=True``，不支持的转置卷积直接报错；显式设为 ``False`` 时才警告并跳过。

NeuroMC 运行时能耗
--------------------

``estimate_neuromc_runtime_energy`` 分析实际执行片段，再按固定的 NeuroMC v1 常量、
逐变量访存方向和倍率估算能耗。它保留作者的成本口径，但不复刻完整的 ZigZag 映射。
便捷入口始终执行前向；提供 ``target`` 和 ``loss_fn``
时继续执行反向，另提供 ``optimizer`` 时估算优化器阶段。手工分阶段统计可使用
:class:`NeuroMCEnergyProfiler <spikingjelly.activation_based.op_counter.neuromc.core.NeuroMCEnergyProfiler>`。

主要限制如下：

* 不支持的能耗相关算子会拒绝生成总量；
* 手工分析使用 ``stage(name, phase=..., reuse_weights=...,
  batch_norm_backward=...)`` 显式传递映射语义，名称本身不再编码协议；同一
  上下文内复用阶段名称时必须使用相同选项；
* 便捷入口会清空已有梯度，但不会调用 ``optimizer.step()`` 或修改模型参数；
* 同一个模块被重复调用时，这些调用必须全部参与反向传播；只对部分调用
  反传会因映射歧义而拒绝报告；
* 结果来自硬件模型，不是真实芯片功耗测量。

SpikeSim 运行时能耗
-------------------

``estimate_spikesim_energy`` 统计实际执行的 Conv2d 推理阶段。默认 ``dense`` 模式
使用作者代码的 PE-cycle 公式；``event`` 模式使用 SpikingJelly 定义的稀疏公式。
主要限制如下：

* 模型应处于 ``eval`` 模式；默认 ``strict=True``，未支持的 Conv2d 阶段或空报告
  直接报错；
* 只统计受支持的 Conv2d 前向阶段，非 Conv2d 工作不进入主要能耗路径；
* 默认 ``activity_mode="dense"`` 时，运行时脉冲稀疏度不会降低能耗；
* ``activity_mode="event"`` 对应 ``spikingjelly_spikesim_event_v1``，
  使用 SpikeSim 常量和本项目定义的 A/R/Z 稀疏公式，不冒充作者 dense 模型；
* ``require_if_lif_neurons=True`` 时只接受 IF/LIF 风格神经元。

能耗估计示例
++++++++++++++++++++++++

Simple Energy 示例
-------------------

面向推理的估计器应在 ``model.eval()`` 后运行。下面先使用 Simple Energy：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False).eval()
    x = torch.rand(2, 8)

    report = op_counter.estimate_simple_energy(model, x)

    print("total energy (pJ):", report.energy_total_pj)
    print("compute energy (pJ):", report.energy_compute_pj)
    print("MAC energy (pJ):", report.energy_mac_pj)
    print("AC energy (pJ):", report.energy_ac_pj)
    print("memory energy (pJ):", report.energy_memory_pj)
    print("counts:", report.counts)

可显式切换成本口径：

.. code-block:: python

    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig.fp16()
    )
    report_fp16 = op_counter.estimate_simple_energy(model, x, config=cfg)
    print("FP16-regime energy (pJ):", report_fp16.energy_total_pj)

Lemaire 估计器还会计入寻址和神经元状态：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, op_counter

    model_snn = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        neuron.IFNode(),
        nn.Linear(16, 4, bias=False),
    ).eval()
    spike_x = (torch.rand(2, 8) > 0.5).float()

    lemaire_report = op_counter.estimate_lemaire_energy(model_snn, spike_x)
    print("Lemaire total (pJ):", lemaire_report.total_pj)
    print("Lemaire breakdown:", lemaire_report.breakdown_pj)

验证与来源
+++++++++++++++++

模型来源
--------

* Simple Energy 算术单价：`Horowitz 2014 <https://doi.org/10.1109/ISSCC.2014.6757323>`_；
  访存单价：`STEP Table 9 <https://openreview.net/pdf?id=SzwU2XrXIS>`_。
* Lemaire：`An Analytical Estimation of Spiking Neural Networks Energy Efficiency
  <https://arxiv.org/abs/2210.13107>`_。
* SpikeSim dense：作者代码 commit
  `c2627bc <https://github.com/Intelligent-Computing-Lab-Yale/SpikeSim/commit/c2627bc091a47bdcb630ca6207eaf44a00bd1da4>`_。
* NeuroMC：作者代码 commit
  `712c66f <https://github.com/dayanhn/NeuroMC/commit/712c66f47cf76ae530a55f8bcad3858bd68788de>`_。

相对趋势检查
------------

该基准只回答一个问题：在选定案例中，SpikingJelly 是否保留来源
模型给出的相对趋势。每个案例记录一对 ``(E_origin, E_SJ)``。SpikeSim 和 NeuroMC
使用固定版本的作者代码；Lemaire 没有公开代码，因此参考值按论文方程 (1)--(20)
计算。参考路径只接收静态拓扑、张量尺寸和独立观测的发放数，不读取
SpikingJelly 报告。

主指标是 Kendall tau-b，并用 2,000 次成对 bootstrap 给出 95% 重采样区间。Spearman rho
和 log-Pearson ``r`` 用作辅助观察。P90 对称倍率先移除中位乘法尺度，再衡量相对
误差。``tau-b >= 0.80`` 和 ``P90 <= 1.50x`` 是预先设定的比较参考线，不是准确性
判定标准。

* **Kendall tau-b** 比较案例两两之间的高低顺序。``1`` 表示顺序完全一致，``0``
  表示没有稳定的排序关系，``-1`` 表示顺序完全相反。
* **Spearman rho** 计算两组名次的相关性。它也位于 ``[-1, 1]``，并且比 tau-b 更
  关注单个案例的名次移动幅度。
* **P90 对称倍率** 是移除固定尺度差后，相对误差倍率的经验 90 分位数。
  ``1.0x`` 最理想；``1.5x`` 表示该分位点对应参考相对值的 ``1 / 1.5`` 到
  ``1.5`` 倍。

.. list-table:: 验证结果
   :header-rows: 1
   :widths: 20 12 23 14 14 14 14

   * - 估计器模式
     - 可比案例数
     - Kendall tau-b（95% bootstrap 区间）
     - Spearman rho
     - Log-Pearson r
     - P90 倍率
     - 中位尺度 E_SJ/E_origin
   * - Lemaire
     - 12
     - 0.939 [0.729, 1.000]
     - 0.979
     - 0.998
     - 1.478x
     - 0.877x
   * - SpikeSim dense
     - 7（另有 5 个压力案例）
     - 1.000 [1.000, 1.000]
     - 1.000
     - 1.000
     - 1.000x
     - 1.000x
   * - NeuroMC
     - 13
     - 0.795 [0.541, 0.971]
     - 0.934
     - 0.981
     - 1.189x
     - 0.396x

.. figure:: ../../_static/tutorials/op_counter/energy_model_validation.png
   :alt: 归一化参考值与 SpikingJelly 评分，以及各模型的 tau-b 与 P90 减一
   :align: center

   左图比较归一化后的成对评分。靠近虚线只表示相对趋势接近，不能说明绝对能耗
   准确。右图汇总各模型的 tau-b 和 ``P90 - 1``。

Lemaire 的 tau-b 和 P90 都达到参考线。NeuroMC 的 P90 达标，tau-b 为 ``0.795``，
略低于 ``0.80``。SpikeSim 的 7 个可比案例完全一致，因为 ``dense`` 模式直接实现
作者公式；该结果主要检查集成和计算过程。另有 5 个动态压力案例不参与相关性
计算，其动态/静态比值为 ``0.500x`` 到 ``3.000x``。这里不作统一的“通过”判断。

**局限性：**

* 每组只有 7、12 或 13 个选定案例，不能代表更广泛的网络和发放模式；bootstrap
  区间只反映这些案例内的重采样稳定性。
* 两条路径共享拓扑、尺寸和发放数，网络规模本身可能产生高相关性。高 tau 或 rho
  不能逐项验证能耗项或系数。
* 参考值来自其他分析模型，不是硬件测量。相关性只能说明与这些模型的趋势接近，
  不能证明物理能耗准确。
* 排名和 P90 都弱化或移除了绝对尺度；即使存在固定的绝对偏差，也可能得到较好的
  指标。
* 覆盖仍不完整：Lemaire 参考值由论文方程重建，本基准只验证 NeuroMC 前向能耗，
  Simple Energy 和 SpikeSim event 未使用独立的端到端外部估计器。

手动运行基准脚本：

.. code-block:: bash

    uv run python benchmark/energy_model_validation.py \
        --spikesim-root /path/to/SpikeSim \
        --neuromc-root /path/to/NeuroMC

精确的案例输入、双边评分、指标、仓库版本、依赖版本和参考版本均记录在
:download:`案例级 CSV <../../_static/tutorials/op_counter/energy_model_validation.csv>` 中。
该脚本依赖固定版本的外部仓库，因此不在 CI 中运行。

总结
++++++++++++++++++++++++

``op_counter`` 按给定输入记录实际执行。基础计数器适合分析操作量和流量；能耗
估计器用于各自模型范围内的相对比较。不同估计器的绝对值不能直接比较。
