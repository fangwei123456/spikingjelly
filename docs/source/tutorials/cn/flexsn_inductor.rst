FlexSN Inductor 后端
====================

本页作者：`wei.fang <https://github.com/fangwei123456>`_

English version: :doc:`../en/flexsn_inductor`

``FlexSN`` 自 ``0.0.0.1.0`` 版本起新增 ``backend="inductor"`` 选项，将自定义的单步动力学函数 ``core``
编译成高效的 Triton GPU 内核。与 ``backend="triton"``（自研 FX→Triton 映射）相比，Inductor 路径：

* 不需要手动维护算子映射表，``core`` 中出现的绝大多数 PyTorch 算子均可直接编译；
* 无需设置 ``PYTORCH_JIT=0``；
* 支持 ``torch.compile``，可与外层网络联合编译，实现跨层算子融合；
* 推理和训练均内置专用 Triton 核，性能优于或持平 ``backend="triton"``。

.. admonition:: 注意
   :class: warning

   * 目前仅支持 CUDA 设备。
   * ``core`` 中的算子需在 ``FX_TO_TRITON`` 映射表内，不在表内的算子自动回退 ``eager_scan``（日志中会有提示）。
     支持的算子见下方 :ref:`算子覆盖 <flexsn-inductor-op-coverage>` 一节。
   * 训练时 ``core`` 应使用 surrogate gradient（如 :class:`spikingjelly.activation_based.surrogate.Sigmoid`）
     而非硬阈值，否则梯度为零。

快速上手 — 推理
---------------

.. code:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    neuron = FlexSN(core=lif_core, num_inputs=1, num_states=1,
                    num_outputs=1, step_mode="m", backend="inductor").cuda()

    x = torch.randn(8, 64, 512, device="cuda")
    with torch.no_grad():
        out = neuron(x)   # 直接调用，无需 torch.compile

    # 可选：套 torch.compile 实现与外层 Linear 的跨层融合
    model = nn.Sequential(nn.Linear(512, 512), neuron, nn.Linear(512, 512)).cuda()
    model = torch.compile(model, fullgraph=True)
    out = model(x)

快速上手 — 训练
---------------

训练时使用 surrogate gradient 使脉冲信号可微：

.. code:: python

    import torch
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    sg = surrogate.Sigmoid(alpha=4.0)

    def lif_core_sg(x: torch.Tensor, v: torch.Tensor):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = sg(h - v_th)        # Sigmoid surrogate gradient
        return s, h * (1.0 - s)

    neuron = FlexSN(core=lif_core_sg, num_inputs=1, num_states=1,
                    num_outputs=1, step_mode="m", backend="inductor").cuda()

    x = torch.randn(8, 64, 512, device="cuda", requires_grad=True)
    out = neuron(x)
    out.sum().backward()        # BPTT via Triton fwd+bwd 核
    print(x.grad.shape)         # [8, 64, 512]

    # 也可套 torch.compile：Dynamo 追踪时间循环，Inductor 编译
    model = torch.compile(neuron, fullgraph=True)
    out = model(x)
    out.sum().backward()

内核分发策略
------------

``backend="inductor"`` 的 ``multi_step_forward`` 根据上下文自动选择路径：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 条件
     - 路径
   * - 推理（no grad）+ CUDA
     - Triton 单核 scan（``tl.static_range(T)``，1 次 launch）
   * - 训练 + CUDA + 在 ``torch.compile`` **外**
     - Triton FlexSNFunction（专用 fwd+bwd 核，BPTT）
   * - 训练 + CUDA + 在 ``torch.compile`` **内**
     - ``eager_scan``（Dynamo 追踪 for 循环，Inductor 编译，标准 autograd backward）
   * - CPU 或 kernel 不可用
     - ``eager_scan`` / ``flex_sn_scan`` HOP fallback

与其它后端对比
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 22 20

   * - 属性
     - ``"torch"``
     - ``"triton"``
     - ``"inductor"``
     - 备注
   * - 可用设备
     - CPU / CUDA
     - CUDA
     - CUDA
     -
   * - 需要 PYTORCH_JIT=0
     - 否
     - 是
     - 否
     -
   * - 需要 torch.compile
     - 否
     - 否
     - 否（可选）
     - compile 可获跨层融合
   * - 推理性能（vs triton）
     - 慢
     - 基准
     - ≤ 0.52× **更快**
     - RTX 4090 实测
   * - 训练性能（vs triton）
     - 慢
     - 基准
     - ≤ 0.69× **更快**
     - VGG16-BN，CIFAR-10
   * - 跨层融合
     - 否
     - 否
     - 是（需 torch.compile）
     -
   * - 支持 float16
     - 是
     - 是
     - 是
     -

性能说明
--------

**推理**：初始化时通过 ``make_fx`` 追踪 ``core``，使用 FlexSN 模板生成带
``tl.static_range(T)`` 时间循环的单个 Triton 扫描内核，每次推理只触发一次 kernel launch。

**训练**：初始化时通过 ``aot_function`` 同时追踪正向和反向计算图（无需 ``PYTORCH_JIT=0``），
生成专用的 Triton 正向核（保存中间值）和反向核（时间逆序扫描）。
对于非可微输出（如硬阈值脉冲），自动生成 shim wrapper 适配 AOT backward 的调用签名。

在 SpikingVGG-16-BN（T=4, B=64, CIFAR-10）训练基准中，FlexSN inductor
比 LIFNode triton 快约 12%，比纯 PyTorch 快约 45%。

使用建议
--------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 场景
     - 推荐后端
     - 原因
   * - 快速原型 / CPU
     - ``"torch"``
     - 无约束
   * - CUDA 推理，追求极致性能
     - ``"inductor"``
     - 单核 scan，无需 PYTORCH_JIT=0
   * - CUDA 训练，无 torch.compile
     - ``"inductor"``
     - Triton fwd+bwd 核，快于 triton
   * - CUDA 训练 + torch.compile
     - ``"inductor"``
     - 兼容；compile 下走 eager_scan + Inductor 编译
   * - 跨层融合
     - ``"inductor"`` + ``torch.compile``
     - 外层 Linear/Conv 可与 FlexSN 联合编译

.. _flexsn-inductor-op-coverage:

算子覆盖
--------

``FX_TO_TRITON`` 映射表目前覆盖以下 ATen 算子（推理和训练路径均支持）：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 类别
     - 算子
   * - 四则运算
     - ``add``、``sub``、``mul``、``div``、``reciprocal``、``neg``、``rsub``
   * - 超越函数
     - ``exp``、``log``、``log2``、``sqrt``、``rsqrt``、``tanh``、``sin``、``cos``、``erf``
   * - 取整
     - ``floor``、``ceil``、``round``
   * - 激活 / 阈值
     - ``relu``、``sigmoid``、``sign``/``sgn``、``abs``
   * - 比较
     - ``eq``、``ne``、``ge``、``le``、``gt``、``lt``
   * - 逻辑位运算
     - ``logical_and``/``or``/``not``、``bitwise_and``/``or``/``not``
   * - 二元数学
     - ``minimum``、``maximum``、``pow``、``fmod``
   * - clamp
     - ``clamp``、``clamp_min``、``clamp_max``
   * - 类型 / 构造
     - ``_to_copy``（类型转换）、``scalar_tensor``、``zeros_like``、``ones_like``
   * - 条件选择
     - ``where``、``masked_fill``
   * - 反向专用
     - ``sigmoid_backward``、``tanh_backward``、``threshold_backward``
   * - 杂项
     - ``clone``、``detach``、``spike_fn``

不在表内的算子（如矩阵运算、复杂控制流等）会触发 ``eager_scan`` fallback 并输出 WARNING 日志。
