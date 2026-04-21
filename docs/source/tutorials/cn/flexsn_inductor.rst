FlexSN Inductor 后端
====================

本页作者：`wei.fang <https://github.com/fangwei123456>`_

English version: :doc:`../en/flexsn_inductor`

``FlexSN`` 自 ``0.0.0.1.0`` 版本起新增 ``backend="inductor"`` 选项，将自定义的单步动力学函数 ``core``
经由 `PyTorch Inductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir/747>`_
编译成 GPU 内核。与 ``backend="triton"``（自研 FX→Triton 映射）相比，Inductor 路径：

* 不需要手动维护算子映射表，``core`` 中出现的绝大多数 PyTorch 算子均可直接编译；
* 依赖 ``torch.compile``，可与外层网络联合编译，实现跨层算子融合；
* 无需设置 ``PYTORCH_JIT=0``。

.. admonition:: 注意
   :class: warning

   * ``backend="inductor"`` 必须配合 ``torch.compile(model, fullgraph=True)`` 使用，单独构造 ``FlexSN`` 不会触发编译。
   * 目前仅支持 CUDA 设备。

快速上手
--------

以 LIF 神经元为例，定义一个 ``core`` 函数并使用 ``backend="inductor"``：

.. code:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        """单步 LIF 动力学：charge → fire → reset。"""
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau        # 充电
        s = (h >= v_th).to(h.dtype)  # 放电
        return s, h * (1.0 - s)      # 输出脉冲 + 更新膜电压

    neuron = FlexSN(
        core=lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    # 必须套 torch.compile；不加则退化为 eager HOP，无编译收益
    model = nn.Sequential(nn.Linear(512, 512), neuron, nn.Linear(512, 512))
    model = torch.compile(model, fullgraph=True)
    model = model.cuda()

    x = torch.randn(8, 64, 512, device="cuda")  # [T, B, N]
    out = model(x)

与 ``backend="triton"`` 数值对齐
---------------------------------

两种后端在 ``atol=1e-5`` 精度下输出完全一致（max abs diff = 0.0）：

.. code:: python

    import os, torch
    os.environ["PYTORCH_JIT"] = "0"
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x, v):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    torch.manual_seed(0)
    x = torch.randn(8, 32, 1024, device="cuda")

    n_tri = FlexSN(lif_core, 1, 1, 1, step_mode="m", backend="triton").cuda()
    n_ind = FlexSN(lif_core, 1, 1, 1, step_mode="m", backend="inductor").cuda()
    c_ind = torch.compile(n_ind, fullgraph=True)

    torch.testing.assert_close(c_ind(x), n_tri(x), atol=1e-5, rtol=1e-5)
    print("数值一致 ✓")

与其它后端对比
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - 属性
     - ``backend="torch"``
     - ``backend="triton"``
     - ``backend="inductor"``
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
     - 是（推荐）
     -
   * - 跨层融合
     - 否
     - 否
     - 是
     - G3 目标
   * - T=8 性能（vs triton）
     - 慢
     - 基准
     - 0.52× **更快**
     - RTX 4090 实测
   * - T=32 性能（vs triton）
     - 慢
     - 基准
     - 0.28× **更快**
     - RTX 4090 实测

性能说明
--------

``backend="inductor"`` 在初始化时通过 ``make_fx`` 追踪 ``core`` 函数，使用现有的 FlexSN 模板基础设施
生成一个带 ``tl.static_range(T)`` 时间循环的单个 Triton 扫描内核。无论 T 取何值，每次推理调用
只触发一次内核启动，GPU 利用率与 ``backend="triton"`` 相当甚至更优。

使用建议
--------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 场景
     - 推荐后端
     - 原因
   * - 快速原型 / 任意设备
     - ``"torch"``
     - 无约束
   * - CUDA 单层，追求极致性能
     - ``"triton"`` 或 ``"inductor"``
     - 两者性能相当
   * - 需要跨层融合 / 任意 T
     - ``"inductor"``
     - 单核扫描 + torch.compile 融合
   * - 需要训练（backward）
     - ``"inductor"``
     - 通过 eager_scan 自动支持 BPTT
