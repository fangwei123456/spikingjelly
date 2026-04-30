FlexSN
====================

本页作者：`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_、`wei.fang <https://github.com/fangwei123456>`_

English version: :doc:`../en/flexsn`

本教程聚焦 ``FlexSN`` 的使用。若你尚未阅读 Triton 后端基础，建议先阅读 :doc:`./triton_backend` ，了解预定义 Triton 神经元内核的启用方式与基本约束。 ``FlexSN`` 可根据用户自定义的单步神经元动力学函数 ``core`` 生成高性能多步内核。
从 ``0.0.0.1.0`` 版本起， ``FlexSN`` 支持 ``backend="triton"`` 与 ``backend="inductor"`` 两条 CUDA 加速路径。

使用 FlexSN 自定义 Triton 神经元内核
------------------------------------

用函数描述神经元动力学
^^^^^^^^^^^^^^^^^^^^^^

绝大多数脉冲神经元模型在一个离散时间步上的动力学可以描述为：

.. math::

    Y_1[t], Y_2[t], \dots, V_1[t], V_2[t], \dots = f_s\left( X_1[t], X_2[t], \dots, V_1[t-1], V_2[t-1], \dots \right)

其中 :math:`Y_i` 表示输出，:math:`V_i` 表示状态变量，:math:`X_i` 表示输入。该公式可以用 PyTorch 函数描述：

.. code:: python

    def single_step_inference(x1, x2, ..., v1, v2, ...):
        ...
        return y1, y2, ..., v1_updated, v2_updated, ...

其中 ``x1, x2, ...`` 表示输入， ``v1, v2, ...`` 表示状态变量， ``y1, y2, ...`` 表示输出，而 ``v1_updated, v2_updated, ...`` 表示更新后的状态变量（与 ``v1, v2, ...`` 对应）。例如，输入不衰减的软重置 LIF 神经元（ ``tau=2`` ， ``v_th=1.0`` ，sigmoid 替代函数 ）可以描述为：

.. code:: python

    from spikingjelly.activation_based import surrogate

    tau = 2.0 # time constant
    v_th = 1.0 # threshold
    spike_fn = surrogate.Sigmoid()

    def lif_single_step_inference(x, v):
        h = (1 - 1/tau) * v + x
        s = spike_fn(h - v_th)
        v = h - s * v_th
        return s, v

这个例子中，输入、输出和状态变量的数量都是1；而对于更复杂的神经元模型，输入、输出和状态变量的数量都可能是多个。另外，此处的模型超参数 ``tau`` ， ``v_th`` 和 ``spike_fn`` 都是固定下来的全局变量。为了灵活配置超参数，可以使用 **函数闭包** ：

.. code:: python

    from spikingjelly.activation_based import surrogate

    def lif_single_step_inference_closure(tau=2., v_th=1., spike_fn=surrogate.Sigmoid()):
        def lif_single_step_inference(x, v):
            h = (1 - 1/tau) * v + x
            s = spike_fn(h - v_th)
            v = h - s * v_th
            return s, v
        return lif_single_step_inference

    f = lif_single_step_inference_closure(tau=99., v_th=0.5)

FlexSN 使用流程
^^^^^^^^^^^^^^

以如下的自定义脉冲神经元为例：

.. code:: python

    import torch
    from spikingjelly.activation_based import surrogate

    def complicated_lif_core_generator(beta: float, gamma: float, spike_fn=surrogate.ATan()):
        def complicated_lif_core(
            x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, rho: torch.Tensor
        ):
            h = beta*v + x
            s1 = spike_fn(h - (rho+1.)) # spike, with threshold adaptation
            s2 = spike_fn(h - 1.)       # spike, without threshold adaptation
            rho = gamma*rho + s1        # adaptation variable update
            v1 = h * (1.-s1)            # hard reset
            v2 = h - s2                 # soft reset
            yy = torch.sigmoid(y)       # modulation factor
            v = v1*yy + v2 * (1.-yy)    # modulated reset
            return s1, s2, v, rho
        return complicated_lif_core

该模型有两个输入 ``x, y`` ，两个输出 ``s1, s2`` ，以及两个状态变量 ``v, rho`` 。不同变量之间的依赖关系如下图所示：

.. image:: ../../_static/tutorials/flexsn/neuron.png
    :width: 100%

为了生成多步 Triton 内核，使用 :class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>` 模块进行包装：

.. code:: python

    from spikingjelly.activation_based import neuron

    f = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="triton",
        store_state_seqs=True,
    )

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    s1, s2 = f(x, y)
    v, rho = f.state_seqs

    print(s1.mean()) # tensor(0.0821, device='cuda:0', grad_fn=<MeanBackward0>)
    print(s2.mean()) # tensor(0.1494, device='cuda:0', grad_fn=<MeanBackward0>)
    print(v.mean()) # tensor(-0.2750, device='cuda:0', grad_fn=<MeanBackward0>)
    print(rho.mean()) # tensor(0.4842, device='cuda:0', grad_fn=<MeanBackward0>)

:class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>` 的构造需要以下关键参数：

* ``core`` ：描述单步神经元动力学的函数，签名为 ``[*inputs, *states] -> [*outputs, *states]`` 。
* ``num_inputs, num_states, num_outputs`` ：输入、状态变量和输出的个数。应与 ``core`` 签名的情况相一致。
* ``example_inputs`` ： ``core`` 的参数示例。 ``FlexSN`` 内部将使用这些示例输入调用 ``core`` ，从而捕获计算图。
* ``requires_grad`` ： ``core`` 参数是否需要求梯度。默认值为 ``None`` ，含义为“所有参数都需要梯度”（即等价于全为 ``True`` ）。
* ``step_mode, backend`` ：类似于其他神经元模块，这两个参数决定了步进模式和后端。 ``triton`` 后端只在 ``step_mode="m"`` 时有效。
* ``store_state_seqs`` ：类似于其他神经元的 ``store_v_seq`` ，该参数决定是否保存状态序列。若为 ``True`` ，则可通过 ``state_seqs`` 属性获取上一次运行的状态序列：该属性是一个列表，列表的每个元素对应着某个状态的序列。 ``FlexSN`` 当然也支持反向传播，如下面的代码片段所示：

.. code:: python

    n_triton = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="triton",
        store_state_seqs=True,
    )

    n_torch = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
    )

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    x_triton = x.clone().requires_grad_(True)
    y_triton = y.clone().requires_grad_(True)
    x_torch = x.clone().requires_grad_(True)
    y_torch = y.clone().requires_grad_(True)

    s1_triton, s2_triton = n_triton(x_triton, y_triton)
    s1_torch, s2_torch = n_torch(x_torch, y_torch)
    grad = torch.randn_like(s1_triton)
    s1_triton.backward(grad)
    s1_torch.backward(grad)

    v_triton, rho_triton = n_triton.state_seqs
    v_torch, rho_torch = n_torch.state_seqs

    assert torch.allclose(s1_triton, s1_torch)
    assert torch.allclose(s2_triton, s2_torch)
    assert torch.allclose(x_triton.grad, x_torch.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y_triton.grad, y_torch.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_triton, v_torch, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rho_triton, rho_torch)
    print(s1_triton.mean())
    print(s2_triton.mean())
    print(x_triton.grad.mean())
    print(y_triton.grad.mean())
    print(v_triton.mean())
    print(rho_triton.mean())

``assert`` 全部通过，输出如下所示。这证明： ``FlexSN`` 生成的 Triton 内核与原始 PyTorch 函数在前向和反向传播时都具有等价性。

.. code:: text

    tensor(0.0821, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.1494, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.0007, device='cuda:0')
    tensor(6.2995e-05, device='cuda:0')
    tensor(-0.2750, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.4842, device='cuda:0', grad_fn=<MeanBackward0>)

使用上述流程，用户可以用极少的代码量得到 Triton 加速的神经元模型。相比曾经的 ``auto_cuda`` 模块（见 :doc:`./cupy_neuron` ）， ``FlexSN`` 更加灵活、泛用。

.. admonition:: 注意
    :class: note

    在上方的例子中，状态变量 ``v`` 和 ``rho`` 采用了 **默认的全零初始化方式** 。用户可以重写 ``init_states()`` 方法，从而改变状态初始化规则。该方法的原始定义如下，其中 ``*args`` 代表 ``forward()`` 方法的参数：

    .. code:: python

        class FlexSN(base.MemoryModule):

            ...

            @staticmethod
            def init_states(num_states: int, step_mode: str, *args) -> List[torch.Tensor]:
                if step_mode == "s":
                    return [torch.zeros_like(args[0]) for _ in range(num_states)]
                elif step_mode == "m":
                    return [torch.zeros_like(args[0][0]) for _ in range(num_states)]
                else:
                    raise ValueError(f"Unsupported step mode: {step_mode}")

    详见 :meth:`FlexSN.init_states <spikingjelly.activation_based.neuron.flexsn.FlexSN.init_states>` 。

.. admonition:: 注意
    :class: note

    :class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>` 实现了 SpikingJelly 神经元模块的大多数功能。
    若想用轻量化、透明化、函数式的方式直接调用 Triton 内核，请使用 :class:`FlexSNKernel <spikingjelly.activation_based.neuron.flexsn.FlexSNKernel>` 。

.. admonition:: 警告
    :class: warning

    在使用 ``FlexSN`` 时，需注意：

    * 应在 GPU 上运行。
    * Triton 后端仅支持多步运行模式 ``step_mode="m"`` 。
    * PyTorch 后端是通过反复调用 ``core`` 来实现的。
    * ``FlexSN`` 完成一次模拟之后，需要调用 ``reset()`` 方法来重置神经元状态。

FlexSN Inductor 后端
--------------------

``FlexSN`` 自 ``0.0.0.1.0`` 版本起新增 ``backend="inductor"`` 选项，将自定义的单步动力学函数 ``core`` 编译成高效的 Triton GPU 内核。与 ``backend="triton"`` （自研 FX→Triton 映射）相比，Inductor 路径：

* 不需要手动维护算子映射表， ``core`` 中出现的绝大多数 PyTorch 算子均可直接编译；
* 支持 ``torch.compile`` ，可与外层网络联合编译，实现跨层算子融合；
* 推理和训练均内置专用 Triton 核，性能优于或持平 ``backend="triton"`` 。

.. admonition:: 注意
   :class: warning

   * 目前仅支持 CUDA 设备。
   * ``core`` 中的算子需在 ``FX_TO_TRITON`` 映射表内，不在表内的算子自动回退 ``eager_scan`` （日志中会有提示）。
     支持的算子见下方 :ref:`算子覆盖 <flexsn-inductor-op-coverage>` 一节。
   * 训练时 ``core`` 应使用 surrogate gradient（如 :class:`spikingjelly.activation_based.surrogate.Sigmoid` ）
     而非硬阈值，否则梯度为零。

快速上手 — 推理
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

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

.. admonition:: 训练时推荐套 torch.compile （无 fullgraph）
   :class: tip

   训练路径已内置 ``@torch._dynamo.disable`` 包装。套 ``torch.compile()`` （不加 ``fullgraph=True`` ）时，
   Dynamo 在 FlexSN 处产生 graph break，FlexSN 仍使用 Triton 单核正/反向扫描，而前后的 Conv/Linear 层被 Inductor 编译加速。
   实测整体比不加 compile 快约 28%（RTX 4090，T=32）。 ``torch.compile(fullgraph=True)`` 会报错，去掉 ``fullgraph=True`` 即可。

内核分发策略
^^^^^^^^^^^^

``backend="inductor"`` 的 ``multi_step_forward`` 根据上下文自动选择路径：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 条件
     - 路径
   * - 推理（no grad）+ CUDA
     - Triton 单核 scan（ ``tl.static_range(T)`` ，1 次 launch）
   * - 训练 + CUDA（含 torch.compile 内外）
     - ``@torch._dynamo.disable`` 包装的 Triton FlexSNFunction；compile 下产生
       graph break，FlexSN 仍用 Triton 核，周围层被 Inductor 编译
   * - CPU 或 kernel 不可用
     - ``eager_scan`` / ``flex_sn_scan`` HOP fallback

与其它后端对比
^^^^^^^^^^^^^^

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
^^^^^^^^

**推理**：初始化时通过 ``make_fx`` 追踪 ``core`` ，使用 FlexSN 模板生成带 ``tl.static_range(T)`` 时间循环的单个 Triton 扫描内核，每次推理只触发一次 kernel launch。

**训练（不套 torch.compile）**：初始化时通过 ``aot_function`` 同时追踪正向和反向计算图
，生成专用的 Triton 正向核（保存中间值）和反向核（时间逆序扫描），
两者均含 ``tl.static_range(T)`` 时间循环，每方向只触发一次 kernel launch。
在 SpikingVGG-16-BN（T=4, B=64, CIFAR-10）训练基准中，比 LIFNode triton 快约 12%，
比纯 PyTorch 快约 45%。

**训练（套或不套 ``torch.compile `` ）**：FlexSNFunction 经 ``@torch._dynamo.disable`` 包装， ``torch.compile()`` （不含 ``fullgraph=True`` ）下 Dynamo 在此处产生 graph break，
FlexSN 继续使用 Triton 单核扫描，周围层由 Inductor 编译，整体比不加 compile 快约 28%。 ``torch.compile(fullgraph=True)`` 会报错，移除该选项即可。

使用建议
^^^^^^^^

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
     - 单核 scan
   * - CUDA 训练
     - ``"inductor"`` + ``torch.compile()``（可选）
     - Triton 单核 fwd+bwd；compile 下 graph break，周围层编译加速 ~28%
   * - 推理 + 跨层融合
     - ``"inductor"`` + ``torch.compile``
     - 单核 scan + 与外层 Conv/Linear 联合编译

.. _flexsn-inductor-op-coverage:

算子覆盖
^^^^^^^^

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
     - ``_to_copy`` （类型转换）、``scalar_tensor``、``zeros_like``、``ones_like``
   * - 条件选择
     - ``where``、``masked_fill``
   * - 反向专用
     - ``sigmoid_backward``、``tanh_backward``、``threshold_backward``
   * - 杂项
     - ``clone``、``detach``、``spike_fn``

不在表内的算子（如矩阵运算、复杂控制流等）会触发 ``eager_scan`` fallback 并输出 WARNING 日志。
