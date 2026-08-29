FlexSN
======

本页作者：`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_、`wei.fang <https://github.com/fangwei123456>`_

English version: :doc:`../en/flexsn`

``FlexSN`` 可以把用户用纯 PyTorch 编写的单步神经元动力学转换成有状态的
SpikingJelly 神经元，并为多步 CUDA 计算生成 Triton 内核。预定义 IF、LIF、PLIF
神经元的 Triton 用法见 :doc:`./triton_backend`。

用函数描述神经元动力学
----------------------

绝大多数脉冲神经元在一个离散时间步上的动力学可以写成

.. math::

    Y_1[t], Y_2[t], \dots, V_1[t], V_2[t], \dots =
    f_s\left(X_1[t], X_2[t], \dots, V_1[t-1], V_2[t-1], \dots\right).

其中 :math:`X_i` 是输入，:math:`Y_i` 是输出，:math:`V_i` 是跨时间步保存的状态。
在 ``FlexSN`` 中，对应的 Python 函数签名为

.. code-block:: text

    core(*step_inputs, *states, *static_inputs)
        -> (*outputs, *updated_states)

返回值末尾的 ``num_states`` 个 Tensor 必须与输入状态一一对应。以输入不衰减、
软重置的 LIF 神经元为例：

.. code-block:: python

    import torch

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = 0.5 * v + x
        spike = torch.sigmoid(h - 1.0)
        v = h - spike
        return spike, v

``core`` 必须是纯函数，不能捕获 Tensor 或 ``nn.Module``。普通数值超参数可以放进
闭包；需要训练或随 ``state_dict`` 保存的 Tensor 应通过后文的 ``static_inputs``
传入。

构造一个多状态神经元
--------------------

考虑一个具有双输入、双输出和双状态的神经元。``rho`` 调整第一个输出的阈值，
``y`` 决定膜电位采用硬重置还是软重置：

.. code-block:: python

    import torch

    def complicated_lif_core_generator(beta: float, gamma: float):
        def complicated_lif_core(
            x: torch.Tensor,
            y: torch.Tensor,
            v: torch.Tensor,
            rho: torch.Tensor,
        ):
            h = beta * v + x
            s1 = torch.sigmoid(h - (rho + 1.0))
            s2 = torch.sigmoid(h - 1.0)
            rho = gamma * rho + s1
            v_hard = h * (1.0 - s1)
            v_soft = h - s2
            modulation = torch.sigmoid(y)
            v = v_hard * modulation + v_soft * (1.0 - modulation)
            return s1, s2, v, rho

        return complicated_lif_core

该模型中，前两个返回值是输出，最后两个返回值是更新后的 ``v`` 和 ``rho``：

.. image:: ../../_static/tutorials/flexsn/neuron.png
    :width: 100%

构造 ``FlexSN`` 时指定状态数量。输入和输出数量由函数签名以及构造时的一次单位
Tensor 调用推导，不需要示例输入：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    f = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_states=2,
        step_mode="m",
        backend="triton",
        store_state_seqs=True,
    ).cuda()

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    s1, s2 = f(x, y)
    v_seq, rho_seq = f.state_seqs
    final_v, final_rho = f.states

    print(s1.shape, s2.shape)
    print(v_seq.shape, rho_seq.shape)
    print(final_v.shape, final_rho.shape)

当 ``core`` 只有一个输出时，``forward`` 直接返回 Tensor；有多个输出时返回 tuple。
``states`` 和 ``state_seqs`` 始终是 tuple。处理完一段独立序列后，应调用 ``reset()``
清除托管状态。

模块状态与函数式调用
--------------------

``forward`` 会自动初始化、更新并保存状态。需要显式管理状态时，可以调用
``functional_forward``；该调用不修改模块中的 ``states``：

.. code-block:: python

    f_torch = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_states=2,
        backend="torch",
    )
    initial_states = (
        torch.zeros_like(x[0]),
        torch.zeros_like(x[0]),
    )
    (s1, s2), (final_v, final_rho) = f_torch.functional_forward(
        (x, y), initial_states, static_inputs=()
    )
    assert f_torch.states == (None, None)

默认状态是与第一个输入的单步切片同形的零 Tensor。自定义初始化规则时，可继承
``FlexSN`` 并覆盖 ``init_states``：

.. code-block:: python

    class NonzeroFlexSN(neuron.FlexSN):
        @staticmethod
        def init_states(num_states, step_mode, *inputs):
            reference = inputs[0] if step_mode == "s" else inputs[0][0]
            return tuple(torch.ones_like(reference) for _ in range(num_states))

静态输入
--------

每个时间步重复使用的 Tensor 通过 ``static_inputs`` 传入。``Parameter`` 会注册为
模块参数，其他 Tensor 会注册为 buffer；两者都会出现在 ``state_dict`` 中。以下
PLIF 动力学使用可训练参数控制膜电位衰减：

.. code-block:: python

    def plif_core(x, v, w):
        reciprocal_tau = w.sigmoid()
        h = v + reciprocal_tau * (x - v)
        spike = torch.sigmoid(h - 1.0)
        return spike, h * (1.0 - spike)

    w = torch.nn.Parameter(torch.tensor(0.0))
    plif = neuron.FlexSN(
        plif_core,
        num_states=1,
        static_inputs=(w,),
        backend="torch",
    )

函数式调用必须显式传入静态值，因此可以在不修改模块参数的情况下使用另一组参数：

.. code-block:: python

    x_seq = torch.randn(8, 4)
    v0 = (torch.zeros_like(x_seq[0]),)
    outputs, states = plif.functional_forward(
        (x_seq,), v0, static_inputs=(torch.tensor(1.0),)
    )

静态 Tensor 必须是标量，或与单步输入具有相同元素数；不支持任意广播。

验证前向与反向传播
------------------

``backend="torch"`` 是参考实现。开发新动力学时，应先比较 Torch 和 Triton 的输出、
最终状态、状态轨迹和输入梯度：

.. code-block:: python

    core = complicated_lif_core_generator(beta=0.5, gamma=0.9)
    n_torch = neuron.FlexSN(
        core, 2, backend="torch", store_state_seqs=True
    ).cuda()
    n_triton = neuron.FlexSN(
        core, 2, backend="triton", store_state_seqs=True
    ).cuda()

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    x_torch = x.clone().requires_grad_(True)
    y_torch = y.clone().requires_grad_(True)
    x_triton = x.clone().requires_grad_(True)
    y_triton = y.clone().requires_grad_(True)

    s1_torch, s2_torch = n_torch(x_torch, y_torch)
    s1_triton, s2_triton = n_triton(x_triton, y_triton)
    grad = torch.randn_like(s1_torch)
    s1_torch.backward(grad)
    s1_triton.backward(grad)

    torch.testing.assert_close(s1_triton, s1_torch)
    torch.testing.assert_close(s2_triton, s2_torch)
    torch.testing.assert_close(n_triton.states, n_torch.states)
    torch.testing.assert_close(n_triton.state_seqs, n_torch.state_seqs)
    torch.testing.assert_close(x_triton.grad, x_torch.grad)
    torch.testing.assert_close(y_triton.grad, y_torch.grad)

后端与 ``torch.compile``
------------------------

``FlexSN`` 提供三个后端：

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - 后端
     - 设备
     - 用途
   * - ``"torch"``
     - CPU / CUDA
     - 参考实现；支持单步和多步模式
   * - ``"hop"``
     - CPU / CUDA
     - compiler-visible scan；只支持多步模式，适合与外层网络联合编译
   * - ``"triton"``
     - CUDA
     - 根据 ``core`` 生成专用前向与反向内核；只支持多步模式

HOP 路径可以直接交给 ``torch.compile``：

.. code-block:: python

    model = neuron.FlexSN(lif_core, 1, backend="hop")
    compiled_model = torch.compile(model, fullgraph=True)
    output = compiled_model(torch.randn(8, 64, 512))

Triton 路径在第一次收到真实 CUDA 输入时，按输入的 dtype 和 device 构建运行时，
构造模块时不需要示例 Tensor。外层网络仍可使用 ``torch.compile``：

.. code-block:: python

    import torch.nn as nn

    flex = neuron.FlexSN(lif_core, 1, backend="triton").cuda()
    model = nn.Sequential(
        nn.Linear(512, 512),
        flex,
        nn.Linear(512, 512),
    ).cuda()
    model = torch.compile(model, fullgraph=True)
    output = model(torch.randn(8, 64, 512, device="cuda"))

Triton 不支持的 ``core`` 算子或内核构建失败会直接抛出异常，不会切换到 HOP 或
Torch。选择 Triton 后端后，执行路径不会在用户不知情时改变。

使用限制与迁移
--------------

* 多步输入的首维是时间维 ``T``；``T == 0`` 会被拒绝。
* ``hop`` 和 ``triton`` 只支持 ``step_mode="m"``。
* 修改 backend 或 step mode 会保留最终状态，但清除派生的 ``state_seqs``。
* 旧构造参数 ``num_inputs``、``num_outputs``、``example_inputs``、
  ``example_outputs`` 和 ``requires_grad`` 已删除。
* 旧 ``FlexSNKernel`` 和 ``FlexSN.kernel`` 已删除；显式状态调用统一使用
  ``functional_forward``。
