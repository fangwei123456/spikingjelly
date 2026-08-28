FlexSN
======

English version: :doc:`../en/flexsn`

``FlexSN`` 将纯 PyTorch 单步动力学函数转换为有状态的 SpikingJelly 神经元。
它提供作为参考实现的 Torch 路径、供 ``torch.compile`` 使用的白盒 HOP 路径，
以及面向 CUDA 的生成式 Triton 路径。

Core 契约
---------

单步函数依次接收单步输入、状态和静态输入，并返回输出和更新后的状态：

.. code-block:: text

    core(*step_inputs, *states, *static_inputs)
        -> (*outputs, *updated_states)

构造时只需指定 ``num_states``。输入和输出数量由函数签名和单位张量追踪推导。Tensor 参数
必须通过 ``static_inputs`` 传入，不得隐藏在函数闭包中。

自动状态管理
------------

.. code-block:: python

    import torch
    from spikingjelly.activation_based.neuron import FlexSN

    def lif_core(x, v):
        h = v + (x - v) / 2.0
        spike = torch.sigmoid(h - 1.0)
        return spike, h * (1.0 - spike)

    neuron = FlexSN(lif_core, num_states=1, backend="torch")
    spike_seq = neuron(torch.randn(8, 64, 512))
    final_v = neuron.states[0]
    neuron.reset()

存在多个输出时，``forward`` 返回 tuple。``states`` 和可选的
``state_seqs`` 缓存也统一为 tuple。需要完整状态轨迹时设置
``store_state_seqs=True``。

函数式前向传播
--------------

函数式调用显式接收状态和静态输入，不修改模块状态：

.. code-block:: python

    x = torch.randn(8, 64, 512)
    v0 = torch.zeros_like(x[0])
    (spike_seq,), (final_v,) = neuron.functional_forward(
        (x,), (v0,), static_inputs=()
    )

静态输入
--------

静态输入在每个时间步复用。Parameter 会注册为模块参数，其他 Tensor 注册为
buffer。自动状态 ``forward`` 使用注册值，函数式调用则显式接收这些值。

.. code-block:: python

    def plif_core(x, v, w):
        reciprocal_tau = w.sigmoid()
        h = v + reciprocal_tau * (x - v)
        spike = torch.sigmoid(h - 1.0)
        return spike, h * (1.0 - spike)

    w = torch.nn.Parameter(torch.tensor(0.0))
    neuron = FlexSN(plif_core, num_states=1, static_inputs=(w,), backend="torch")

静态 Tensor 必须是标量，或与单步输入具有相同元素数。不支持任意广播。

后端
----

``torch`` 支持 ``step_mode="s"`` 和 ``"m"``；``hop`` 与 ``triton`` 仅支持
多步模式。构造后仍可修改 backend 和 step mode；非法组合会立即报错。合法切换
保留 states，只清除派生的 ``state_seqs`` 缓存。

``backend="triton"`` 需要 CUDA，并自动准备生成式 kernel，无需示例张量。构建
失败会直接抛出，不会回退其他后端。多步输入 ``T == 0`` 会被拒绝。

迁移
----

原 ``FlexSNKernel`` 类和 ``FlexSN.kernel`` 访问器已删除。显式状态执行请使用
``functional_forward``。构造函数不再接受 ``num_inputs``、``num_outputs``、
``example_inputs``、``example_outputs`` 或 ``requires_grad``。
