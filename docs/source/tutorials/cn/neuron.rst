神经元
=======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/neuron`

本节教程主要关注 :class:`spikingjelly.activation_based.neuron`，介绍脉冲神经元。

脉冲神经元模型
-------------------------------------------
在 ``spikingjelly`` 中，我们约定，只要是输出脉冲，即0或1的神经元，都可以称之为“脉冲神经元”。使用脉冲神经元的网络，进而也可以称之为脉冲神经元网络(Spiking Neural Networks, SNNs)。\
:class:`spikingjelly.activation_based.neuron` 中定义了各种常见的脉冲神经元模型，我们以 :class:`spikingjelly.activation_based.neuron.IFNode` 为例来介绍脉冲神经元。

首先导入相关的模块：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt

新建一个IF神经元层：

.. code-block:: python

    if_layer = neuron.IFNode()

IF神经元层有一些构造参数，在API文档中对这些参数有详细的解释，我们暂时只关注下面几个重要的参数：

    - **v_threshold** -- 神经元的阈值电压

    - **v_reset** -- 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；如果设置为 ``None``，则电压会被减去 ``v_threshold``

    - **surrogate_function** -- 反向传播时用来计算脉冲函数梯度的替代函数


你可能会好奇这一层神经元的数量是多少。对于 :class:`spikingjelly.activation_based.neuron.IFNode` 中的绝大多数神经元层，神经元的数量是在初始化或调用 ``reset()`` 函数重新初始化后，根据第一次接收的输入的 ``shape`` 自动决定的。\

与RNN中的神经元非常类似，脉冲神经元也是有状态的，或者说是有记忆。脉冲神经元的状态变量，一般是它的膜电位 :math:`V[t]`。因此，:class:`spikingjelly.activation_based.neuron` 中的神经元，都有成员变量 ``v``。可以打印出刚才新建的IF神经元层的膜电位：

.. code-block:: python

    print(if_layer.v)
    # if_layer.v=0.0

可以发现，现在的 ``if_layer.v`` 是 ``0.0``，因为我们还没有给与它任何输入。我们给与几个不同的输入，观察神经元的电压的 ``shape``，可以发现它与输入的\
数量是一致的：

.. code-block:: python

    x = torch.rand(size=[2, 3])
    if_layer(x)
    print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
    # x.shape=torch.Size([2, 3]), if_layer.v.shape=torch.Size([2, 3])
    if_layer.reset()

    x = torch.rand(size=[4, 5, 6])
    if_layer(x)
    print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
    # x.shape=torch.Size([4, 5, 6]), if_layer.v.shape=torch.Size([4, 5, 6])
    if_layer.reset()

脉冲神经元是有状态的，在输入下一个样本前，一定要先调用 ``reset()`` 函数清除之前的状态。

:math:`V[t]` 和输入 :math:`X[t]` 的关系是什么样的？在脉冲神经元中，:math:`V[t]` 不仅取决于当前时刻的输入 :math:`X[t]`，还取决于它在上一个时刻末的膜电位 :math:`V[t-1]`。

通常使用阈下（指的是膜电位不超过阈值电压 :math:`V_{threshold}` 时）神经动态方程 :math:`\frac{\mathrm{d}V(t)}{\mathrm{d}t} = f(V(t), X(t))` 描述连续时间的脉冲神经元的充电过程，例如对于IF神经元，充电方程为：

.. math::
    \frac{\mathrm{d}V(t)}{\mathrm{d}t} = X(t)

:class:`spikingjelly.activation_based.neuron` 中的神经元，使用离散的差分方程来近似连续的微分方程。在差分方程的视角下，IF神经元的充电方程为：

.. math::
    V[t] - V[t-1] = X[t]

因此可以得到 :math:`V[t]` 的表达式为

.. math::
    V[t] = f(V[t-1], X[t]) = V[t-1] + X[t]

``SimpleBaseNode`` 中的 ``Simple`` 描述的是接口目标，并不是一种神经元数学模型。
该接口使用纯 PyTorch 直接展开 SNN 神经元的充电、放电和重置职责，便于使用者理解
神经元在 SNN 中承担的工作，也便于修改动力学方程。在基于该接口实现的
:class:`spikingjelly.activation_based.neuron.SimpleIFNode` 中，充电方程直接写为：

.. code-block:: python

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

不同的神经元具有不同的充电方程，但通常共享放电和重置方程。
:class:`spikingjelly.activation_based.neuron.SimpleBaseNode` 通过
``neuronal_charge → neuronal_fire → neuronal_reset`` 路径直接表达这三个阶段；
生产级神经元则在 functional 状态转移或专用 kernel 中实现等价计算。下面是
``SimpleBaseNode.neuronal_fire`` 的核心计算：

.. code-block:: python

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

``surrogate_function()`` 在前向传播时是阶跃函数，只要输入大于或等于0，就会返回1，否则会返回0。我们将这种元素仅为0或1的 ``tensor`` 视为脉冲。

释放脉冲消耗了神经元之前积累的电荷，因此膜电位会有一个瞬间的降低，即膜电位的重置。在SNN中，对膜电位重置的实现，有2种方式：

#. Hard方式：释放脉冲后，膜电位直接被设置成重置电压：:math:`V[t] = V_{reset}`

#. Soft方式：释放脉冲后，膜电位减去阈值电压：:math:`V[t] = V[t] - V_{threshold}`

可以发现，对于使用Soft方式的神经元，并不需要重置电压 :math:`V_{reset}` 这个变量。在当前实现中， :class:`spikingjelly.activation_based.neuron` 中的大多数神经元构造函数中的 ``v_reset`` 默认值均为 ``0.0`` ，表示神经元默认使用Hard方式；若设置为 ``None``，则会使用Soft方式。``SimpleBaseNode.neuronal_reset`` 使用如下等价逻辑：

.. code-block:: python

    # The following codes are for tutorials. The actual codes are different, but have the similar behavior.

    def neuronal_reset(self):
        if self.v_reset is None:
            self.v = self.v - self.spike * self.v_threshold
        else:
            self.v = (1. - self.spike) * self.v + self.spike * self.v_reset


描述离散脉冲神经元的三个方程
-------------------------------

至此，我们可以用充电、放电、重置，这3个离散方程来描述任意的离散脉冲神经元。充电、放电方程为：

.. math::
    H[t] & = f(V[t-1], X[t]) \\
    S[t] & = \Theta(H[t] - V_{threshold})

其中 :math:`\Theta(x)` 即为构造函数参数中的 ``surrogate_function``，是一个阶跃函数：

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

Hard方式重置方程为：

.. math::
    V[t] = H[t] \cdot (1 - S[t]) + V_{reset} \cdot S[t]

Soft方式重置方程为：

.. math::
    V[t] = H[t] - V_{threshold} \cdot S[t]

其中 :math:`X[t]` 是外源输入，例如电压增量；为了避免混淆，我们使用 :math:`H[t]` 表示神经元充电后、释放脉冲前的膜电位；:math:`V[t]` 是神经元释放脉冲后的膜电位；:math:`f(V[t-1], X[t])` 是神经元的状态更新方程，不同的神经元，区别就在于更新方程不同。

神经元的动态如下图所示（图片来自 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_）：

.. image:: ../../_static/tutorials/neuron/neuron.*
    :width: 100%


仿真
-------------------------------------------

接下来，我们将逐步给与神经元输入，并查看它的膜电位和输出脉冲。

现在让我们给与IF神经元层持续的输入，并画出其放电后的膜电位和输出脉冲：

.. code-block:: python

    if_layer.reset()
    x = torch.as_tensor([0.02])
    T = 150
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(if_layer(x))
        v_list.append(if_layer.v)

    dpi = 300
    figsize = (12, 8)
    visualizing.plot_one_neuron_v_s(torch.cat(v_list).numpy(), torch.cat(s_list).numpy(), v_threshold=if_layer.v_threshold,
                                    v_reset=if_layer.v_reset,
                                    figsize=figsize, dpi=dpi)
    plt.show()

我们给与的输入 ``shape=[1]``，因此这个IF神经元层只有1个神经元。它的膜电位和输出脉冲随着时间变化情况如下：

.. image:: ../../_static/tutorials/neuron/0.*
    :width: 100%

下面我们将神经元层重置，并给与 ``shape=[32]`` 的输入，查看这32个神经元的膜电位和输出脉冲：

.. code-block:: python

    if_layer.reset()
    T = 50
    x = torch.rand([32]) / 8.
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(if_layer(x).unsqueeze(0))
        v_list.append(if_layer.v.unsqueeze(0))

    s_list = torch.cat(s_list)
    v_list = torch.cat(v_list)

    figsize = (12, 8)
    dpi = 200
    visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
                                ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)


    visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
                            ylabel='neuron index', figsize=figsize, dpi=dpi)

    plt.show()


结果如下：

.. image:: ../../_static/tutorials/neuron/1.*
    :width: 100%

.. image:: ../../_static/tutorials/neuron/2.*
    :width: 100%

步进模式和后端
-------------------------------------------
在 :doc:`./basic_concept` 中我们已经介绍过单步和多步模式，在本教程前面的内容中，我们使用的都是\
单步模式。切换成多步模式非常简单，只需要设置 ``step_mode`` 即可：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, functional
    if_layer = neuron.IFNode(step_mode='s')
    T = 8
    N = 2
    x_seq = torch.rand([T, N])
    y_seq = functional.multi_step_forward(x_seq, if_layer)
    if_layer.reset()

    if_layer.step_mode = 'm'
    y_seq = if_layer(x_seq)
    if_layer.reset()

此外，部分神经元在单步和多步模式下都支持 ``cupy`` 后端； ``IFNode`` , ``LIFNode`` 和 ``ParametricLIFNode`` 等神经元在多步模式下还支持 ``triton`` 后端。设置 ``backend`` 之后，前反向传播会使用对应后端进行加速。

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    if_layer = neuron.IFNode()
    print(f'if_layer.backend={if_layer.backend}')
    # if_layer.backend=torch

    print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
    # step_mode=s, supported_backends=('torch', 'cupy')

    if_layer.step_mode = 'm'
    print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
    # step_mode=m, supported_backends=('torch', 'cupy', 'triton')

    device = 'cuda:0'
    if_layer.to(device)
    if_layer.backend = 'cupy'  # switch to the cupy backend
    print(f'if_layer.backend={if_layer.backend}')
    # if_layer.backend=cupy

    x_seq = torch.rand([8, 4], device=device)
    y_seq = if_layer(x_seq)
    if_layer.reset()

    if_layer.backend = 'triton'  # switch to the triton backend
    print(f'if_layer.backend={if_layer.backend}')
    # if_layer.backend=triton

    y_seq = if_layer(x_seq)
    if_layer.reset()

自定义神经元
-------------------------------------------
SpikingJelly 为修改神经元动力学和高性能执行提供了两类接口。``SimpleBaseNode`` 是
纯 PyTorch 的教学与自定义接口，而不是新的神经元数学模型；它优先保证职责和方程清晰，
使使用者能直接修改动力学。

.. list-table:: 神经元扩展接口
    :header-rows: 1
    :widths: 20 30 25 25

    * - 基类
      - 前向模型
      - ``to_functional_forward``
      - 适用场景
    * - :class:`SimpleBaseNode <spikingjelly.activation_based.neuron.SimpleBaseNode>`
      - ``neuronal_charge`` → ``neuronal_fire`` → ``neuronal_reset``
      - 通用状态替换路径
      - 教学、动力学实验和快速原型
    * - :class:`BaseNode <spikingjelly.activation_based.neuron.BaseNode>`
      - 原生 functional 状态转移，可使用专用多步 kernel
      - 直接调用 functional forward
      - 生产级神经元和后端实现

若只需要修改神经元方程，应继承 ``SimpleBaseNode``。其单步前向固定按照充电、放电、
重置的顺序执行，多步前向则逐时间步调用完整的单步前向。因而通常只需要实现
``neuronal_charge``。``SimpleIFNode`` 和 ``SimpleLIFNode`` 也是基于这一接口实现的。

``SimpleBaseNode`` 没有原生 functional 状态转移。直接调用 ``functional_forward`` 会报错。
对其实例调用
:func:`to_functional_forward <spikingjelly.activation_based.base.to_functional_forward>`
时，会通过通用 fallback 临时换入显式状态、执行原有前向并恢复模块状态。这能保持方程扩展
接口，但效率低于
``LIFNode`` 等原生 functional 神经元，因此不适合作为依赖频繁 functional 转换的高性能实现。

生产级 ``MemoryModule`` 在 functional forward 前会调用
``materialize_states(inputs, states, step_mode)``。
默认实现原样返回状态；只有当标量或空状态需要依据当前输入转换成张量时才需要重写。
``inputs`` 是当前前向传播的完整输入；实现可依据 ``step_mode`` 在多步模式下选取第一个
时间步作为形状和设备参照，不能假定所有输入都具有时间维。该方法返回新的状态元组，不应
修改模块 memory。旧的 ``BaseNode.v_float_to_tensor`` 已删除。


假设我们构造一种平方积分发放神经元，其充电方程为：

.. math::
    V[t] = f(V[t-1], X[t]) = V[t-1] + X[t]^{2}

实现方式如下：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    class SquareIFNode(neuron.SimpleBaseNode):
        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x.square()

使用平方积分发放神经元进行单步或多步传播：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    class SquareIFNode(neuron.SimpleBaseNode):
        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x.square()

    sif_layer = SquareIFNode()

    T = 4
    N = 1
    x_seq = torch.rand([T, N])
    print(f'x_seq={x_seq}')

    for t in range(T):
        yt = sif_layer(x_seq[t])
        print(f'sif_layer.v[{t}]={sif_layer.v}')

    sif_layer.reset()
    sif_layer.step_mode = 'm'
    y_seq = sif_layer(x_seq)
    print(f'y_seq={y_seq}')
    sif_layer.reset()


输出为

.. code-block:: shell

    x_seq=tensor([[0.7452],
            [0.8062],
            [0.6730],
            [0.0942]])
    sif_layer.v[0]=tensor([0.5554])
    sif_layer.v[1]=tensor([0.])
    sif_layer.v[2]=tensor([0.4529])
    sif_layer.v[3]=tensor([0.4618])
    y_seq=tensor([[0.],
            [1.],
            [0.],
            [0.]])

若要实现与 ``LIFNode`` 一样可直接转换并可接入 CuPy、Triton 或 Inductor kernel 的生产级
神经元，应继承 ``BaseNode`` 并实现 ``single_step_functional_forward``。该方法的接口为
``(self, inputs, states, **kwargs) -> (outputs, updated_states)``，且不得修改模块中注册的
memory 或传入的 ``states``。只有存在独立序列实现或专用 kernel 时，才需要重写
``multi_step_functional_forward``。

.. warning::

    ``BaseNode`` 的常规前向已改为 functional-backed，并已移除 ``neuronal_charge``、
    ``neuronal_fire`` 和 ``neuronal_reset``。旧代码若通过这些方法修改 Python 神经元方程，
    请将基类改为 ``SimpleBaseNode``；原有方程无需重写。生产级神经元或自定义后端应迁移为
    上述 functional 接口。
