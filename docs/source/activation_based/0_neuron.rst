时间驱动：神经元
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 :class:`spikingjelly.clock_driven.neuron`，介绍脉冲神经元，和时间驱动的仿真方法。

脉冲神经元模型
----------------
在 ``spikingjelly`` 中，我们约定，只能输出脉冲，即0或1的神经元，都可以称之为“脉冲神经元”。使用脉冲神经元的网络，进而也可以称之为脉冲神经元网络(Spiking Neural Networks, SNNs)。
:class:`spikingjelly.clock_driven.neuron` 中定义了各种常见的脉冲神经元模型，我们以 :class:`spikingjelly.clock_driven.neuron.LIFNode` 为例来介绍脉冲神经元。

首先导入相关的模块：

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.clock_driven import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt

新建一个LIF神经元层：

.. code-block:: python

    lif = neuron.LIFNode(tau=100.)

LIF神经元层有一些构造参数，在API文档中对这些参数有详细的解释：

    - **tau** -- 膜电位时间常数

    - **v_threshold** -- 神经元的阈值电压

    - **v_reset** -- 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；如果设置为 ``None``，则电压会被减去 ``v_threshold``

    - **surrogate_function** -- 反向传播时用来计算脉冲函数梯度的替代函数

其中 ``surrogate_function`` 参数，在前向传播时的行为与阶跃函数完全相同；我们暂时不会用到反向传播，因此可以先不关心反向传播。

你可能会好奇这一层神经元的数量是多少。对于 :class:`spikingjelly.clock_driven.neuron.LIFNode` 中的绝大多数神经元层，神经元的数量是在初始化或调用 ``reset()`` 函数重新初始化后，根据第一次接收的输入的 ``shape`` 自动决定的。

与RNN中的神经元非常类似，脉冲神经元也是有状态的，或者说是有记忆。脉冲神经元的状态变量，一般是它的膜电位 :math:`V[t]`。因此，:class:`spikingjelly.clock_driven.neuron` 中的神经元，都有成员变量 ``v``。可以打印出刚才新建的LIF神经元层的膜电位：

.. code-block:: python

    print(lif.v)
    # 0.0

可以发现，现在的 ``lif.v`` 是 ``0.0``，因为我们还没有给与它任何输入。我们给与几个不同的输入，观察神经元的电压的 ``shape``，可以发现它与输入的
数量是一致的：

.. code-block:: python

    x = torch.rand(size=[2, 3])
    lif(x)
    print('x.shape', x.shape, 'lif.v.shape', lif.v.shape)
    # x.shape torch.Size([2, 3]) lif.v.shape torch.Size([2, 3])
    lif.reset()

    x = torch.rand(size=[4, 5, 6])
    lif(x)
    print('x.shape', x.shape, 'lif.v.shape', lif.v.shape)
    # x.shape torch.Size([4, 5, 6]) lif.v.shape torch.Size([4, 5, 6])

:math:`V[t]` 和输入 :math:`X[t]` 的关系是什么样的？在脉冲神经元中，不仅取决于当前时刻的输入 :math:`X[t]`，还取决于它在上一个时刻末的膜电位 :math:`V[t-1]`。

通常使用阈下（指的是膜电位不超过阈值电压 ``V_{threshold}`` 时）神经动态方程 :math:`\frac{\mathrm{d}V(t)}{\mathrm{d}t} = f(V(t), X(t))` 描述连续时间的脉冲神经元的充电过程，例如对于LIF神经元，充电方程为：

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

其中 :math:`\tau_{m}` 是膜电位时间常数，:math:`V_{reset}` 是重置电压。对于这样的微分方程，由于 :math:`X(t)` 并不是常量，因此难以求出显示的解析解。

:class:`spikingjelly.clock_driven.neuron` 中的神经元，使用离散的差分方程来近似连续的微分方程。在差分方程的视角下，LIF神经元的充电方程为：

.. math::
    \tau_{m} (V[t] - V[t-1]) = -(V[t-1]- V_{reset}) + X[t]

因此可以得到 :math:`V[t]` 的表达式为

.. math::
    V[t] = f(V[t-1], X[t]) = V[t-1] + \frac{1}{\tau_{m}}(-(V[t - 1] - V_{reset}) + X[t])

可以在 :class:`spikingjelly.clock_driven.neuron.LIFNode.neuronal_charge` 中找到如下所示的代码：

.. code-block:: python

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v += (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v += (x - self.v) / self.tau
            else:
                self.v += (x - (self.v - self.v_reset)) / self.tau

不同的神经元，充电方程不尽相同。但膜电位超过阈值电压后，释放脉冲，以及释放脉冲后，膜电位的重置都是相同的。因此它们全部继承自 :class:`spikingjelly.clock_driven.neuron.BaseNode`，共享相同的放电、重置方程。可以在 :class:`spikingjelly.clock_driven.neuron.BaseNode.neuronal_fire` 中找到释放脉冲的代码：

.. code-block:: python

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

``surrogate_function()`` 在前向传播时是阶跃函数，只要输入大于或等于0，就会返回1，否则会返回0。我们将这种元素仅为0或1的 ``tensor`` 视为脉冲。

释放脉冲消耗了神经元之前积累的电荷，因此膜电位会有一个瞬间的降低，即膜电位的重置。在SNN中，对膜电位重置的实现，有2种方式：

#. Hard方式：释放脉冲后，膜电位直接被设置成重置电压：:math:`V[t] = V_{reset}`

#. Soft方式：释放脉冲后，膜电位减去阈值电压：:math:`V[t] = V[t] - V_{threshold}`

可以发现，对于使用Soft方式的神经元，并不需要重置电压 :math:`V_{reset}` 这个变量。:class:`spikingjelly.clock_driven.neuron` 中的神经元，在构造函数的参数之一 ``v_reset``，默认为 ``1.0`` ，表示神经元使用Hard方式；若设置为 ``None``，则会使用Soft方式。在 :class:`spikingjelly.clock_driven.neuron.BaseNode.neuronal_fire.neuronal_reset` 中可以找到膜电位重置的代码：

.. code-block:: python

    def neuronal_reset(self):
        # ...
        if self.v_reset is None:
            self.v = self.v - self.spike * self.v_threshold
        else:
            self.v = (1. - self.spike) * self.v + self.spike * self.v_reset


描述离散脉冲神经元的三个方程
-------------------------------

至此，我们可以用充电、放电、重置，这3个离散方程来描述任意的离散脉冲神经元。充电、放电方程为：

.. math::
    H[t] & = f(V[t-1], X[t]) \\
    S[t] & = g(H[t] - V_{threshold}) = \Theta(H[t] - V_{threshold})

其中 :math:`\Theta(x)` 即为构造函数参数中的 ``surrogate_function()``，是一个阶跃函数：

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

其中 :math:`V[t]` 是神经元的膜电位；:math:`X[t]` 是外源输入，例如电压增量；为了避免混淆，我们使用 :math:`H[t]` 表示神经元充电后、释放脉冲前的膜电位；:math:`V[t]` 是神经元释放脉冲后的膜电位；:math:`f(V[t-1], X[t])` 是神经元的状态更新方程，不同的神经元，区别就在于更新方程不同。

时间驱动的仿真方式
----------------------

:class:`spikingjelly.clock_driven` 使用时间驱动的方式，对SNN逐步进行仿真。

接下来，我们将逐步给与神经元输入，并查看它的膜电位和输出脉冲。

现在让我们给与LIF神经元层持续的输入，并画出其放电后的膜电位和输出脉冲：

.. code-block:: python

    lif.reset()
    x = torch.as_tensor([2.])
    T = 150
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(lif(x))
        v_list.append(lif.v)

    visualizing.plot_one_neuron_v_s(np.asarray(v_list), np.asarray(s_list), v_threshold=lif.v_threshold, v_reset=lif.v_reset,
                                    dpi=200)
    plt.show()

我们给与的输入 ``shape=[1]``，因此这个LIF神经元层只有1个神经元。它的膜电位和输出脉冲随着时间变化情况如下：

.. image:: ../_static/tutorials/clock_driven/0_neuron/0.*
    :width: 100%

下面我们将神经元层重置，并给与 ``shape=[32]`` 的输入，查看这32个神经元的膜电位和输出脉冲：

.. code-block:: python

    lif.reset()
    x = torch.rand(size=[32]) * 4
    T = 50
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(lif(x).unsqueeze(0))
        v_list.append(lif.v.unsqueeze(0))

    s_list = torch.cat(s_list)
    v_list = torch.cat(v_list)

    visualizing.plot_2d_heatmap(array=np.asarray(v_list), title='Membrane Potentials', xlabel='Simulating Step',
                                ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
    visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                               ylabel='Neuron Index', dpi=200)
    plt.show()

结果如下：

.. image:: ../_static/tutorials/clock_driven/0_neuron/1.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/0_neuron/2.*
    :width: 100%