自连接和有状态突触
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

自连接模块
-----------------------
自连接指的是从输出到输入的连接，例如 [#Effective]_ 一文中的SRNN(recurrent networks of spiking neurons)，如下图所示：

.. image:: ../_static/tutorials/clock_driven/15_recurrent_connection_and_stateful_synapse/SRNN_example.*
    :width: 100%

使用惊蜇框架很容易构建出带有自连接的模块。考虑最简单的一种情况，我们给神经元增加一个回路，使得它在 :math:`t` 时刻的输出 :math:`s[t]`，会与下一个时刻的外界
输入 :math:`x[t+1]` 相加，共同作为输入。这可以由 :class:`spikingjelly.clock_driven.layer.ElementWiseRecurrentContainer` 轻松实现。
``ElementWiseRecurrentContainer`` 是一个包装器，给任意的 ``sub_module`` 增加一个额外的自连接。连接的形式可以使用用户自定义的逐元素函数
操作 :math:`z=f(x, y)` 来实现。记 :math:`x[t]` 为 :math:`t` 时刻整个模块的输入，:math:`i[t]` 和 :math:`y[t]` 是 ``sub_module`` 的
输入和输出（注意 :math:`y[t]` 同时也是整个模块的输出），则

.. math::

    i[t] = f(x[t], y[t-1])

其中 :math:`f` 是用户自定义的逐元素操作。默认 :math:`y[-1] = 0`。

现在让我们用 ``ElementWiseRecurrentContainer`` 来包装一个IF神经元，逐元素操作设置为加法，因而

.. math::

    i[t] = x[t] + y[t-1].

我们使用软重置，且给与 :math:`x[t]=[1.5, 0, ..., 0]` 的输入：

.. code-block:: python

    T = 8
    def element_wise_add(x, y):
        return x + y
    net = ElementWiseRecurrentContainer(neuron.IFNode(v_reset=None), element_wise_add)
    print(net)
    x = torch.zeros([T])
    x[0] = 1.5
    for t in range(T):
        print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

    functional.reset_net(net)

输出为：

.. code-block:: bash

    ElementWiseRecurrentContainer(
      element-wise function=<function element_wise_add at 0x000001FE0F7968B0>
      (sub_module): IFNode(
        v_threshold=1.0, v_reset=None, detach_reset=False
        (surrogate_function): Sigmoid(alpha=1.0, spiking=True)
      )
    )
    0 x[t]=1.5, s[t]=1.0
    1 x[t]=0.0, s[t]=1.0
    2 x[t]=0.0, s[t]=1.0
    3 x[t]=0.0, s[t]=1.0
    4 x[t]=0.0, s[t]=1.0
    5 x[t]=0.0, s[t]=1.0
    6 x[t]=0.0, s[t]=1.0
    7 x[t]=0.0, s[t]=1.0

可以发现，由于存在自连接，即便 :math:`t \\geu 1` 时 :math:`x[t]=0`，由于输出的脉冲能传回到输入，神经元也能持续释放脉冲。

可以使用 :class:`spikingjelly.clock_driven.layer.LinearRecurrentContainer` 实现更复杂的全连接形式的自连接。

有状态的突触
-----------------------

[#Unsupervised]_ [#Exploiting]_ 等文章使用有状态的突触。将 :class:`spikingjelly.clock_driven.layer.SynapseFilter` 放在普通无状
态突触的后面，对突触输出的电流进行滤波，就可以得到有状态的突触，例如：

.. code-block:: python

    stateful_conv = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
        SynapseFilter(tau=100, learnable=True)
    )


.. [#Effective] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

.. [#Unsupervised] Diehl P U, Cook M. Unsupervised learning of digit recognition using spike-timing-dependent plasticity[J]. Frontiers in computational neuroscience, 2015, 9: 99.

.. [#Exploiting] Fang H, Shrestha A, Zhao Z, et al. Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network[J].
