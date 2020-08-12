时间驱动：神经元
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 ``SpikingFlow.clock_driven.neuron``，介绍脉冲神经元，和时间驱动的仿真方法。

脉冲神经元模型
----------------
在 ``SpikingFlow`` 中，我们约定，只能输出脉冲，即0或1的神经元，都可以称之为“脉冲神经元”。使用脉冲神经元的网络，进而也可以称之为
脉冲神经元网络（Spiking Neural Networks）。
``SpikingFlow.clock_driven.neuron`` 中定义了各种常见的脉冲神经元模型，我们以 ``SpikingFlow.clock_driven.neuron.LIFNode``
为例来介绍脉冲神经元。首先导入相关的模块，并新建一个LIF神经元层：

.. code-block:: python

    from SpikingFlow.clock_driven import neuron
    lif = neuron.LIFNode()

与RNN中的神经元非常类似，脉冲神经元也是有状态的，或者说是有记忆。脉冲神经元的状态变量，一般是它的膜电位 :math:`V_{t}`。因此，
``SpikingFlow.clock_driven.neuron`` 中的神经元，都有成员变量 ``v``。可以打印出刚才新建的LIF神经元层的膜电位：

.. code-block:: python
    print(lif.v)  # 0.0

可以发现，现在的 ``v`` 是 ``0.0``，因为我们还没有给与它任何输入。那么 :math:`V_{t}` 和输入 :math:`X_{t}` 的关系是什么样的？在
脉冲神经元中，不仅取决于当前时刻的输入 :math:`X_{t}`，还取决于它在上一个时刻末的膜电位 :math:`V_{t-1}`。

通常使用阈下（指的是膜电位不超过阈值电压 ``V_{threshold}`` 时）微分方程 :math:`\frac{\mathrm{d}V(t)}{\mathrm{d}t} = f(V(t), X(t))` 描述连续时间的
脉冲神经元的充电过程，例如对于LIF神经元，更新方程为：

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

其中 :math:`\tau_{m}` 是膜电位时间常数，:math:`V_{reset}` 是重置电压。对于这样的微分方程，由于 :math:`X(t)` 并不是常量，因此
难以求出显示的解析解。

``SpikingFlow.clock_driven.neuron`` 中的神经元，使用离散的差分方程来近似连续的微分方程。在差分方程的视角下，LIF神经元的充电方程为：

.. math::
    \tau_{m} (V_{t} - V_{t-1}) = -(V{t-1} - V_{reset}) + X(t)

因此可以得到 :math:`V_{t}` 的表达式为

.. math::
    V_{t} = f(V_{t-1}, X_{t}) = V_{t-1} + \frac{1}{\tau_{m}}(-(V_{t - 1} - V_{reset}) + X_{t})

可以在 ``LIFNode`` 的 ``forward`` 中找到对应的代码：

.. code-block:: python

    def forward(self, dv: torch.Tensor):
        self.v += (dv - (self.v - self.v_reset)) / self.tau
        return self.spiking()

脉冲神经元的另一个普遍特性是，当膜电位超过阈值电压后，神经元会释放脉冲。释放脉冲消耗了神经元之前积累的电荷，因此膜电位会有一个瞬间
的降低。


