事件驱动 SpikingFlow.event_driven
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 ``SpikingFlow.event_driven``，介绍事件驱动概念、Tempotron神经元。

需要注意的是，``SpikingFlow.event_driven`` 与 ``SpikingFlow.softbp`` 类似，也是一个相对独立的包，与其他\
的 ``SpikingFlow.*`` 中的神经元、突触等组件不能混用。

事件驱动的SNN仿真
-----------------
``SpikingFlow.softbp`` 使用时间驱动的方法对SNN进行仿真，因此在代码中都能够找到在时间上的循环，例如：

.. code-block:: python

    for t in range(T):
        if t == 0:
            out_spikes_counter = net(encoder(img).float())
        else:
            out_spikes_counter += net(encoder(img).float())

而使用事件驱动的SNN仿真，并不需要在时间上进行循环，神经元的状态更新由事件触发，例如产生脉冲或接受输入脉冲，因而不同神经元的活动\
可以异步计算，不需要在时钟上保持同步。

脉冲响应模型(Spike response model, SRM)
--------------------------------------

在脉冲响应模型(Spike response model, SRM)中，使用显式的 :math:`V-t` 方程来描述神经元的活动，而不是用微分方程去描述神经元的充\
电过程。由于 :math:`V-t` 是已知的，因此给与任何输入 :math:`X(t)`，神经元的响应 :math:`V(t)` 都可以被直接算出。

Tempotron神经元
---------------

Tempotron神经元是 [#f1]_ 提出的一种SNN神经元，其命名来源于ANN中的感知器（Perceptron）。感知器是最简单的ANN神经元，对输入数据\
进行加权求和，输出二值0或1来表示数据的分类结果。Tempotron可以看作是SNN领域的感知器，它同样对输入数据进行加权求和，并输出二分类\
的结果。

Tempotron的膜电位定义为：

.. math::
    V(t) = \sum_{i} w_{i} \sum_{t_{i}} K(t - t_{i}) + V_{reset}

其中 :math:`w_{i}` 是第 :math:`i` 个输入的权重，也可以看作是所连接的突触的权重；:math:`t_{i}` 是第 :math:`i` 个输入的脉冲发\
放时刻，:math:`K(t - t_{i})` 是由于输入脉冲引发的突触后膜电位(postsynaptic potentials, PSPs)；:math:`V_{reset}` 是Tempotron\
的重置电位，或者叫静息电位。

:math:`K(t - t_{i})` 是一个关于 :math:`t_{i}` 的函数(PSP Kernel)，[#f1]_ 中使用的函数形式如下：

.. math::
    K(t - t_{i}) =
    \begin{cases}
    V_{0} (exp(-\frac{t - t_{i}}{\tau}) - exp(-\frac{t - t_{i}}{\tau_{s}})), & t \geq t_{i} \\
    0, & t < t_{i}
    \end{cases}

其中 :math:`V_{0}` 是归一化系数，使得函数的最大值为1；:math:`\tau` 是膜电位时间常数，可以看出输入的脉冲在Tempotron上会引起\
瞬时的点位激增，但之后会指数衰减；:math:`\tau_{s}` 则是突触电流的时间常数，这一项的存在表示突触上传导的电流也会随着时间衰减。

单个的Tempotron可以作为一个二分类器，分类结果的判别，是看Tempotron的膜电位在仿真周期内是否过阈值:

.. math::
    y =
    \begin{cases}
    1, & V_{t_{max}} \geq V_{threshold} \\
    0, & V_{t_{max}} < V_{threshold}
    \end{cases}

其中 :math:`t_{max} = \mathrm{argmax} \{V_{t}\}`。
从Tempotron的输出结果也能看出，Tempotron只能发放不超过1个脉冲。单个Tempotron只能做二分类，但多个Tempotron就可以做多分类。

训练Tempotron
-----------------
使用Tempotron的SNN网络，通常是“全连接层 + Tempotron”的形式，网络的参数即为全连接层的权重。使用梯度下降法来优化网络参数。

以二分类为例，损失函数被定义为仅在分类错误的情况下存在。当实际类别是1而实际输出是0，损失为 :math:`V_{threshold} - V_{t_{max}}`;\
当实际类别是0而实际输出是1，损失为 :math:`V_{t_{max}} - V_{threshold}`。可以统一写为：

.. math::
    E = (y - \hat{y})(V_{threshold} - V_{t_{max}})

直接对参数求梯度，可以得到：

.. math::
    \frac{\partial E}{\partial w_{i}} = (y - \hat{y}) (\sum_{t_{i} < t_{max}} K(t_{max} - t_{i}) \
    + \frac{\partial V(t_{max})}{\partial t_{max}} \frac{\partial t_{max}}{\partial w_{i}}) \\
    = (y - \hat{y})(\sum_{t_{i} < t_{max}} K(t_{max} - t_{i}))

因为 :math:`\frac{\partial V(t_{max})}{\partial t_{max}}=0`。

并行实现
--------
如前所述，对于脉冲响应模型，一旦输入给定，神经元的响应方程已知，任意时刻的神经元状态都可以求解。此外，计算 :math:`t` 时刻的电\
压值，并不需要依赖于 :math:`t-1` 时刻的电压值，因此不同时刻的电压值完全可以并行求解。在 `SpikingFlow/event_driven/neuron.py` 中\
实现了集成全连接层、并行计算的Tempotron，读者如有兴趣可以直接阅读源代码。


.. [#f1] Gutig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature Neuroscience, 2006, 9(3): 420-428.