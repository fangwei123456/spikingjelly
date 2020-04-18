事件驱动 SpikingFlow.event_driven
=======================================

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

而使用事件驱动的SNN仿真，将时间看作是一个维度，同时使用显式的 :math:`V-t` 方程来描述神经元的活动，而不是用微分方程描述。由\
于 :math:`V-t` 是已知的，因此任一时刻的神经元活动都可以直接算出，因此可以并行的计算出神经元在不同时刻的电压值，这也是为何能够\
将时间看作是一个维度。