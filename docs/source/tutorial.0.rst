神经元 SpikingFlow.neuron
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 ``SpikingFlow.neuron``，包括如何使用已有神经元、如何定义新的神经元。

LIF神经元仿真
------------

我们使用一个LIF神经元进行仿真，代码如下：

.. code-block:: python

    import SpikingFlow
    import SpikingFlow.neuron as neuron
    # 导入绘图模块
    from matplotlib import pyplot
    import torch

    # 新建一个LIF神经元
    lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
    # 新建一个空list，保存仿真过程中神经元的电压值
    v_list = []
    # 新建一个空list，保存神经元的输出脉冲
    spike_list = []

    T = 200
    # 运行200次
    for t in range(T):
        # 前150次，输入电流都是0.1
        if t < 150:
            spike_list.append(lif_node(0.1).float().item())
        # 后50次，不输入，也就是输入0
        else:
            spike_list.append(lif_node(0).float().item())

        # 记录每一次输入后，神经元的电压
        v_list.append(lif_node.v.item())

    # 画出电压的变化
    pyplot.subplot(2, 1, 1)
    pyplot.plot(v_list, label='v')
    pyplot.xlabel('t')
    pyplot.ylabel('voltage')
    pyplot.legend()

    # 画出脉冲
    pyplot.subplot(2, 1, 2)
    pyplot.bar(torch.arange(0, T).tolist(), spike_list, label='spike')
    pyplot.xlabel('t')
    pyplot.ylabel('spike')
    pyplot.legend()
    pyplot.show()

    print('t', 'v', 'spike')
    for t in range(T):
        print(t, v_list[t], spike_list[t])

运行后得到的电压和脉冲如下：

.. image:: ./_static/tutorials/1.png

你会发现，LIF神经元在有恒定输入电流时，电压会不断增大，但增速越来越慢。

如果输入电流不是足够大，最终在每个dt内，LIF神经元的电压衰减值会恰好等于输入电流造成的电压增加值，电压不再增大，导致无法充电到\
过阈值、发放脉冲。

当停止输入后，LIF神经元的电压会指数衰减，从图中500个dt后的曲线可以看出。

我们修改代码，给予更大的电流输入：

.. code-block:: python

    ...
    for t in range(T):
        # 前150次，输入电流都是0.12
        if t < 150:
            spike_list.append(lif_node(0.12).float().item())
    ...

运行后得到的电压和脉冲如下（需要说明的是，脉冲是以pyplot柱状图的形式\
画出，当柱状图的横轴，也就是时间太长时，而图像的宽度又不够大，一些“落单”的脉冲在图像上会无法画出，因为宽度小于一个像素点）：

.. image:: ./_static/tutorials/2.png

可以发现，LIF神经元已经开始发放脉冲了：

定义新的神经元
-------------

在SNN中，不同的神经元模型，区别往往体现在描述神经元的微分方程。上文所使用的LIF神经元，描述其动态特性的微分方程为：

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

其中 :math:`\tau_{m}` 是细胞膜的时间常数， :math:`V(t)` 是膜电位， :math:`V_{reset}` 是静息电压， :math:`R_{m}` 是膜电\
阻， :math:`I(t)` 是输入电流

SpikingFlow是时间驱动（time-driven）的框架，即将微分方程视为差分方程，通过逐步仿真来进行计算。例如LIF神经元，\
代码位于 ``SpikingFlow.neuron.LIFNode``，参考它的实现：

.. code-block:: python

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return: out_spike: shape与self.shape相同，输出脉冲
        '''
        out_spike = self.next_out_spike

        # 将上一个dt内过阈值的神经元重置
        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset

        v_decay = -(self.v - self.v_reset)
        self.v += (self.r * i + v_decay) / self.tau
        self.next_out_spike = (self.v >= self.v_threshold)
        self.v[self.next_out_spike] = self.v_threshold
        self.v[self.v < self.v_reset] = self.v_reset

        return out_spike

从代码中可以发现，t-dt时刻电压没有达到阈值，t时刻电压达到了阈值，则到t+dt时刻才会放出脉冲。这是为了方便查看波形图，如果不这样\
设计，若t-dt时刻电压为0.1，v_threshold=1.0，v_reset=0.0, t时刻增加了0.9，直接在t时刻发放脉冲，则从波形图上看，电压从0.1直接\
跳变到了0.0，不利于进行数据分析。

此外，“脉冲”被定义为“torch.bool”类型的变量。SNN中的神经元，输出的应该是脉冲而不是电压之类的其他值。

如果想自行实现其他类型的神经元，只需要继承 ``SpikingFlow.neuron.BaseNode``，并实现 ``__init__()``, ``forward()``, ``reset()`` 函数即可。