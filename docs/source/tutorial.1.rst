编码器 SpikingFlow.encoding
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注SpikingFlow.encoding，包括如何使用已有编码器、如何定义新的编码器。

泊松编码器
------------
SNN中，脉冲在层与层之间传递信息。在SpikingFlow中，脉冲被定义为“torch.bool”类型的tensor，也就是离散的0和1。因此，常见的数据\
类型，例如图像，并不能直接输入到SNN中，需要做一些转换，将这些数据转换成脉冲，这一过程即为“编码”。

泊松编码器是最简单，但也效果良好、广泛使用的一种编码。

对于输入 :math:`x \in [0, 1]` ，泊松编码器将其编码为发放次数的分布符合泊松过程的脉冲。实现泊松过程非常简单，因为泊松流具有独\
立增量性、增量平稳性的性质。在一个仿真步长内，令发放脉冲的概率为 :math:`p = x` ，就可以实现泊松编码。

参考SpikingFlow.encoding.PoissonEncoder的代码：

.. code-block:: python

    def forward(self, x):
        '''
        :param x: 要编码的数据，任意形状的tensor，要求x的数据范围必须在[0, 1]

        将输入数据x编码为脉冲，脉冲发放的概率即为对应位置元素的值
        '''
        out_spike = torch.rand_like(x).le(x)
        # torch.rand_like(x)生成与x相同shape的介于[0, 1)之间的随机数， 这个随机数小于等于x中对应位置的元素，则发放脉冲
        return out_spike

使用起来也比较简单：

.. code-block:: python

    pe = encoding.PoissonEncoder()
    x = torch.rand(size=[8])
    print(x)
    for i in range(10):
        print(pe(x))

更复杂的编码器
-------------

更复杂的一些编码器，并不能像泊松编码器这样使用。例如某个编码器，将输入0.1编码成[0, 0, 1, 0]这样的脉冲，由于我们的框架是时间\
驱动的，因此编码器需要逐步输出0, 0, 1, 0。

对于这样的编码器，例如SpikingFlow.encoding.LatencyEncoder，调用时需要先编码一次，也就是条用forward()函数编码数据\
然后再输出T次，也就是调用T次step()，例如：

.. code-block:: python

    x = torch.rand(size=[3, 2])
    max_spike_time = 20
    le = encoding.LatencyEncoder(max_spike_time)

    le(x)
    print(x)
    print(le.spike_time)
    for i in range(max_spike_time):
        print(le.step())


定义新的编码器
-------------

编码器的实现非常灵活，因此在编码器的基类SpikingFlow.encoding.BaseEncoder中，并没有很多的限制。对于在一个仿真步长内就能输出\
所有脉冲的编码器，只需要实现forward()函数；对于需要多个仿真步长才能输出所有编码脉冲的编码器，需要实现forward(),step(),reset()\
函数。

