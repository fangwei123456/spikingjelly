突触连接
=======================================

本节教程主要关注SpikingFlow.connection，包括如何使用已有突触连接、如何定义新的突触连接

脉冲电流转换器
-------------
SpikingFlow中，神经元的输出都是torch.bool类型的脉冲，而输入则为torch.float类型的电流。将脉冲转换为电流的转换器，功能上类似\
于突触连接，但又有所不同，因此被视为突触连接包的子模块，定义在SpikingFlow.connection.transform中

最简单的将脉冲转换为电流的方式，自然是不做任何处理，直接转换。SpikingFlow.connection.transform.SpikeCurrent正是这样做的：

.. code-block:: python

    def forward(self, in_spike):
        '''
        :param in_spike: 输入脉冲
        :return: 输出电流

        简单地将输入脉冲转换为0/1的浮点数，然后乘以amplitude
        '''
        return in_spike.float() * self.amplitude

其中self.amplitude是初始化时给定的放大系数

更复杂的转换器，例如SpikingFlow.connection.transform.ExpDecayCurrent，具有指数衰减的特性。若当前时刻到达一个脉冲，则\
电流变为amplitude，否则电流按时间常数为tau进行指数衰减：

.. code-block:: python

        def forward(self, in_spike):
        '''
        :param in_spike: 输入脉冲
        :return: 输出电流
        '''
        in_spike_float = in_spike.float()
        i_decay = -self.i / self.tau
        self.i += i_decay * (1 - in_spike_float) + self.amplitude * in_spike_float
        return self.i

ExpDecayCurrent可以看作是一个能够瞬间充满电的电容器，有脉冲作为输入时，则直接充满电；无输入时则自行放电。这种特性，使得\
ExpDecayCurrent与SpikeCurrent相比，具有了“记忆”，因而它需要额外定义一个重置状态的函数：

.. code-block:: python

        def reset(self):
        '''
        :return: None
        重置所有状态变量为初始值，对于ExpDecayCurrent而言，直接将电流设置为0即可
        '''
        self.i = 0

定义新的脉冲电流转换器
---------------------

从之前的例子也可以看出，脉冲电流转换器的定义比较简单，接受torch.bool类型的脉冲作为输入，输出torch.float类型的电流。只需\
要继承SpikingFlow.connection.transform.BaseTransformer，实现__init__()和forward()函数，对于有记忆（状态）的转换器，则额\
外实现一个reset()函数

突触连接
-------
SpikingFlow里的突触连接，与传统ANN中的连接非常类似，都可以看作是一个简单的矩阵，而信息在突触上的传递，则可以看作是矩阵运算

例如SpikingFlow.connection.Linear，与PyTorch中的nn.Linear的行为基本相同：

.. code-block:: python

    def forward(self, x):
        '''
        :param x: 输入电流，shape=[batch_size, *, in_num]
        :return: 输出电流，shape=[batch_size, *, out_num]
        '''
        return torch.matmul(x, self.w.t())


定义新的突触连接
---------------
定义新的突触连接，与定义新的脉冲电流转换器非常类似，只需要继承SpikingFlow.connection.BaseConnection，实现__init__()和\
forward()函数。对于有记忆（状态）的突触，也需要额外实现reset()函数

