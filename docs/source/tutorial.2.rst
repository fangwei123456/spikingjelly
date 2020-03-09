仿真器 SpikingFlow.simulating
=======================================
本教程作者： `Yanqi-Chen <https://github.com/Yanqi-Chen>`_

本节教程主要关注 ``SpikingFlow.simulating``，包括如何使用仿真器。

仿真原理
------------
所谓“仿真器”，即为将多个模块囊括其中，并统一运行的工具。实现STDP等学习功能，需要在仿真器层面进行操作。因此，如果想要实现其他\
类型的学习功能，需要对仿真器的实现过程有一定的了解。

采取时间驱动（time-driven）的模型均是由若干个module顺序连接而成，前一个module的输出作为后一个的输入。设输入为  :math:`x_0`, 第i个module :math:`M_i` 的输出为 :math:`x_{i+1}` ，仿真的数据流可以简单地描述为 

.. math::
    x_0 \stackrel{M_0}{\longrightarrow} x_1 \stackrel{M_1}{\longrightarrow} \dots \stackrel{M_{n-1}}{\longrightarrow} x_n

仿真器建立时，首先将所有需要用到的module按顺序存放在 ``module_list`` 中，在每一个仿真时间步内， ``x[i]`` 将经由上述路径到达输出 ``x[i+1]`` 。仿真器用一个列表 ``pipeline`` 依次记录各个module在当前仿真时间步的输出，特别地， ``pipeline[0]`` 为输入。

在整个仿真过程中，输入保持固定，随着仿真时间的推移，脉冲逐渐向后面的module传递，如下面的代码所示。

.. code-block:: python

    ...
    for i in range(self.module_list.__len__(), 0, -1):
        #  x[n] = module[n-1](x[n-1])
        #  x[n-1] = module[n-2](x[n-2])
        #  ...
        #  x[1] = module[0](x[0])
        if self.pipeline[i - 1] is not None:
            self.pipeline[i] = self.module_list[i - 1](self.pipeline[i - 1])
        else:
            self.pipeline[i] = None
    ...

对于输入尚未给出（神经信号还未传到）的module，其输出定为 ``None``。

可以看出，这里给出的仿真器中，每个神经信号在一个仿真步长内只能转移到连接的下一个模块。因此对于一个包含n个模块的模型，需要n个仿真时间步之后才会在最后一层有输出。


快速仿真
------------
然而，很多时候我们希望模型在仿真的第一时间就给出输出，而不是慢慢等信号依次传递到最后一个模块才输出结果。为此，我们增加了一个快速仿真选项。仿真器初始化时 **默认开启** 快速仿真，可以手动设置关闭。

若开启快速仿真，则在仿真器首次运行时，会运行n步而不是1步，并认为这n步的输入都是input_data

.. code-block:: python

    ...
    # 快速仿真开启时，首次运行时跑满pipeline
    # x[0] -> module[0] -> x[1]
    # x[1] -> module[1] -> x[2]
    # ...
    # x[n-1] -> module[n-1] -> x[n]
    if self.simulated_steps == 0 and self.fast:
        for i in range(self.module_list.__len__()):
            # i = 0, 1, ..., n-1
            self.pipeline[i + 1] = self.module_list[i](self.pipeline[i])
    ...
