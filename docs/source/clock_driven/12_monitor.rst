监视器
======

本教程作者：\ `Yanqi-Chen <https://github.com/Yanqi-Chen>`__

本节教程将介绍如何使用监视器监视网络状态与高阶统计数据。

监视网络脉冲
------------

首先我们搭建一个简单的两层网络

.. code:: python

   import torch
   from torch import nn
   from spikingjelly.clock_driven import neuron

   net = nn.Sequential(
                   nn.Linear(784, 100, bias=False),
                   neuron.IFNode(),
                   nn.Linear(100, 10, bias=False),
                   neuron.IFNode()
               )           

在网络运行之前，我们先创建一个监视器。注意监视器除去网络之外，还有\ ``device``\ 和\ ``backend``\ 两个参数，可以指定用numpy数组或者PyTorch张量记录结果并计算。此处我们用PyTorch后端，CPU进行处理。

.. code:: python

   from spikingjelly.clock_driven.monitor import Monitor
   mon = Monitor(net, device='cpu', backend='torch')

这样就将一个网络与监视器绑定了起来。但是此时监视功能还处于默认的禁用模式，因此在开始记录之前需要手动启用监视功能：

.. code:: python

   mon.enable()

给网络以随机的输入\ :math:`X\sim\mathcal{U}(1,3)`

.. code:: python

   neuron_num = 784
   T = 8
   batch_size = 3 
   x = torch.rand([T, batch_size, neuron_num]) * 2 + 1
   x = x.cuda()

   for t in range(T):
       net(x[t])

运行结束之后，可以通过监视器获得网络各层神经元的输出脉冲原始数据。Monitor的\ ``s``\ 成员记录了脉冲，为一个以网络层名称为键的字典，每个键对应的的值为一个长为\ ``T``\ 的列表，列表中的元素是尺寸为\ ``[batch_size, ...(神经元尺寸)]``\ 形状的张量。

在使用结束之后，如果需要清空数据进行下一轮记录，需要使用\ ``reset``\ 方法清空已经记录的数据。

.. code:: python

   mon.reset()

如果不再需要监视器记录数据（如仅在测试时记录，训练时不记录），可调用\ ``disable``\ 方法暂停记录。

.. code:: python

   mon.disable()

暂停后监视器仍然与网络绑定，可在下次需要记录时通过\ ``enable``\ 方法重新启用。

监视多步网络
------------

监视器同样支持多步模块，在使用上没有区别

.. code:: python

   import torch
   from torch import nn
   from spikingjelly.cext import neuron as cext_neuron
   from spikingjelly.clock_driven import layer

   neuron_num = 784
   T = 8
   batch_size = 3 
   x = torch.rand([T, batch_size, neuron_num]) * 2 + 1
   x = x.cuda()

   net = nn.Sequential(
                   layer.SeqToANNContainer(
                       nn.Linear(784, 100, bias=False)
                   ),
                   cext_neuron.MultiStepIFNode(alpha=2.0),
                   layer.SeqToANNContainer(
                       nn.Linear(100, 10, bias=False)
                   ),
                   cext_neuron.MultiStepIFNode(alpha=2.0),
               )

   mon = Monitor(net, 'cpu', 'torch')
   mon.enable()
   net(x)

高阶统计数据
------------

目前，监视器支持计算神经元层的\ **平均发放率**\ 与\ **未发放神经元占比**\ 两个指标。使用者既可以指定某一层的名称（按照PyTorch的模块名称字符串）也可以指定所有层的数据。以对前述的单步网络计算平均发放率为例：

指定参数\ ``all=True``\ 为时，记录所有层的平均发放率：

.. code:: python

   print(mon.get_avg_firing_rate(all=True)) # tensor(0.2924)

也可以具体到记录某一层：

.. code:: python

   print(mon.get_avg_firing_rate(all=False, module_name='1')) # tensor(0.3183)
   print(mon.get_avg_firing_rate(all=False, module_name='3')) # tensor(0.0333)
