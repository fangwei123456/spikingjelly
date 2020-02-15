欢迎来到SpikingFlow的文档
=======================================
SpikingFlow是一个基于 `PyTorch <https://pytorch.org/>`_ 的脉冲神经网络(Spiking Neuron Network, SNN)框架。

各个模块的文档如下所示：

.. toctree::
   :maxdepth: 2

   modules

目录
==================
* :doc:`快速上手教程 <./quickstart>`

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



项目信息
=======================================
SpikingFlow由北京大学信息科学技术学院数字媒体所媒体学习组 `Multimedia Learning Group <https://pkuml.org/>`_ 开发。

SpikingFlow目前还在起步阶段，并不适合真正地去使用。

开发规范
=======================================
1. 所有的模块都应该继承自它的基类，例如实现1给新的神经元模型，则应该继承neuron中的BaseNode

2. 信息流过每个模块，需要消耗1个dt的时间。因而1个直线排列、长度为L的神经网络，在输入数据后，经过
   L个dt才能得到第1个输出

3. 突触传递的是电流

4. 神经元输出的是脉冲

5. 脉冲的数据类型为torch.bool
