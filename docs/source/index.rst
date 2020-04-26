欢迎来到SpikingFlow的文档
-------------------------

`SpikingFlow <https://github.com/fangwei123456/SpikingFlow>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ 的脉冲神经网络(Spiking Neuron Network, SNN)框架。

安装
----------------

注意，SpikingFlow是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装SpikingFlow。

推荐从GitHub下载源代码，因为pip上的版本通常更新慢，bug可能比较多。

从pip安装：

.. code-block:: bash

    pip install SpikingFlow

或者对于开发者，下载源代码，进行代码补充、修改和测试：

.. code-block:: bash

    git clone https://github.com/fangwei123456/SpikingFlow.git


快速上手教程
-------------------------

``SpikingFlow.softbp`` 使用时间驱动仿真SNN，使用反向传播、梯度下降来学习。

``SpikingFlow.event_driven`` 是使用事件驱动仿真SNN，使用反向传播、梯度下降来学习。

而其他的 ``SpikingFlow.*`` 是使用事件驱动仿真SNN，使用生物可解释性的方法（例如STDP）来学习。

因此，``SpikingFlow.softbp`` 和 ``SpikingFlow.event_driven`` 以及其他的 ``SpikingFlow.*`` 包，三者是平行关系，互不交叉。例\
如使用者对 ``SpikingFlow.softbp`` 感兴趣，他只需要阅读 ``SpikingFlow.softbp`` 相关的教程或源代码就可以上手。

* :doc:`神经元 SpikingFlow.neuron<./tutorial.0>`
* :doc:`编码器 SpikingFlow.encoding<./tutorial.1>`
* :doc:`仿真器 SpikingFlow.simulating<./tutorial.2>`
* :doc:`突触连接 SpikingFlow.connection<./tutorial.3>`
* :doc:`学习规则 SpikingFlow.learning<./tutorial.4>`
* :doc:`软反向传播 SpikingFlow.softbp <./tutorial.5>`
* :doc:`事件驱动 SpikingFlow.event_driven <./tutorial.6>`

模块文档
-------------------------

.. toctree::
   :maxdepth: 4

   modules

文档索引
-------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


项目信息
-------------------------
SpikingFlow由北京大学信息科学技术学院数字媒体所媒体学习组 `Multimedia Learning Group <https://pkuml.org/>`_ 开发。

SpikingFlow目前还在起步阶段，因此许多功能还不够完善。
