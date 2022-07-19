从老版本迁移
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

新版的SpikingJelly改动较大，使用老版本SpikingJelly的用户若想迁移到新版本，则需要阅读此教程。\
SpikingJelly的版本升级尽可能前向兼容，因此用户无需做出太多代码上的修改，即可轻松迁移到新版本。

推荐老版本用户也阅读新版本的教程 :doc:`../activation_based/basic_concept`。

“老版本SpikingJelly”均指的是版本号 ``<=0.0.0.0.12`` 的SpikingJelly。

子包重命名
-------------------------------------------
新版的SpikingJelly对子包进行了重命名，与老版本的对应关系为：

===============  ==================
老版本            新版本             
===============  ==================
clock_driven     activation_based
event_driven     timing_based    
===============  ==================

单步多步模块和传播模式
-------------------------------------------
``<=0.0.0.0.12`` 的老版本SpikingJelly，在默认情况下所有模块都是单步的，除非其名称含有前缀 ``MultiStep``。\
而新版的SpikingJelly，则不再使用前缀对单步和多步模块进行区分，取而代之的是同一个模块，拥有单步和多步两种步进模式，\
使用 ``step_mode`` 进行控制。具体信息可以参见 :doc:`../activation_based/basic_concept`。

因而在新版本中不再有单独的多步模块，取而代之的则是融合了单步和多步的统一模块。例如，在老版本的SpikingJelly中，若想使用单步LIF神经元，\
是按照如下方式：

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.LIFNode()

在新版本中，所有模块默认是单步的，所以与老版本的代码几乎相同，除了将 ``clock_driven`` 换成了 ``activation_based``：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode()

在老版本的SpikingJelly中，若想使用多步LIF神经元，是按照如下方式：

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.MultiStepLIFNode()

在新版本中，单步和多步模块进行了统一，因此只需要指定为多步模块即可：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode(step_mode='m')


在老版本中，若想分别搭建一个逐步传播和逐层传播的网络，按照如下方式：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.clock_driven import neuron, layer, functional

    with torch.no_grad():

        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])

        # step-by-step
        net_sbs = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            neuron.IFNode()
        )
        y_seq = functional.multi_step_forward(x_seq, net_sbs)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net_sbs)



        # layer-by-layer
        net_lbl = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C),
            ),
            neuron.MultiStepIFNode()
        )
        y_seq = net_lbl(x_seq)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net_lbl)


而在新版本中，由于单步和多步模块已经融合，可以通过 :class:`spikingjelly.activation_based.functional.set_step_mode` 对整个网络的步进模式进行转换。\
在所有模块使用单步模式时，整个网络就可以使用逐步传播；所有模块都使用多步模式时，整个网络就可以使用逐层传播：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, functional

    with torch.no_grad():

        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])

        # the network uses step-by-step because step_mode='s' is the default value for all modules
        net = nn.Sequential(
            layer.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(C),
            neuron.IFNode()
        )
        y_seq = functional.multi_step_forward(x_seq, net)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net)

        # set the network to use layer-by-layer
        functional.set_step_mode(net, step_mode='m')
        y_seq = net(x_seq)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net)