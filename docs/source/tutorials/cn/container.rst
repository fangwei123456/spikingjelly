包装器
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

SpikingJelly中主要提供了如下几种包装器：

* 函数风格的 :class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` 和模块风格的 :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>`
* 函数风格的 :class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` 和模块风格的 :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>`
* 对单步模块进行包装以进行单步/多步传播的 :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>`

:class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` 可以将一个单步模块进行多步传播，而 \
:class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 则可以将一个单步模块包装成多步模块，例如：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, functional, layer

    net_s = neuron.IFNode(step_mode='s')
    T = 4
    N = 1
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W])
    y_seq = functional.multi_step_forward(x_seq, net_s)
    # y_seq.shape = [T, N, C, H, W]

    net_s.reset()
    net_m = layer.MultiStepContainer(net_s)
    z_seq = net_m(x_seq)
    # z_seq.shape = [T, N, C, H, W]

    # z_seq is identical to y_seq

对于无状态的ANN网络层，例如 :class:`torch.nn.Conv2d`，其本身要求输入数据的 ``shape = [N, *]``，若用于多步模式，则可以用多步的包装器进行包装：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer

    with torch.no_grad():
        T = 4
        N = 1
        C = 3
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])
        
        conv = nn.Conv2d(C, 8, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)
        
        y_seq = functional.multi_step_forward(x_seq, (conv, bn))
        # y_seq.shape = [T, N, 8, H, W]
        
        net = layer.MultiStepContainer(conv, bn)
        z_seq = net(x_seq)
        # z_seq.shape = [T, N, 8, H, W]
        
        # z_seq is identical to y_seq

但是ANN的网络层本身是无状态的，不存在前序依赖，没有必要在时间上串行的计算，可以使用函数风格的 \
:class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` 或模块风格的 \
:class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` 进行包装。\
:class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` 将 \
``shape = [T, N, *]`` 的数据首先变换为 ``shape = [TN, *]``，再送入无状态的网络层进行计算，\
输出的结果会被重新变换为 ``shape = [T, N, *]``。不同时刻的数据是并行计算的，因而速度更快：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer

    with torch.no_grad():
        T = 4
        N = 1
        C = 3
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])

        conv = nn.Conv2d(C, 8, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)

        y_seq = functional.multi_step_forward(x_seq, (conv, bn))
        # y_seq.shape = [T, N, 8, H, W]

        net = layer.MultiStepContainer(conv, bn)
        z_seq = net(x_seq)
        # z_seq.shape = [T, N, 8, H, W]

        # z_seq is identical to y_seq
        
        p_seq = functional.seq_to_ann_forward(x_seq, (conv, bn))
        # p_seq.shape = [T, N, 8, H, W]

        net = layer.SeqToANNContainer(conv, bn)
        q_seq = net(x_seq)
        # q_seq.shape = [T, N, 8, H, W]

        # q_seq is identical to p_seq, and also identical to y_seq and z_seq


常用的网络层，在 :class:`spikingjelly.activation_based.layer` 已经定义过，更推荐使用 :class:`spikingjelly.activation_based.layer` 中的网络层，\
而不是使用 :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` 手动包装，\
尽管 :class:`spikingjelly.activation_based.layer` 中的网络层实际上就是用包装器包装 `forward` 函数实现的。\
:class:`spikingjelly.activation_based.layer` 中的网络层，优势在于：

* 支持单步和多步模式，而 :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` 和 :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 包装的层，只支持多步模式
* 包装器会使得 ``state_dict`` 的 ``keys()`` 也增加一层包装，给加载权重带来麻烦
  
例如

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer, neuron


    ann = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU()
    )

    print(f'ann.state_dict.keys()={ann.state_dict().keys()}')

    net_container = nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
        ),
        neuron.IFNode(step_mode='m')
    )
    print(f'net_container.state_dict.keys()={net_container.state_dict().keys()}')

    net_origin = nn.Sequential(
        layer.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        neuron.IFNode(step_mode='m')
    )
    print(f'net_origin.state_dict.keys()={net_origin.state_dict().keys()}')

    try:
        print('net_container is trying to load state dict from ann...')
        net_container.load_state_dict(ann.state_dict())
        print('Load success!')
    except BaseException as e:
        print('net_container can not load! The error message is\n', e)

    try:
        print('net_origin is trying to load state dict from ann...')
        net_origin.load_state_dict(ann.state_dict())
        print('Load success!')
    except BaseException as e:
        print('net_origin can not load! The error message is', e)



输出为

.. code-block:: shell

    ann.state_dict.keys()=odict_keys(['0.weight', '1.weight', '1.bias', '1.running_mean', '1.running_var', '1.num_batches_tracked'])
    net_container.state_dict.keys()=odict_keys(['0.0.weight', '0.1.weight', '0.1.bias', '0.1.running_mean', '0.1.running_var', '0.1.num_batches_tracked'])
    net_origin.state_dict.keys()=odict_keys(['0.weight', '1.weight', '1.bias', '1.running_mean', '1.running_var', '1.num_batches_tracked'])
    net_container is trying to load state dict from ann...
    net_container can not load! The error message is
    Error(s) in loading state_dict for Sequential:
        Missing key(s) in state_dict: "0.0.weight", "0.1.weight", "0.1.bias", "0.1.running_mean", "0.1.running_var". 
        Unexpected key(s) in state_dict: "0.weight", "1.weight", "1.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked". 
    net_origin is trying to load state dict from ann...
    Load success!


:class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 和 :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` 都是只支持多步模式的，不允许切换为单步模式。\

:class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` 类似于融合版的 :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 和 :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>`，\
可以用于包装无状态或有状态的单步模块，需要在包装时指明是否有状态，但此包装器还支持切换单步和多步模式。


包装无状态层的示例：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, layer


    with torch.no_grad():
        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])
        net = layer.StepModeContainer(
            False,
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        net.step_mode = 'm'
        y_seq = net(x_seq)
        # y_seq.shape = [T, N, C, H, W]

        net.step_mode = 's'
        y = net(x_seq[0])
        # y.shape = [N, C, H, W]

包装有状态层的示例：


.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, layer, functional


    with torch.no_grad():
        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])
        net = layer.StepModeContainer(
            True,
            neuron.IFNode()
        )
        net.step_mode = 'm'
        y_seq = net(x_seq)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net)

        net.step_mode = 's'
        y = net(x_seq[0])
        # y.shape = [N, C, H, W]
        functional.reset_net(net)

使用 :class:`set_step_mode <spikingjelly.activation_based.functional.set_step_mode>` 改变 :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` 是安全的，只会改变包装器本身的 ``step_mode``，\
而包装器内的模块仍然保持单步：

.. code-block:: python
    
    import torch
    from spikingjelly.activation_based import neuron, layer, functional


    with torch.no_grad():
        net = layer.StepModeContainer(
            True,
            neuron.IFNode()
        )
        functional.set_step_mode(net, 'm')
        print(f'net.step_mode={net.step_mode}')
        print(f'net[0].step_mode={net[0].step_mode}')

如果模块本身就支持单步和多步模式的切换，则不推荐使用 :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 或 :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` 对其进行包装。\
因为包装器使用的多步前向传播，可能不如模块自身定义的前向传播速度快。


通常需要用到 :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` 或 :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` 的\
是一些没有定义多步的模块，例如一个在 ``torch.nn`` 中存在，但在 ``spikingjelly.activation_based.layer`` 中不存在的网络层。