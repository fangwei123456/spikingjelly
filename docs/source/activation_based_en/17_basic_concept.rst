Basic Conception
=======================================
Author： `fangwei123456 <https://github.com/fangwei123456>`_

Translator: Qiu Haonan

This tutorial introduces ``spikingjelly.activation_based`` It is recommended that all users read some basic concepts of based before using the spikengjelly framework.

Spikingjelly framework is a SNN deep learning framework based on pytorch. Users using spikingjelly framework should first be familiar with the use of pytorch.\
If you don't know much about pytorch, we recommend that you learn the basic tutorial of pytorch first `PyTorch的基础教程 <https://pytorch.org/tutorials/>`_ 。

Representation Based on Activation Value
-------------------------------------------
``spikingjelly.activation_based`` The pulse is represented by a tensor whose value is only 0 or 1, for example：

.. code-block:: python

    import torch

    v = torch.rand([8])
    v_th = 0.5
    spike = (v >= v_th).to(v)
    print('spike =', spike)
    # spike = tensor([0., 0., 0., 1., 1., 0., 1., 0.])

Data Format
-------------------------------------------
In ``spikingjelly.activation_based``, There are two formats for data:

* Data representing a single moment, it‘s ``shape = [N, *]``, where ``N`` is the batch dimension, ``*`` represents any extra dimension.
* Represents data at multiple time, it's ``shape = [T, N, *]``, where ``T`` is the time dimension of the data, ``N`` is the batch dimension, and `*` represents any additional dimension


Step Mode
-------------------------------------------
``spikingjelly.activation_based``, which has two propagation modes, namely single-step mode and multi-step mode. In single-step mode, the data is in the ``shape = [N, *]`` format; In multi-step mode, the data is in ``shape = [T, N, *]`` format.

A module can specify the ``step_mode`` it uses when it is initialized, or it can be modified directly after it is built:

.. code-block:: python
    
    import torch
    from spikingjelly.activation_based import neuron

    net = neuron.IFNode(step_mode='m')
    # 'm' is the multi-step mode
    net.step_mode = 's'
    # 's' is the single-step mode

If we want to enter the sequence data of ``shape = [T, N, *]`` for a step mode module, we usually need to do a time loop manually, split the data into ``T`` ``shape = [N, *]`` data and gradually enter it. Let's create a new layer of IF neurons and set it to single-step mode to input data step by step and get output:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    net_s = neuron.IFNode(step_mode='s')
    T = 4
    N = 1
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W])
    y_seq = []
    for t in range(T):
        x = x_seq[t]  # x.shape = [N, C, H, W]
        y = net_s(x)  # y.shape = [N, C, H, W]
        y_seq.append(y.unsqueeze(0))

    y_seq = torch.cat(y_seq)
    # y_seq.shape = [T, N, C, H, W]

:class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>`  provides encapsulation of ``shape = [T, N, *]`` sequence data into a single step module for progressive forward propagation, it is more convenient to use:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, functional
    net_s = neuron.IFNode(step_mode='s')
    T = 4
    N = 1
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W])
    y_seq = functional.multi_step_forward(x_seq, net_s)
    # y_seq.shape = [T, N, C, H, W]

However, it is actually more convenient to directly set the module as a multi-step module:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    net_m = neuron.IFNode(step_mode='m')
    T = 4
    N = 1
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W])
    y_seq = net_m(x_seq)
    # y_seq.shape = [T, N, C, H, W]

To maintain compatibility with older versions of SpikingJelly code, the default step mode for all stateful modules is single step.

State Saving and Resetting
-------------------------------------------
Neurons and other modules in SNN, similar to RNN, have hidden states and their outputs :math:`Y[t]` not only with the current moment input :math: 'X[t]',  also with one at the end of the state :math:`H[t-1]` related, that is :math:`Y[t] = f(X[t], H[t-1])`.

PyTorch is designed for RNN to print the state as well, can be referred :class:`torch.nn.RNN`. In ``spikingjelly.activation_based``, the state is stored inside the module. For example, we create a new layer of IF neurons, set them to single-step mode, and check the default voltage before and after input:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    net_s = neuron.IFNode(step_mode='s')
    x = torch.rand([4])
    print(net_s)
    print(f'the initial v={net_s.v}')
    y = net_s(x)
    print(f'x={x}')
    print(f'y={y}')
    print(f'v={net_s.v}')

    # outputs are:

    '''
    IFNode(
    v_threshold=1.0, v_reset=0.0, detach_reset=False
    (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    the initial v=0.0
    x=tensor([0.5543, 0.0350, 0.2171, 0.6740])
    y=tensor([0., 0., 0., 0.])
    v=tensor([0.5543, 0.0350, 0.2171, 0.6740])
    '''


After initialization, the ``v`` of the IF neuron layer is set to 0 and is automatically broadcast to ``shape`` after the first input.

If we give a new input, we should first clear the previous state of the neuron and restore it to its initialization state, which can be done by calling the module's ``self.reset()`` function:


.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    net_s = neuron.IFNode(step_mode='s')
    x = torch.rand([4])
    print(f'check point 0: v={net_s.v}')
    y = net_s(x)
    print(f'check point 1: v={net_s.v}')
    net_s.reset()
    print(f'check point 2: v={net_s.v}')
    x = torch.rand([8])
    y = net_s(x)
    print(f'check point 3: v={net_s.v}')

    # outputs are:

    '''
    check point 0: v=0.0
    check point 1: v=tensor([0.9775, 0.6598, 0.7577, 0.2952])
    check point 2: v=0.0
    check point 3: v=tensor([0.8728, 0.9031, 0.2278, 0.5089, 0.1059, 0.0479, 0.5008, 0.8530])
    '''

For convenience, you can also call :class:`spikingjelly.activation_based.functional.reset_net` For convenience, you can also cal.

If the network uses a stateful module, it must be reset after running during training and reasoning:

.. code-block:: python

    from spikingjelly.activation_based import functional
    # ...
    for x, label in tqdm(train_data_loader):
        # ...
        optimizer.zero_grad()
        y = net(x)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()

        functional.reset_net(net)
        # Never forget to reset the network!

If you forget to reset, you may get an error output during reasoning or an error directly during training:

.. code-block:: shell

    RuntimeError: Trying to backward through the graph a second time (or directly access saved variables after they have already been freed). 
    Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). 
    Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved variables after calling backward.

Propagation Mode
-------------------------------------------
If a network consists entirely of single-step modules, the computation order of the entire network is in a progressive propagation mode, for example:

.. code-block:: python

    for t in range(T):
        x = x_seq[t]
        y = net(x)
        y_seq_step_by_step.append(y.unsqueeze(0))

    y_seq_step_by_step = torch.cat(y_seq_step_by_step, 0)

If the network consists entirely of multi-step modules, the computation order of the entire network is carried out in a layer-by-layer propagation mode, for example:

.. code-block:: python 

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, functional, layer
    T = 4
    N = 2
    C = 8
    x_seq = torch.rand([T, N, C]) * 64.

    net = nn.Sequential(
        layer.Linear(C, 4),
        neuron.IFNode(),
        layer.Linear(4, 2),
        neuron.IFNode()
    )

    functional.set_step_mode(net, step_mode='m')
    with torch.no_grad():
        y_seq_layer_by_layer = x_seq
        for i in range(net.__len__()):
            y_seq_layer_by_layer = net[i](y_seq_layer_by_layer)

In most cases we don't need an explicit implementation ``for i in range(net.__len__())`` this cycle, because :class:`torch.nn.Sequential` has already done that for us,\
so we can actually do that：

.. code-block:: python 
    
    y_seq_layer_by_layer = net(x_seq)

Step by step propagation and layer by layer propagation, in fact, only the calculation order is different, their calculation results are exactly the same:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, functional, layer
    T = 4
    N = 2
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W]) * 64.

    net = nn.Sequential(
    layer.Conv2d(3, 8, kernel_size=3, padding=1, stride=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    neuron.IFNode(),
    layer.Flatten(start_dim=1),
    layer.Linear(8 * H // 2 * W // 2, 10),
    neuron.IFNode(),
    )

    print(f'net={net}')

    with torch.no_grad():
        y_seq_step_by_step = []
        for t in range(T):
            x = x_seq[t]
            y = net(x)
            y_seq_step_by_step.append(y.unsqueeze(0))

        y_seq_step_by_step = torch.cat(y_seq_step_by_step, 0)
        # we can also use `y_seq_step_by_step = functional.multi_step_forward(x_seq, net)` to get the same results

        print(f'y_seq_step_by_step=\n{y_seq_step_by_step}')

        functional.reset_net(net)
        functional.set_step_mode(net, step_mode='m')
        y_seq_layer_by_layer = net(x_seq)

        max_error = (y_seq_layer_by_layer - y_seq_step_by_step).abs().max()
        print(f'max_error={max_error}')

The output of the above code is:

.. code-block:: shell

    net=Sequential(
    (0): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
    (1): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, step_mode=s)
    (3): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    (4): Flatten(start_dim=1, end_dim=-1, step_mode=s)
    (5): Linear(in_features=128, out_features=10, bias=True)
    (6): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    y_seq_step_by_step=
    tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

            [[0., 1., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]],

            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 0., 0., 1., 0., 0., 0.]],

            [[0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]]])
    max_error=0.0

The following figure shows the sequence of progressively propagated build computations：


.. image:: ../_static/tutorials/activation_based/basic_concept/step-by-step.png
    :width: 100%


The following image shows the order in which the computation graph is constructed layer by layer：

.. image:: ../_static/tutorials/activation_based/basic_concept/layer-by-layer.png
    :width: 100%


The calculation graph of SNN has two dimensions, namely, time steps and network depth. Network propagation is actually the process of generating a complete calculation graph, as shown in the two pictures above. In fact, stepwise propagation is depth-first traversal, while layer-by-layer propagation is breadth-first traversal.

Although the difference is only in the order of computation, there are slight differences in computation speed and memory consumption.\

* When using gradient substitution method for training, it is usually recommended to use layer by layer propagation. When the network is built correctly, the parallelism of layer by layer propagation is greater and the speed is faster
* Use step-by-step propagation when memory is limited. For example, a very large ``T`` is required in the ANN2SNN task. In the layer by layer propagation mode, for stateless layers, the real batch size is ``TN`` rather than ``N``. when ``T`` is too large, the memory consumption is very high
