Container
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `Qiu Haonan <https://github.com/Maybe2022>`_, `fangwei123456 <https://github.com/fangwei123456>`_

The major containers in SpikingJelly are: 

* :class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` in functional style and :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` in module style
* :class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` in functional style and :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` in module style
* :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` for wrapping a single-step module for single/multi-step propagation

:class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` can use a single-step module to implement multi-step propagation, \
and :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` can wrap a single-step module to a multi-step module. For example:

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

For a stateless ANN layer such as :class:`torch.nn.Conv2d`, which requires input data with ``shape = [N, *]``, to be used in multi-step mode, we can wrap it by the multi-step containers:

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

However, the ANN layers are stateless and :math:`Y[t]` is only determined by :math:`X[t]`. Hence, it is not necessary to calculate :math:`Y[t]` step-bt-step.\
We can use :class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` or :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` to wrap, \
which will reshape the input with ``shape = [T, N, *]`` to  ``shape = [TN, *]``, send data to ann layers, and reshape output to ``shape = [T, N, *]``. The calculation in different time-steps are in parallelism and faster:

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

Most frequently-used ann modules have been defined in :class:`spikingjelly.activation_based.layer`. It is recommended to use modules in :class:`spikingjelly.activation_based.layer`, \
rather than using a container to wrap the ann layers manually. Althouth the modules in :class:`spikingjelly.activation_based.layer` are implementd by using :class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` to \
wrap forward function, the advantages of modules in :class:`spikingjelly.activation_based.layer` are:

* Both single-step and multi-step modes are supported. When using :class:`SeqToANNContainer<spikingjelly.activation_based.layer.SeqToANNContainer>` or :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` to wrap modules, only the multi-step mode is supported.
* The wrapping of containers will add a prefix of ``keys()`` of ``state_dict``, which brings some troubles for loading weights.

For example:

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



The outputs are

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


:class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` and :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` only support for multi-step mode and do not allow to switch to single-step mode.

:class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` works like the merged version of :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` and :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>`, which can be used to wrap stateless or stateful single-step modules.\
The user should specify whether the wrapped modules are stateless or stateful when using this container. This container also supports switching step modes.

Here is an example of wrapping a stateless layer:


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

Here is an example of wrapping a stateful layer:


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

It is safe to use :class:`set_step_mode <spikingjelly.activation_based.functional.set_step_mode>` to change the step mode of :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>`. Only the ``step_mode`` of the container itself is changed, and the modules inside the container still use single-step:

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

If the module itself supports for switching between single-step and multi-step modes, is not recommended to use :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` or :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` to wrap.\
Because the multi-step forward implemented by the container may not be as fast as the forward defined by the module itself.


In most cases, we use :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` or :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` to wrap modules which do not define the multi-step forward, such as a network layer that exists in ``torch.nn`` but does not exist in ``spikingjelly.activation_based.layer``.
