Wrapper
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: Qiu Haonan

The following types of wrappers are mainly provided in SpikingJelly:

* Functional style :class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` and module style :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>`
* Functional style :class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` and module style :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>`
* Packaging a single step module for single/multi-step propagation :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>`

:class:`multi_step_forward <spikingjelly.activation_based.functional.multi_step_forward>` can be a single step module for multi-step propagation, and \
:class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` can be a single step module packaging step into many modules, such as:

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

For a stateless ANN layer, such as :class:`torch.nn.Conv2d`, which itself requires input data of ``shape = [N, *]``, if used in multi-step mode, it can be wrapped with a multi-step wrapper:

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

But ANN's network layer itself is stateless, there is no pre-order dependence, there is no need for serial computation in time, you can use functional style \
:class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>` or module style \
:class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` for packaging. \
:class:`seq_to_ann_forward <spikingjelly.activation_based.functional.seq_to_ann_forward>`  The ``shape = [T, N, *]`` data is first transformed into ``shape = [TN, *]`` data, and then sent to the stateless network layer for calculation, \ The output is re-transformed to ``shape = [T, N, *]``. Data at different times are computed in parallel, so it is faster:


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

The common network layer is defined in :class:`spikingjelly.activation_based.layer` and is more recommended in :class:`spikingjelly.activation_based.layer`, \
rather than using :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` manual packing, \
although the network layer in :class:`spikingjelly.activation_based.layer` is actually implemented by wrapping the 'forward' function with a wrapper. \The network layer in :class:`spikingjelly.activation_based.layer` has the advantages of:

* Support single step and multi-step mode, and :class:`SeqToANNContainer<spikingjelly.activation_based.layer.SeqToANNContainer>` and :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` packing layer, only supports the multi-step pattern
* The wrapper adds a layer to the ``keys()`` of ``state_dict``, making loading weights difficult

Such as

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



The output is

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


:class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` and :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>` both support multi-step mode only and do not allow switching to single-step mode.\

:class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` is similar to the merged version :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` and :class:`SeqToANNContainer <spikingjelly.activation_based.layer.SeqToANNContainer>`, it can be used to wrap stateless or stateful single-step modules, which need to be specified at packaging time, but this wrapper also supports switching between single-step and multi-step modes.

Example of wrapping a stateless layer:


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

Example of wrapping a stateful layer:


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

It is safe to use :class:`set_step_mode <spikingjelly.activation_based.functional.set_step_mode>` to change :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>`, only the ``step_mode`` of the wrapper itself is changed, the module inside the wrapper remains one-step:

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

If the module itself supports switching between single-step and multi-step modes, is not recommended :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` or :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` on the packaging.\
Because the multi-step forward propagation used by the wrapper may not be as fast as the forward propagation defined by the module itself.


Usually need to use :class:`MultiStepContainer <spikingjelly.activation_based.layer.MultiStepContainer>` or :class:`StepModeContainer <spikingjelly.activation_based.layer.StepModeContainer>` are some of the module does not define multiple steps, such as a network layer that exists in ``torch.nn`` but does not exist in ``spikingjelly.activation_based.layer``.
