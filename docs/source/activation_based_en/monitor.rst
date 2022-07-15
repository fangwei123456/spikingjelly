Monitor
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

:class:`spikingjelly.activation_based.monitor` has defined some commonly used monitors, with which the users can record \
the data that they are interested in. Now let us try these monitors.


Usage
-------------------------------------------
All monitors have similar usage. Let us take :class:`spikingjelly.activation_based.monitor.OutputMonitor` as the example.

Firstly, let us build a simple single-step network. To avoid no spikes, we set all weights to be positive:

.. code-block:: python

    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode)
    T = 4
    N = 1
    x_seq = torch.rand([T, N, 8])

    with torch.no_grad():
        net(x_seq)

The recorded data will be stored in ``.records`` whose type is ``list``. The data are recorded by the order in how they are created:

.. code-block:: python

    print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')

The outputs are:

.. code-block:: shell

    spike_seq_monitor.records=
    [tensor([[[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]],

            [[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]]]), tensor([[[0., 0.]],

            [[1., 0.]],

            [[0., 1.]],

            [[1., 0.]]])]


We can also use the index to get the ``i``-th data:

.. code-block:: python

    print(f'spike_seq_monitor[0]={spike_seq_monitor[0]}')

The outputs are:

.. code-block:: shell

    spike_seq_monitor[0]=tensor([[[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]],

            [[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]]])
    

The names of monitored layers are stored in ``.monitored_layers``:

.. code-block:: python

    print(f'net={net}')
    print(f'spike_seq_monitor.monitored_layers={spike_seq_monitor.monitored_layers}')

The outputs are:

.. code-block:: shell

    net=Sequential(
    (0): Linear(in_features=8, out_features=4, bias=True)
    (1): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    (2): Linear(in_features=4, out_features=2, bias=True)
    (3): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    spike_seq_monitor.monitored_layers=['1', '3']

We can also use the name as the index to get the recorded data of the layer, which are stored in a ``list``:

.. code-block:: python

    print(f"spike_seq_monitor['1']={spike_seq_monitor['1']}")

The outputs are:

.. code-block:: shell

    spike_seq_monitor['1']=[tensor([[[0., 0., 0., 0.]],

        [[1., 1., 1., 1.]],

        [[0., 0., 0., 0.]],

        [[1., 1., 1., 1.]]])]


We can call ``.clear_recorded_data()`` to clear the recorded data:

.. code-block:: python

    spike_seq_monitor.clear_recorded_data()
    print(f'spike_seq_monitor.records={spike_seq_monitor.records}')
    print(f"spike_seq_monitor['1']={spike_seq_monitor['1']}")

The outputs are:

.. code-block:: shell

    spike_seq_monitor.records=[]
    spike_seq_monitor['1']=[]

All ``monitor`` will remove hooks when they are deleted. However, python will not guarantee to call the ``__del__()`` function of the monitor even if we call ``del a_monitor`` manually:

.. code-block:: python

    del spike_seq_monitor
    # hooks may still work

Instead, we should call ``remove_hooks`` to remove all hooks:

.. code-block:: python

    spike_seq_monitor.remove_hooks()


:class:`OutputMonitor <spikingjelly.activation_based.monitor.OutputMonitor>` can also process the data when recording, which is implemented by ``function_on_output``. \
The default value of ``function_on_output`` is ``lambda x: x``, which means record the origin data. If we want to record the firing rates, we can define the \
function of calculating the firing rates:

.. code-block:: python

    def cal_firing_rate(s_seq: torch.Tensor):
        # s_seq.shape = [T, N, *]
        return s_seq.flatten(1).mean(1)

Then, we can set this function as ``function_on_output`` to get a firing rates monitor:

.. code-block:: python

    fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)

``.disable()`` can pause ``monitor``, and ``.enable()`` can restart ``monitor``:

.. code-block:: python

    with torch.no_grad():
        fr_monitor.disable()
        net(x_seq)
        functional.reset_net(net)
        print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')

        fr_monitor.enable()
        net(x_seq)
        print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
        functional.reset_net(net)
        del fr_monitor

The outputs are:

.. code-block:: shell

    after call fr_monitor.disable(), fr_monitor.records=
    []
    after call fr_monitor.enable(), fr_monitor.records=
    [tensor([0.0000, 1.0000, 0.5000, 1.0000]), tensor([0., 1., 0., 1.])]

Record Attributes
-------------------------------------------

To record the attributes of some modules, e.g., the membrane potential, we can use :class:`spikingjelly.activation_based.monitor.AttributeMonitor`.

``store_v_seq: bool = False`` is the default arg in ``__init__`` of spiking neurons, which means only ``v`` at the last time-step will be stored, \
and ``v_seq`` at each time-step will not be sotred. To record all :math:`V[t]`, we set  ``store_v_seq = True``:

.. code-block:: python

    for m in net.modules():
        if isinstance(m, neuron.IFNode):
            m.store_v_seq = True

Then, we use :class:`spikingjelly.activation_based.monitor.AttributeMonitor` to record: 

.. code-block:: python

    v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=net, instance=neuron.IFNode)
    with torch.no_grad():
        net(x_seq)
        print(f'v_seq_monitor.records=\n{v_seq_monitor.records}')
        functional.reset_net(net)
        del v_seq_monitor

The outputs are:

.. code-block:: shell

    v_seq_monitor.records=
    [tensor([[[0.8102, 0.8677, 0.8153, 0.9200]],

            [[0.0000, 0.0000, 0.0000, 0.0000]],

            [[0.0000, 0.8129, 0.0000, 0.9263]],

            [[0.0000, 0.0000, 0.0000, 0.0000]]]), tensor([[[0.2480, 0.4848]],

            [[0.0000, 0.0000]],

            [[0.8546, 0.6674]],

            [[0.0000, 0.0000]]])]

Record Inputs
-------------------------------------------
To record inputs, we can use :class:`spikingjelly.activation_based.monitor.InputMonitor`, which is similar to :class:`spikingjelly.activation_based.monitor.OutputMonitor`:

.. code-block:: python

    input_monitor = monitor.InputMonitor(net, neuron.IFNode)
    with torch.no_grad():
        net(x_seq)
        print(f'input_monitor.records=\n{input_monitor.records}')
        functional.reset_net(net)
        del input_monitor

The outputs are:

.. code-block:: shell

    input_monitor.records=
    [tensor([[[1.1710, 0.7936, 0.9325, 0.8227]],

            [[1.4373, 0.7645, 1.2167, 1.3342]],

            [[1.6011, 0.9850, 1.2648, 1.2650]],

            [[0.9322, 0.6143, 0.7481, 0.9770]]]), tensor([[[0.8072, 0.7733]],

            [[1.1186, 1.2176]],

            [[1.0576, 1.0153]],

            [[0.4966, 0.6030]]])]

Record the Input Gradients :math:`\frac{\partial L}{\partial Y}`
--------------------------------------------------------------------------------------
We can use :class:`spikingjelly.activation_based.monitor.GradOutputMonitor` to record the input gradients :math:`\frac{\partial L}{\partial S}` of each module:

.. code-block:: python

    spike_seq_grad_monitor = monitor.GradOutputMonitor(net, neuron.IFNode)
    net(x_seq).sum().backward()
    print(f'spike_seq_grad_monitor.records=\n{spike_seq_grad_monitor.records}')
    functional.reset_net(net)
    del spike_seq_grad_monitor

The outputs are:

.. code-block:: python

    spike_seq_grad_monitor.records=
    [tensor([[[1., 1.]],

            [[1., 1.]],

            [[1., 1.]],

            [[1., 1.]]]), tensor([[[ 0.0803,  0.0383,  0.1035,  0.1177]],

            [[-0.1013, -0.1346, -0.0561, -0.0085]],

            [[ 0.5364,  0.6285,  0.3696,  0.1818]],

            [[ 0.3704,  0.4747,  0.2201,  0.0596]]])]

Note that the input gradients of the last layer's output spikes are all ``1`` because we use ``.sum().backward()``.

Record the Output Gradients :math:`\frac{\partial L}{\partial X}`
--------------------------------------------------------------------------------------
We can use :class:`spikingjelly.activation_based.monitor.GradInputMonitor` to record the output gradients :math:`\frac{\partial L}{\partial X}` of each module.

Let us build a deep SNN, tune ``alpha`` for surrogate functions, and compare the effect:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import monitor, neuron, functional, layer, surrogate

    net = []
    for i in range(10):
        net.append(layer.Linear(8, 8))
        net.append(neuron.IFNode())

    net = nn.Sequential(*net)

    functional.set_step_mode(net, 'm')

    T = 4
    N = 1
    x_seq = torch.rand([T, N, 8])

    input_grad_monitor = monitor.GradInputMonitor(net, neuron.IFNode, function_on_grad_input=torch.norm)

    for alpha in [0.1, 0.5, 2, 4, 8]:
        for m in net.modules():
            if isinstance(m, surrogate.Sigmoid):
                m.alpha = alpha
        net(x_seq).sum().backward()
        print(f'alpha={alpha}, input_grad_monitor.records=\n{input_grad_monitor.records}\n')
        functional.reset_net(net)
        # zero grad
        for param in net.parameters():
            param.grad.zero_()

        input_grad_monitor.records.clear()


The outputs are:

.. code-block:: shell

    alpha=0.1, input_grad_monitor.records=
    [tensor(0.3868), tensor(0.0138), tensor(0.0003), tensor(9.1888e-06), tensor(1.0164e-07), tensor(1.9384e-09), tensor(4.0199e-11), tensor(8.6942e-13), tensor(1.3389e-14), tensor(2.7714e-16)]

    alpha=0.5, input_grad_monitor.records=
    [tensor(1.7575), tensor(0.2979), tensor(0.0344), tensor(0.0045), tensor(0.0002), tensor(1.5708e-05), tensor(1.6167e-06), tensor(1.6107e-07), tensor(1.1618e-08), tensor(1.1097e-09)]

    alpha=2, input_grad_monitor.records=
    [tensor(3.3033), tensor(1.2917), tensor(0.4673), tensor(0.1134), tensor(0.0238), tensor(0.0040), tensor(0.0008), tensor(0.0001), tensor(2.5466e-05), tensor(3.9537e-06)]

    alpha=4, input_grad_monitor.records=
    [tensor(3.5353), tensor(1.6377), tensor(0.7076), tensor(0.2143), tensor(0.0369), tensor(0.0069), tensor(0.0026), tensor(0.0006), tensor(0.0003), tensor(8.5736e-05)]

    alpha=8, input_grad_monitor.records=
    [tensor(4.3944), tensor(2.4396), tensor(0.8996), tensor(0.4376), tensor(0.0640), tensor(0.0122), tensor(0.0053), tensor(0.0016), tensor(0.0013), tensor(0.0005)]
