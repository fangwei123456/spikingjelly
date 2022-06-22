Migrate From Old Versions
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

There is some difference between the old and new versions of SpikingJelly. We recommend the users read this \
tutorial if they are familiar with the old version and want to try the new version. SpikingJelly has nice compatibility \
for the old version, and the users do not need to do much change to their codes to Migrate from the old version to the new version.

We also recommend that the users read the tutorial :doc:`../activation_based_en/basic_concept`

The old version of SpikingJelly means the version number ``<=0.0.0.0.12``.

Rename of Packages
-------------------------------------------
In the new version, SpikingJelly renames some sub-packages, which are:

===============  ==================
Old              New            
===============  ==================
clock_driven     activation_based
event_driven     timing_based    
===============  ==================

Step Mode and Propagation Patterns
-------------------------------------------
All modules in the old version (``<=0.0.0.0.12``) of SpikingJelly are the single-step modules by default, except for the module that has the prefix ``MultiStep``.\

The new version of SpikingJelly does not use the prefix to distinguish the single/multi-step module. Now the step mode is controlled by the module itself, which is \
the attribute ``step_mode``. Refer to :doc:`../activation_based_en/basic_concept` for more details.

Hence, there is no multi-step module defined additionally in the new version of SpikingJelly. Now one module can be both the single-step module and the multi-step module, which is determined by ``step_mode`` is ``'s'`` or ``'m'``.\
In the old version of SpikingJelly, if we want to use the LIF neuron with single-step, we write codes as:

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.LIFNode()

In the new version of SpikingJelly, all modules are single-step modules by default. We write codes similar to the old version, except we replace ``clock_driven``with ``activation_based``: 

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode()

In the old version of SpikingJelly, if we want to use the LIF neuron with multi-step, we should write codes as:

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.MultiStepLIFNode()

In the new version of SpikingJelly, one module can use both single-step and multi-step. We can use the LIF neuron with multi-step easily by setting ``step_mode='m'``:

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode(step_mode='m')


In the old version of SpikingJelly, we use the step-by-step or layer-by-layer propagation patterns as the following codes:

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


In the new version of SpikingJelly, we can use :class:`spikingjelly.activation_based.functional.set_step_mode` to change the step mode of all modules in the whole network.\
If all modules use single-step, the network can use a step-by-step propagation pattern; if all modules use multi-step, the network can use a layer-by-layer propagation pattern:

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