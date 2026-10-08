Migrate From Old Versions
=======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/migrate_from_legacy`

This page separates V2 interface migration from the historical namespace
migration for ``<=0.0.0.0.12``. V2 includes breaking changes; update configuration
explicitly rather than expecting implicit compatibility.

V2: automatic execution and interface migration
-----------------------------------------------

.. list-table:: Previous usage and current usage
    :header-rows: 1
    :widths: 40 60

    * - Previous usage
      - Current usage
    * - Neuron ``backend=`` or assignment to ``.backend``
      - Remove configuration; move modules and inputs to the same device
    * - ``functional.set_backend`` or ``supported_backends``
      - Remove calls; use ``functional.neuron_implementation`` for diagnostics
    * - Backend-specific functional functions
      - Use public ``if_step``, ``lif_step`` or ``*_multi_step``; check arguments/results
    * - Private imports from ``cuda_kernel/`` or ``triton_kernel/``
      - Use public neuron, functional or precision APIs, not ``spikingjelly._ops``
    * - Experimental IF/LIF/PLIF classes
      - Use ``IFNode``, ``LIFNode`` and ``ParametricLIFNode``
    * - Auto CUDA and retired code/inference-graph generators
      - Use ``FlexSN`` for custom dynamics; stop maintaining generated legacy kernels
    * - ``FlexSNKernel`` or ``FlexSN.kernel``
      - Use ``FlexSN.functional_forward`` with explicit states/static inputs
    * - ``SpikeLinear``, ``SpikeConv*``, ``spike_linear`` or ``spike_conv*``
      - Ordinary Linear/Conv plus memopt; retained fused/packed/sparse projections for specific algorithms
    * - CuPy dependencies and ``cupy11``/``cupy12`` extras
      - Remove them; install Triton or build optional native CUDA extensions

Old code (cannot run with the current version):

.. code-block:: text

    neuron.LIFNode(step_mode="m", backend="cupy")
    functional.set_backend(net, "triton")

Current standalone example:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, functional

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node = neuron.LIFNode(step_mode="m", store_v_seq=True).to(device)
    x = torch.rand(4, 2, 8, device=device, requires_grad=True)
    node(x).sum().backward()
    functional.reset_net(node)

IF/LIF/PLIF sequence functional interfaces take explicit initial state and return spikes,
final state and an optional trace. For example:

.. code-block:: python

    spikes, v_final, v_seq = functional.lif_multi_step(
        x, torch.zeros_like(x[0]), tau=2.0, store_v_seq=True
    )

Single-step ``lif_step`` returns ``(spike, v_next)``; ``lif_multi_step`` returns
three values. Removing a function-name suffix mechanically is insufficient;
check the signature in :doc:`./neuron` and the public API.

Normal usage needs no implementation choice. See :doc:`./triton_backend` for
installation/diagnostics, :doc:`./precision` for policies, :doc:`./flexsn` for
custom dynamics and :doc:`./memopt` for memory optimization/projections.
There is no automatic migration script or promise that an old whole-module
pickle/checkpoint loads directly. Prefer trusted ``state_dict`` files and check
keys/shapes against the current model definition.

Historical migration: <=0.0.0.0.12
-------------------------------------------

The early namespace/step-mode migration below is retained; examples on the old
version side cannot run directly in the current version. Also read :doc:`./basic_concept`.

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
the attribute ``step_mode``. Refer to :doc:`./basic_concept` for more details.

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

The full-trace configuration in the current example avoids the verified
Triton LIF final-state backward compilation issue. See :doc:`./triton_backend`
for limits and additional memory cost.
