Upgrade to V2
=======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/migrate_from_legacy`

V2 has breaking interface changes. Update old configuration using the table below.

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
    node = neuron.LIFNode(step_mode="m").to(device)
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
three values. Check arguments and return values in :doc:`./neuron` and the
public API when migrating a function name.

Normal usage needs no implementation choice. See :doc:`./triton_backend` for
installation/diagnostics, :doc:`./precision` for policies, :doc:`./flexsn` for
custom dynamics and :doc:`./memopt` for memory optimization/projections.
There is no automatic migration script or promise that an old whole-module
pickle/checkpoint loads directly. Prefer trusted ``state_dict`` files and check
keys/shapes against the current model definition.

Rename of Packages
-------------------------------------------
In the new version, SpikingJelly renames some sub-packages, which are:

===============  ==================
Old              New            
===============  ==================
clock_driven     activation_based
event_driven     timing_based    
===============  ==================

For earlier migration examples, see the `0.0.0.0.14 documentation <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/#index-en>`_.
