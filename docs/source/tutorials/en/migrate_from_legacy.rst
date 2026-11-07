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
    * - ``STBIFNeuron``
      - ``STBIFNode``
    * - Positional ``tau`` and ``detach_reset`` of Spikformer
      - Pass them as keyword arguments
    * - ``cuda_threads``, ``cuda_compiler_options``, ``cuda_compiler_backend``,
        ``save_spike_as_bool_in_neuron_kernel``, ``save_bool_spike_level`` and
        the matching ``SJ_*`` variables
      - Delete them
    * - ``MSTDPLearner(..., batch_size=...)``
      - Delete ``batch_size``; call ``reset()`` before the batch size changes
    * - FPTT/OTTT/SLTT imports and old STDP example paths
      - ``spikingjelly.activation_based.learning.online_learning``
        (``ottt_online_training`` also covers SLTT); examples live in
        ``spikingjelly.activation_based.examples.plasticity.{stdp_trace,mstdp,mstdpet,stdp_mnist}``
    * - Tempotron ``T``/``ret_type``, dense-spike forward, ``mse_loss`` and the
        checkpoint key ``model.summation_layer.weight``
      - Use ``duration``. ``forward(spike_times, input_indices, batch_offsets)``
        returns ``(v_max, t_peak, t_spike)``. Write the loss yourself. The
        checkpoint key is ``weight``
    * - ``GaussianTuning.encode``
      - ``forward``; the checkpoint buffers are ``mu`` and ``sigma2``
    * - Inference artifacts of schema v1
      - Re-export them; loaders accept schema v2 only

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

Package rename
-------------------------------------------
The ``clock_driven``/``event_driven`` rename shipped in 0.0.0.0.14 and is
documented in the `0.0.0.0.14 documentation <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/>`_.
