FlexSN
======

中文版: :doc:`../cn/flexsn`

``FlexSN`` turns a pure PyTorch single-step dynamics function into a stateful
SpikingJelly neuron. It provides a reference Torch path, a white-box HOP path
for ``torch.compile``, and a generated Triton path for CUDA.

Core contract
-------------

The single-step callable receives step inputs, states, then static inputs, and
returns outputs followed by updated states:

.. code-block:: text

    core(*step_inputs, *states, *static_inputs)
        -> (*outputs, *updated_states)

Only ``num_states`` is configured. FlexSN infers input and output arities from
the callable signature and a unit-tensor trace. Tensor-valued parameters must be passed through
``static_inputs`` rather than captured by the callable.

Managed forward
---------------

.. code-block:: python

    import torch
    from spikingjelly.activation_based.neuron import FlexSN

    def lif_core(x, v):
        h = v + (x - v) / 2.0
        spike = torch.sigmoid(h - 1.0)
        return spike, h * (1.0 - spike)

    neuron = FlexSN(lif_core, num_states=1, backend="torch")
    spike_seq = neuron(torch.randn(8, 64, 512))
    final_v = neuron.states[0]
    neuron.reset()

For multiple outputs, ``forward`` returns a tuple. ``states`` and the optional
``state_seqs`` cache are also tuples. Set ``store_state_seqs=True`` when the
complete trajectories are needed.

Functional forward
------------------

Functional execution takes explicit state and static inputs and does not
modify module state:

.. code-block:: python

    x = torch.randn(8, 64, 512)
    v0 = torch.zeros_like(x[0])
    (spike_seq,), (final_v,) = neuron.functional_forward(
        (x,), (v0,), static_inputs=()
    )

Static inputs
-------------

Static inputs are reused at every time step. Parameters are registered as
module parameters and other tensors as buffers. Managed ``forward`` uses the
registered values; functional execution receives them explicitly.

.. code-block:: python

    def plif_core(x, v, w):
        reciprocal_tau = w.sigmoid()
        h = v + reciprocal_tau * (x - v)
        spike = torch.sigmoid(h - 1.0)
        return spike, h * (1.0 - spike)

    w = torch.nn.Parameter(torch.tensor(0.0))
    neuron = FlexSN(plif_core, num_states=1, static_inputs=(w,), backend="torch")

Each static tensor must be scalar or have the same number of elements as a
single-step input. Arbitrary broadcasting is not supported.

Backends
--------

``torch`` supports ``step_mode="s"`` and ``"m"``. ``hop`` and ``triton``
support multi-step mode only. Backend and step mode can be changed after
construction; invalid combinations raise immediately. Legal changes preserve
states and clear only the derived ``state_seqs`` cache.

``backend="triton"`` requires CUDA and prepares its generated kernels
automatically without user-provided example tensors. Build failures are reported directly and never
fall back to another backend. Multi-step inputs with ``T == 0`` are rejected.

Migration
---------

The former ``FlexSNKernel`` class and ``FlexSN.kernel`` accessor were removed.
Use ``functional_forward`` for explicit-state execution. The constructor no
longer accepts ``num_inputs``, ``num_outputs``, ``example_inputs``,
``example_outputs``, or ``requires_grad``.
