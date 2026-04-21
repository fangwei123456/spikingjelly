FlexSN Inductor Backend
=======================

Author: `wei.fang <https://github.com/fangwei123456>`_

中文版: :doc:`../cn/flexsn_inductor`

``FlexSN`` adds a ``backend="inductor"`` option that compiles a user-defined
single-step dynamics function ``core`` into GPU kernels via
`PyTorch Inductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir/747>`_.
Compared with ``backend="triton"`` (the in-house FX→Triton transpiler), the
Inductor path:

* Requires no manual op-mapping table — most PyTorch ops inside ``core`` just work.
* Integrates with ``torch.compile``, enabling cross-layer fusion with surrounding modules.
* Does **not** require ``PYTORCH_JIT=0``.

.. admonition:: Important
   :class: warning

   * ``backend="inductor"`` must be used with ``torch.compile(model, fullgraph=True)``.
     Instantiating ``FlexSN`` alone does not trigger compilation.
   * Only CUDA devices are supported.

Quick Start
-----------

Define a ``core`` function for an LIF neuron and use ``backend="inductor"``:

.. code:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        """Single-step LIF dynamics: charge → fire → reset."""
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau        # charge
        s = (h >= v_th).to(h.dtype)  # fire
        return s, h * (1.0 - s)      # spike output + updated membrane potential

    neuron = FlexSN(
        core=lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    # torch.compile is required; without it the HOP runs in eager mode
    model = nn.Sequential(nn.Linear(512, 512), neuron, nn.Linear(512, 512))
    model = torch.compile(model, fullgraph=True)
    model = model.cuda()

    x = torch.randn(8, 64, 512, device="cuda")  # [T, B, N]
    out = model(x)

Numerical Parity with ``backend="triton"``
------------------------------------------

Both backends produce identical results (max abs diff = 0.0) within ``atol=1e-5``:

.. code:: python

    import os, torch
    os.environ["PYTORCH_JIT"] = "0"
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x, v):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    torch.manual_seed(0)
    x = torch.randn(8, 32, 1024, device="cuda")

    n_tri = FlexSN(lif_core, 1, 1, 1, step_mode="m", backend="triton").cuda()
    n_ind = FlexSN(lif_core, 1, 1, 1, step_mode="m", backend="inductor").cuda()
    c_ind = torch.compile(n_ind, fullgraph=True)

    torch.testing.assert_close(c_ind(x), n_tri(x), atol=1e-5, rtol=1e-5)
    print("Parity verified ✓")

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Property
     - ``"torch"``
     - ``"triton"``
     - ``"inductor"``
     - Notes
   * - Device
     - CPU / CUDA
     - CUDA
     - CUDA
     -
   * - Requires PYTORCH_JIT=0
     - No
     - Yes
     - No
     -
   * - Requires torch.compile
     - No
     - No
     - Yes (recommended)
     -
   * - Cross-layer fusion
     - No
     - No
     - Yes
     - Design goal G3
   * - T=8 perf (vs triton)
     - slow
     - baseline
     - 0.52× **faster**
     - RTX 4090 measured
   * - T=32 perf (vs triton)
     - slow
     - baseline
     - 0.28× **faster**
     - RTX 4090 measured

Performance Notes
-----------------

At initialisation time, ``backend="inductor"`` traces ``core`` with ``make_fx``
and uses the existing FlexSN template infrastructure to build a single Triton
scan kernel with a ``tl.static_range(T)`` time loop. Every inference call
triggers exactly one kernel launch regardless of T, matching or exceeding
the throughput of ``backend="triton"``.

When to Use Each Backend
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Recommended backend
     - Reason
   * - Prototyping / any device
     - ``"torch"``
     - No constraints
   * - CUDA single layer, max performance
     - ``"triton"`` or ``"inductor"``
     - Comparable throughput
   * - Cross-layer fusion / any T
     - ``"inductor"``
     - Single-kernel scan + torch.compile fusion
   * - Training (backward)
     - ``"inductor"``
     - BPTT supported via eager_scan
