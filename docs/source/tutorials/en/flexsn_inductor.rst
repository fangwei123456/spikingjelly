FlexSN Inductor Backend
=======================

Author: `wei.fang <https://github.com/fangwei123456>`_

中文版: :doc:`../cn/flexsn_inductor`

``FlexSN`` adds a ``backend="inductor"`` option that compiles a user-defined
single-step dynamics function ``core`` into efficient Triton GPU kernels.
Compared with ``backend="triton"`` (the in-house FX→Triton transpiler), the
Inductor path:

* Requires no manual op-mapping table — most PyTorch ops inside ``core`` just work.
* Does **not** require ``PYTORCH_JIT=0``.
* Integrates with ``torch.compile``, enabling cross-layer fusion with surrounding modules.
* Ships dedicated Triton kernels for both inference and training,
  matching or exceeding ``backend="triton"`` throughput.

.. admonition:: Important
   :class: warning

   * Only CUDA devices are supported.
   * Ops inside ``core`` must be in the ``FX_TO_TRITON`` table (common
     element-wise ops: add/sub/mul/div, comparisons, sigmoid, type casts, etc.).
     Unsupported ops fall back to ``eager_scan`` with a log warning.
   * For training, ``core`` should use a surrogate gradient
     (e.g. :class:`spikingjelly.activation_based.surrogate.Sigmoid`) instead
     of a hard threshold — hard thresholds yield zero gradients by design.

Quick Start — Inference
-----------------------

.. code:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    neuron = FlexSN(core=lif_core, num_inputs=1, num_states=1,
                    num_outputs=1, step_mode="m", backend="inductor").cuda()

    x = torch.randn(8, 64, 512, device="cuda")
    with torch.no_grad():
        out = neuron(x)   # no torch.compile required

    # Optional: wrap with torch.compile for cross-layer fusion
    model = nn.Sequential(nn.Linear(512, 512), neuron, nn.Linear(512, 512)).cuda()
    model = torch.compile(model, fullgraph=True)
    out = model(x)

Quick Start — Training
----------------------

Use a surrogate gradient to make spike signals differentiable:

.. code:: python

    import torch
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    sg = surrogate.Sigmoid(alpha=4.0)

    def lif_core_sg(x: torch.Tensor, v: torch.Tensor):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = sg(h - v_th)        # Sigmoid surrogate gradient
        return s, h * (1.0 - s)

    neuron = FlexSN(core=lif_core_sg, num_inputs=1, num_states=1,
                    num_outputs=1, step_mode="m", backend="inductor").cuda()

    x = torch.randn(8, 64, 512, device="cuda", requires_grad=True)
    out = neuron(x)
    out.sum().backward()        # BPTT via dedicated Triton fwd+bwd kernels
    print(x.grad.shape)         # [8, 64, 512]

    # Also works under torch.compile: Dynamo unrolls the loop, Inductor compiles
    model = torch.compile(neuron, fullgraph=True)
    out = model(x)
    out.sum().backward()

Kernel Dispatch Strategy
------------------------

``multi_step_forward`` selects a path automatically based on context:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Condition
     - Path
   * - Inference (no grad) + CUDA
     - Single Triton scan kernel (``tl.static_range(T)``, 1 launch)
   * - Training + CUDA + **outside** ``torch.compile``
     - Triton FlexSNFunction (dedicated fwd+bwd kernels, BPTT)
   * - Training + CUDA + **inside** ``torch.compile``
     - ``eager_scan`` (Dynamo unrolls loop, Inductor compiles, autograd backward)
   * - CPU or kernels unavailable
     - ``eager_scan`` / ``flex_sn_scan`` HOP fallback

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 22 20

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
     - No (optional)
     - compile enables cross-layer fusion
   * - Inference perf (vs triton)
     - slow
     - baseline
     - ≤ 0.52× **faster**
     - RTX 4090 measured
   * - Training perf (vs triton)
     - slow
     - baseline
     - ≤ 0.69× **faster**
     - VGG16-BN, CIFAR-10
   * - Cross-layer fusion
     - No
     - No
     - Yes (needs torch.compile)
     -
   * - float16 support
     - Yes
     - Yes
     - Yes
     -

Performance Notes
-----------------

**Inference**: at initialisation time, ``core`` is traced with ``make_fx`` and
the FlexSN template generates a single Triton scan kernel with a
``tl.static_range(T)`` time loop. Every inference call triggers exactly one
kernel launch regardless of T.

**Training**: at initialisation time, ``aot_function`` traces both the forward
and backward of ``core`` (no ``PYTORCH_JIT=0`` required) and compiles dedicated
Triton forward and backward scan kernels. For non-differentiable outputs (e.g.
hard-threshold spikes), a shim wrapper is generated automatically to match the
AOT backward signature. Under ``torch.compile``, the training path uses
``eager_scan`` so Dynamo can trace the time loop into an FX graph.

On SpikingVGG-16-BN (T=4, B=64, CIFAR-10) FlexSN inductor is ~12% faster than
LIFNode triton and ~45% faster than the pure PyTorch baseline.

When to Use Each Backend
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Scenario
     - Recommended backend
     - Reason
   * - Prototyping / CPU
     - ``"torch"``
     - No constraints
   * - CUDA inference, max throughput
     - ``"inductor"``
     - Single-kernel scan, no PYTORCH_JIT=0
   * - CUDA training, no torch.compile
     - ``"inductor"``
     - Triton fwd+bwd kernels, faster than triton
   * - CUDA training + torch.compile
     - ``"inductor"``
     - Compatible; compile uses eager_scan + Inductor
   * - Cross-layer fusion
     - ``"inductor"`` + ``torch.compile``
     - Surrounding Linear/Conv compiled jointly with FlexSN
