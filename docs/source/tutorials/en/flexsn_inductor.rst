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
   * Ops inside ``core`` must be in the ``FX_TO_TRITON`` table.
     Unsupported ops fall back to ``eager_scan`` with a WARNING log.
     See :ref:`Op Coverage <flexsn-inductor-op-coverage-en>` below for the full list.
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

.. admonition:: torch.compile() is recommended for training (no fullgraph)
   :class: tip

   The training path wraps ``FlexSNFunction.apply`` with
   ``@torch._dynamo.disable``.  Under ``torch.compile()`` (without
   ``fullgraph=True``) Dynamo creates a graph break at FlexSN; the Triton
   fwd+bwd scan kernels continue to run at full speed while surrounding
   Conv/Linear layers are compiled by Inductor.  Net result: ~28% faster
   than without compile on RTX 4090 (T=32).

   ``torch.compile(fullgraph=True)`` raises an error; simply remove
   ``fullgraph=True``.

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
   * - Training + CUDA (with or without ``torch.compile``)
     - ``@torch._dynamo.disable``-wrapped Triton FlexSNFunction; under
       ``torch.compile()`` Dynamo creates a graph break here, FlexSN still
       uses Triton kernels, surrounding layers compiled by Inductor
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

**Training (without torch.compile)**: at initialisation time, ``aot_function``
traces both the forward and backward of ``core`` (no ``PYTORCH_JIT=0`` required)
and compiles dedicated Triton forward and backward scan kernels, each with a
``tl.static_range(T)`` time loop — one kernel launch per direction regardless of T.
On SpikingVGG-16-BN (T=4, B=64, CIFAR-10) this is ~12% faster than LIFNode triton
and ~45% faster than the pure PyTorch baseline.

**Training with or without torch.compile**: ``FlexSNFunction`` is wrapped with
``@torch._dynamo.disable``.  Under ``torch.compile()`` (without ``fullgraph=True``)
Dynamo creates a graph break and FlexSN continues using its fast Triton scan
kernels while surrounding layers are compiled by Inductor — ~28% faster overall
than without compile.  ``torch.compile(fullgraph=True)`` raises an error.

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
   * - CUDA training
     - ``"inductor"`` + ``torch.compile()`` (optional)
     - Single-kernel fwd+bwd scan; compile (no fullgraph) adds ~28% speedup
   * - Inference + cross-layer fusion
     - ``"inductor"`` + ``torch.compile``
     - Single-kernel scan + joint compilation with surrounding Conv/Linear

.. _flexsn-inductor-op-coverage-en:

Op Coverage
-----------

The ``FX_TO_TRITON`` table currently covers the following ATen ops
(supported for both inference and training):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Category
     - Ops
   * - Arithmetic
     - ``add``, ``sub``, ``mul``, ``div``, ``reciprocal``, ``neg``, ``rsub``
   * - Transcendentals
     - ``exp``, ``log``, ``log2``, ``sqrt``, ``rsqrt``, ``tanh``, ``sin``, ``cos``, ``erf``
   * - Rounding
     - ``floor``, ``ceil``, ``round``
   * - Activation / threshold
     - ``relu``, ``sigmoid``, ``sign`` / ``sgn``, ``abs``
   * - Comparisons
     - ``eq``, ``ne``, ``ge``, ``le``, ``gt``, ``lt``
   * - Logic / bitwise
     - ``logical_and`` / ``or`` / ``not``, ``bitwise_and`` / ``or`` / ``not``
   * - Binary math
     - ``minimum``, ``maximum``, ``pow``, ``fmod``
   * - Clamp
     - ``clamp``, ``clamp_min``, ``clamp_max``
   * - Type / construction
     - ``_to_copy`` (type cast), ``scalar_tensor``, ``zeros_like``, ``ones_like``
   * - Selection
     - ``where``, ``masked_fill``
   * - Backward-only
     - ``sigmoid_backward``, ``tanh_backward``, ``threshold_backward``
   * - Misc
     - ``clone``, ``detach``, ``spike_fn``

Ops not in this table (e.g. matrix ops, complex control flow) trigger
``eager_scan`` fallback with a WARNING log.
