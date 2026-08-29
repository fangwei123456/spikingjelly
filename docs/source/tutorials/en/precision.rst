Training and Inference Precision
================================

中文版： :doc:`../cn/precision`

``PrecisionConfig`` records two sets of options, both applied before optimizer
construction:

* ``mode`` controls regular model operations; ``fp8`` uses NVIDIA Transformer
  Engine;
* ``triton_storage``, ``triton_fwd``, and ``triton_bwd`` control state storage
  and forward/backward arithmetic in existing multi-step Triton IF/LIF/PLIF
  neurons.

The two sets can be used independently. Regular layers may run in BF16 while
Triton neurons remain in FP32, for example.

Installation
------------

BF16 and FP16 require only PyTorch. Model-level FP8 requires Transformer Engine:

.. code-block:: bash

    uv pip install --editable ".[fp8]"

Triton-neuron mixed precision additionally requires:

.. code-block:: bash

    uv pip install --editable ".[triton]"

Configuring regular layers
--------------------------

The order matters: move the model to its target device, call
``prepare_model_for_precision``, and only then create the optimizer. FP8
preparation replaces some modules; an optimizer created earlier would retain the
old parameters.

.. code-block:: python

    import torch
    from torch import nn

    from spikingjelly.activation_based.precision import (
        PrecisionConfig,
        prepare_model_for_precision,
    )

    device = torch.device("cuda")
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    precision = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode="fp8", fp8_recipe="auto"),
    )
    model = precision.model
    optimizer = torch.optim.AdamW(model.parameters())

    optimizer.zero_grad(set_to_none=True)
    with precision.autocast_context():
        output = model(torch.randn(256, 4096, device=device))
        loss = output.square().mean()
    precision.backward(loss, optimizer)

``mode`` accepts ``fp32``, ``fp16``, ``bf16``, or ``fp8``. The returned
``PrecisionArtifacts`` creates the FP16 GradScaler when needed, so the same loop
works for all four modes.

Inspecting the conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

FP8 currently converts aligned ``torch.nn.Linear``, SpikingJelly ``layer.Linear``,
pointwise Conv1d, and supported LayerNorm or adjacent fused patterns. A Linear
input dimension must be divisible by 16 and its output dimension by 8. Unaligned
layers remain in high precision and appear in the diagnostics:

.. code-block:: python

    report = precision.describe()
    print(report["conversion_report"])

Preparation raises when Transformer Engine, the hardware, or the recipe is
unavailable; it does not switch to BF16. At least one module must be convertible.
``fp8_recipe="auto"`` uses the installed Transformer Engine default. Select
``delayed``, ``current``, ``block``, or ``mxfp8`` when the numerical recipe must
be fixed explicitly.

Triton-neuron precision
-----------------------

Triton precision is set by ``prepare_model_for_precision`` rather than on every
neuron constructor. This example uses BF16 for regular layers, BF16 neuron storage
and forward arithmetic, and FP32 neuron backward arithmetic:

.. code-block:: python

    config = PrecisionConfig(
        mode="bf16",
        triton_storage="bf16",
        triton_fwd="bf16",
        triton_bwd="fp32",
    )
    precision = prepare_model_for_precision(model, device, config)

Only IFNode, LIFNode, and ParametricLIFNode instances with ``backend="triton"``
and ``step_mode="m"`` use these options. The function does not switch backends.
``triton_fwd`` and ``triton_bwd`` independently accept ``fp8``, ``fp16``,
``bf16``, or ``fp32``. FP8 arithmetic requires ``triton_storage`` to be
``float8_e4m3fn`` or ``float8_e5m2``. Exponentials and sensitive surrogate
operations remain FP32 inside the kernels and are not user options.

Distributed FP8
---------------

``distributed.vision`` prepares precision before DDP wrapping and optimizer
construction. It uses the DDP process group to synchronize Transformer Engine
scaling metadata. Model FP8 and Triton-neuron mixed precision currently require
DDP with TP=PP=1:

MCore LLMs do not use ``PrecisionConfig``. They keep their native transformer
and optimizer precision options.

.. code-block:: bash

    uv run torchrun --standalone --nproc-per-node=2 \
        benchmark/vision_distributed.py \
        --model spikformer --dataset synthetic \
        --data-parallel ddp --precision fp8 \
        --image-size 128 --classes 1024 \
        --batch-size 32 --max-steps 10

This command checks that FP8 and DDP work together. Performance remains
model-dependent; FP16 and BF16 are both faster in the Spikformer measurements
below.

When FP8 is faster
------------------

The short answer is that FP8 benefits large matrices, not every low-precision
workload. It does not help the regular FC-SNN training case or the current
Spikformer DDP case; sufficiently wide Linear and FC-SNN inference workloads do.

Measurements were taken on 2026-08-29. Linear and FC-SNN used one RTX 5090
32 GiB GPU, while DDP used two. The software stack was PyTorch 2.11.0+cu128 and
Transformer Engine 2.18.0. Linear and FC-SNN ran three times with rotated
precision order, and the tables report medians. A 5% throughput increase is the
cutoff for a useful win. Model conversion is excluded from timing.

Linear/GEMM crossover
~~~~~~~~~~~~~~~~~~~~~

The two ratios in each cell are ``FP8 / FP16`` and ``FP8 / BF16``:

.. list-table::
    :header-rows: 1

    * - workload (batch, width, depth)
      - training throughput
      - inference throughput
      - result
    * - 4096, 2304, 8
      - 0.553x / 0.527x
      - 0.915x / 0.966x
      - FP8 is slower
    * - 4096, 2560, 8
      - 0.720x / 0.705x
      - 1.117x / 1.100x
      - inference crosses the threshold
    * - 4096, 3072, 8
      - 0.978x / 1.026x
      - 1.454x / 1.629x
      - training remains below the threshold
    * - 4096, 3200, 8
      - 1.060x / 1.067x
      - 1.464x / 1.614x
      - training and inference both cross
    * - 3072, 4096, 8
      - 1.390x / 1.472x
      - 1.513x / 1.684x
      - training and inference both clearly win

For this dense Linear workload, with batch=4096 and depth=8, the inference
crossover lies between widths 2304 and 2560 and the training crossover between
widths 3072 and 3200. With width=4096, the training crossover lies between batches
2048 and 3072.

FP8 does not imply a fixed memory saving. At the 4096×3200×8 crossover, FP8
allocated memory is about 9%/14% lower than FP16 for training/inference, but
24%/34% higher than BF16.

End-to-end SNN and DDP
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - workload
      - training FP8 / FP16, BF16
      - inference FP8 / FP16, BF16
    * - FC-SNN: T16, batch 256, width 4096, depth 20
      - 0.954x / 0.942x
      - 0.901x / 0.896x
    * - FC-SNN: T16, batch 256, width 8192, depth 10
      - 0.908x / 0.887x
      - 1.310x / 1.299x

At width=8192, the inference matrices are large enough to offset the FP8 overhead.
Training still pays for backward and neuron operations and remains slower. Its
FP8 training memory is about 46%--47% higher than FP16/BF16.

The two-GPU Spikformer DDP benchmark used five timed steps:

.. list-table::
    :header-rows: 1

    * - batch per GPU
      - FP8 images/s
      - FP16 / BF16 images/s
      - FP8 / 16-bit allocated memory per GPU
    * - 32
      - 505.9
      - 566.2 / 562.6
      - 5517 MiB / 3513 MiB
    * - 144
      - 1005.3
      - 1591.8 / 1488.1
      - 23807 MiB / 14896 MiB

This Spikformer configuration has no FP8 crossover before reaching the practical
memory limit. FP8 and DDP work together, but FP16 or BF16 is the better choice for
this workload.

Choosing a mode
---------------

Start with BF16. Try FP8 after profiling shows that aligned Linear/MLP operations
dominate runtime and the matrix dimensions approach the measured crossover. FP8
usually cannot recover its conversion and metadata overhead when CNN, neuron, or
communication work dominates. Its memory use may also exceed BF16, so it is not a
memory-optimization switch.

Re-run the crossover benchmark on the target GPU:

.. code-block:: bash

    uv run python benchmark/benchmark_fp8_training_inference.py \
        --batch-size 4096 --width 3200 --depth 8 --num-classes 3200 \
        --warmup 8 --training-steps 20 --inference-steps 40 --trials 3 \
        --precisions fp16 bf16 fp8 --baseline-precision bf16 \
        --output benchmark/output/fp8-vs-bf16.json

Then change ``--baseline-precision`` to ``fp16`` and check both 16-bit baselines.
A full training comparison must also measure convergence on real data. This
benchmark checks finite loss, output, parameter updates, and steady-state
performance only.
