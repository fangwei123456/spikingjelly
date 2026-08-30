Training and Inference Precision
================================

中文版： :doc:`../cn/precision`

SpikingJelly has three precision configuration paths. Select the entry point for
the workflow at hand:

* custom PyTorch loops use ``PrecisionConfig`` and
  ``prepare_model_for_precision``;
* ``distributed.vision`` stores ``PrecisionConfig`` in its training, evaluation,
  or prediction configuration;
* ``distributed.llm`` uses Megatron Core's own ``TransformerConfig`` and
  ``OptimizerConfig`` rather than ``PrecisionConfig``.

Model-level FP16/BF16/FP8 and
Triton-neuron precision are separate dimensions. Regular layers may run in BF16
while Triton neurons remain in FP32.

Installation
------------

BF16 and FP16 require only PyTorch. Model-level FP8 requires Transformer Engine:

.. code-block:: bash

    uv pip install --editable ".[fp8]"

Triton-neuron mixed precision additionally requires:

.. code-block:: bash

    uv pip install --editable ".[triton]"

In a pip environment with PyTorch already installed, add
``--no-build-isolation`` so Transformer Engine selects its CUDA wheel from the
existing ``torch.version.cuda``. An isolated build may temporarily install a
different PyTorch release and select a wheel that does not match the runtime:

.. code-block:: bash

    python -m pip install --no-build-isolation \
        "transformer-engine[pytorch]>=2.16,<3"

Path 1: custom training or inference
------------------------------------

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

Inference uses the same context without ``backward``:

.. code-block:: python

    model.eval()
    with torch.inference_mode(), precision.autocast_context():
        output = model(input_tensor)

Inspecting an FP8 conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Controlling unconverted operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transformer Engine changes only converted modules. FP8 wraps the remaining
CUDA operations in BF16 autocast by default, avoiding a model-wide FP32 boundary
through Conv2d, BatchNorm, or neurons:

.. code-block:: python

    config = PrecisionConfig(
        mode="fp8",
        fp8_recipe="auto",
    )

Use ``fp8_fallback_dtype="fp16"`` or ``"fp32"`` to reproduce an explicit
experimental path. Here, fallback means operations not covered by FP8, not
recovery from an error. The field controls ordinary CUDA autocast and TE output
boundaries; it does not claim that those operations use FP8 kernels.

Some Transformer Engine recipes serialize FP8 metadata as a pickle. Set
``NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`` only when restoring a trusted
checkpoint; do not enable it for unknown checkpoints.

Configuring Triton neurons
~~~~~~~~~~~~~~~~~~~~~~~~~~

Triton precision is set by ``prepare_model_for_precision`` rather than on every
neuron constructor. Regular layers can use BF16 with BF16 neuron storage
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

Path 2: ``distributed.vision``
--------------------------------

The high-level vision API reads ``precision`` before parallel wrapping and
optimizer construction. ``TrainingConfig``, ``EvaluationConfig``, and
``PredictionConfig`` all accept the same ``PrecisionConfig``:

.. code-block:: python

    from spikingjelly.activation_based import distributed
    from spikingjelly.activation_based.precision import PrecisionConfig

    config = distributed.vision.TrainingConfig(
        model=model_config,
        dataset_builder=dataset_builder,
        precision=PrecisionConfig(
            mode="bf16",
            triton_storage="bf16",
            triton_fwd="bf16",
            triton_bwd="fp32",
        ),
    )
    result = distributed.vision.train_classification(config)

In the repository CLI, ``--precision`` maps to ``mode``. The remaining fields
come from ``--fp8-recipe``, ``--triton-storage``, ``--triton-fwd``, and
``--triton-bwd``:

.. code-block:: bash

    uv run torchrun --standalone --nproc-per-node=2 \
        benchmark/vision_distributed.py \
        --model spikformer --dataset synthetic \
        --data-parallel ddp --precision fp8 \
        --image-size 128 --classes 1024 \
        --batch-size 32 --max-steps 10

``distributed.vision`` prepares precision before DDP wrapping and optimizer
construction. It uses the DDP process group to synchronize Transformer Engine
scaling metadata. Model FP8 and Triton-neuron mixed precision currently require
DDP with TP=PP=1. Vision PP supports FP32 and BF16. Ordinary FP32/BF16 execution
is not subject to this experimental-precision restriction.

Path 3: ``distributed.llm``
----------------------------

``distributed.llm`` does not use ``PrecisionConfig``. Set model and optimizer
precision in MCore ``TransformerConfig`` and ``OptimizerConfig`` and keep the two
configurations consistent:

.. code-block:: python

    import torch
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import TransformerConfig

    transformer = TransformerConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        ffn_hidden_size=8192,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    optimizer = OptimizerConfig(
        lr=3e-4,
        min_lr=3e-5,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )

Place ``transformer`` in the concrete ``distributed.llm.ModelConfig`` and
``optimizer`` in ``distributed.llm.TrainingConfig``. For FP16, set
``fp16=True``, ``bf16=False``, and ``params_dtype=torch.float16`` in both
configurations. MCore FP8 normally uses a BF16 base with ``fp8="hybrid"`` and a
matching ``fp8_recipe`` in ``TransformerConfig``; set the same recipe on
``OptimizerConfig``. With PP, ``pipeline_dtype`` must match ``params_dtype``.

Standalone evaluation and cached generation reuse the model's
``TransformerConfig``. SGLang artifact export currently requires MCore BF16.
See :doc:`./distributed_training` for complete model, data, and training
configuration examples.

When FP8 is faster
------------------

FP8 benefits large matrices, not every low-precision workload. FC-SNN depends on
the neuron backend and the Linear-to-neuron boundary; Spikformer has a larger
convolution, BatchNorm, and neuron share and does not cross the FP16/BF16
baseline on one GPU yet.

The dense Linear and two-GPU DDP results below were measured on 2026-08-29 with
Transformer Engine 2.18.0. The replacement FC-SNN rows and the new single-GPU
Spikformer rows were measured on 2026-08-30 with PyTorch 2.11.0+cu128,
Transformer Engine 2.17.1, and one 32-GiB RTX 5090. The latter used 2.17.1
because the image's TE 2.18.0 extension could not load; the two software stacks
must not be conflated. Each FC-SNN/Spikformer steady-state case used three
independent processes and reports the median. A 5% throughput increase is the
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
    * - FC-SNN: T16, batch 256, width 4096, depth 20 (Triton LIF, FP16 fallback)
      - 1.533x / 1.677x
      - 1.578x / 1.786x
    * - FC-SNN: T16, batch 256, width 8192, depth 10 (Triton LIF, FP32 fallback)
      - 1.544x / 1.468x
      - 1.648x / 1.471x

The ratios are end-to-end throughput ratios in the order ``FP8 / FP16`` and
``FP8 / BF16``. Both FC-SNN cases use the existing Triton LIF; Triton neuron
storage is not enabled and neuron computation remains high precision. W4096
uses ``fp8_fallback_dtype="fp16"`` and its training/inference peak allocated
memory is 4279.1/1460.3 MiB; W8192 is the earlier result without an outer
autocast. Thus “FP8 Linear + Triton LIF” wins at these sizes, but this does not
mean that all neuron state is FP8 or that FP8 always saves memory.

The slower FC-SNN numbers previously shown in this tutorial used Torch LIF and
are retained only in the Nsight root-cause report, not as the recommended
FC-SNN benchmark.

Requested FP8 tracked dense-MAC coverage for W4096/depth20 is 100%.

Reproduce the FC-SNN profile with:

.. code-block:: bash

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o fcsnn-fp8 \
      uv run python benchmark/benchmark_train_precision_snn_fc.py \
        --backend triton --precisions fp8 --fp8-fallback-dtype fp16 \
        --profile --profile-steps 10 --output fcsnn-fp8.json

Single-GPU Spikformer
~~~~~~~~~~~~~~~~~~~~~

For ``spikformer_ti`` with ``T=4``, input size ``224``, eager execution, and
Triton LIF, the RTX 5090 end-to-end results are:

.. list-table::
    :header-rows: 1

    * - workload
      - precision path
      - step latency
      - throughput
      - throughput vs FP16 / BF16
      - peak allocated
    * - Inference: batch 64, 1000 classes
      - FP16
      - 16.220 ms
      - 3945.7 images/s
      - --
      - 1974.8 MiB
    * - Inference: batch 64, 1000 classes
      - BF16
      - 16.372 ms
      - 3909.1 images/s
      - --
      - 1974.8 MiB
    * - Inference: batch 64, 1000 classes
      - FP8 + FP32 fallback
      - 37.173 ms
      - 1721.7 images/s
      - 0.436x / 0.441x
      - 3377.3 MiB
    * - Inference: batch 64, 1000 classes
      - FP8 + FP16 fallback
      - 22.280 ms
      - 2872.6 images/s
      - 0.728x / 0.735x
      - 2004.8 MiB
    * - Inference: batch 64, 1000 classes
      - FP8 + BF16 fallback (default)
      - 22.402 ms
      - 2856.9 images/s
      - 0.724x / 0.731x
      - 2004.8 MiB
    * - Training: batch 32, 1024 classes (alignment diagnostic)
      - FP16
      - 37.869 ms
      - 845.0 samples/s
      - --
      - 4508.2 MiB
    * - Training: batch 32, 1024 classes (alignment diagnostic)
      - BF16
      - 44.671 ms
      - 716.4 samples/s
      - --
      - 4511.5 MiB
    * - Training: batch 32, 1024 classes (alignment diagnostic)
      - FP8 + FP32 fallback
      - 59.538 ms
      - 537.5 samples/s
      - 0.636x / 0.750x
      - 7724.4 MiB
    * - Training: batch 32, 1024 classes (alignment diagnostic)
      - FP8 + FP16 fallback
      - 41.499 ms
      - 771.1 samples/s
      - 0.913x / 1.076x
      - 4398.4 MiB
    * - Training: batch 32, 1024 classes (alignment diagnostic)
      - FP8 + BF16 fallback (default)
      - 48.654 ms
      - 657.7 samples/s
      - 0.778x / 0.918x
      - 4398.4 MiB

The training row uses 1024 classes only to satisfy the current TE FP8 backward
16-alignment requirement; an ImageNet-1000 head currently fails with
``lda % 16 == 0``. Nsight shows that FP8 autocast covers only TE pointwise
Conv1d/Linear modules; patch-stem Conv2d, BatchNorm, and LIF outputs fall back
to FP32, adding elementwise, copy, and layout kernels. Choose FP16/BF16 for
Spikformer until that boundary is fixed; do not infer a Spikformer win from the
FC-SNN Triton result. Reproduce a single-GPU profile with:

.. code-block:: bash

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o spikformer-fp8 \
      uv run python benchmark/benchmark_snn_single_gpu.py case \
        --model spikformer_ti --phase inference --execution eager \
        --batch-size 64 --warmup 50 --steps 10 --profile --precision fp8 \
        --neuron-backend triton --fp8-fallback-dtype bf16 \
        --tensor-metadata spikformer-fp8.tensors.jsonl \
        --output spikformer-fp8.json

The ``--profile`` flag bounds capture through ``cudaProfilerApi``.

The default BF16 fallback reduces FP8 training/inference latency by 18.3%/39.7%,
but does not cross BF16. The explicit FP16 fallback is faster but has less
dynamic range. Requested FP8 tracked dense-MAC coverage is 41.98%. The current
RTX 5090 stack does not expose FP8 Conv2d; an emulated path is not evidence of
increased hardware coverage.

The historical two-GPU Spikformer DDP benchmark used five timed steps:

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

Start with BF16. Try FP8 for FC-SNN only with Triton LIF and matrix sizes near
the table's crossover. For CNN/neuron-heavy models such as Spikformer, continue
with FP16/BF16 until the FP32 boundary is optimized. Profile against the
end-to-end step; FP8 memory may exceed BF16 and is not a memory-optimization
switch.

Run the crossover benchmark on the target GPU:

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
