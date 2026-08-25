Distributed SNN Training and Inference
======================================

Authors: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_, `Wei Fang (fangwei123456) <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/distributed_training`

The high-level interfaces launch predefined training, evaluation, and offline
inference workflows. The low-level interfaces support custom models and loops.
The final section reports throughput and memory on the same four-RTX-4090 host
used for the training measurements.

API design rationale
--------------------

The API is divided into ``vision`` and ``llm`` workloads instead of assuming
that every SNN shares one parallel strategy. Spiking CNN channels, feature maps,
and pipeline boundaries have different semantics from LLM tokens, attention,
and context parallelism. A single model description would merely hide those
differences behind branches. The two paths are therefore symmetric only where
the concepts are genuinely shared: ``ModelConfig`` describes a model,
``ModelBuilder`` connects architecture-specific code, ``TrainingConfig``
describes training, and ``EvaluationConfig`` describes labeled evaluation.
Vision ``PredictionConfig`` and the LLM generation configs describe unlabeled
outputs. Each high-level entry point owns its runtime lifecycle.

The high-level interface is inspired by the current Megatron Core model
extension style. MCore separates declarative ``TransformerConfig`` data and
architecture-specific ``ModuleSpec`` / ``model_provider`` / ``forward_step``
from the common pipeline schedule, optimizer, and checkpoint lifecycle.
SpikingJelly keeps the same boundary—configuration states facts, a builder
adapts an architecture, and the training entry point owns the lifecycle—without
requiring users to edit one large predefined training function. An LLM builder
returns the MCore-native ``model_provider`` and ``forward_step``; a Vision
builder returns the stage, FSDP2 shard roots, and boundary shapes required by
PyTorch pipelines. Their outer style matches while each inner contract follows
its runtime.

The low-level interface follows a “reuse the runtime; add only SNN semantics”
rule. PyTorch supplies DP, FSDP2, device meshes, and general pipelines. Megatron
Core supplies LLM TP, PP, CP, the distributed optimizer, and sharded
checkpoints. SpikingJelly adds the pieces those runtimes do not express: SNN
temporal layouts and state resets, channel-sharded layers for channel-oriented
models, and spike-compression memopt. Memopt also remains separate from MCore
recomputation: the former handles SNN activations and spike representations,
while the latter is used only for non-overlapping Transformer subcomputations
when needed. The high-level ``train`` therefore covers standard workflows,
while custom tasks, models, and schedules can compose the low-level pieces
directly.

High-level APIs
---------------

Vision models
~~~~~~~~~~~~~

``spikingjelly.activation_based.distributed.vision`` provides image
classification training with PyTorch DDP, FSDP2, tensor parallelism, and
pipeline parallelism. ``vision.TrainingConfig`` describes the job and
``vision.train_classification`` executes it:

.. code-block:: python

    from pathlib import Path

    from spikingjelly.activation_based.distributed import vision

    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(
            time_steps=4, num_classes=1000, step_mode="m"
        ),
        dataset_builder=(
            "spikingjelly.activation_based.distributed.vision."
            "build_imagefolder_datasets"
        ),
        dataset_kwargs={"root": Path("/datasets/imagenet")},
        input_layout="NCHW",
        batch_size=32,
        loss_function="torch.nn.functional.cross_entropy",
        loss_kwargs={"label_smoothing": 0.1},
        tensor_parallel_size=2,
        data_parallel="fsdp2",
        precision="bf16",
        memopt_level=1,
    )
    metrics = vision.train_classification(config)

``batch_size`` is the batch size on each DP rank. The global batch is
``batch_size * DP`` and does not include TP, PP, or SNN time steps.
``tensor_parallel_size`` and ``pipeline_parallel_size`` select TP and PP; the
remaining ranks become DP replicas. The built-in models are
``SEWResNet34Config``, ``SpikformerConfig``, and ``SpikformerCIFAR10Config``.
The CIFAR-10 variant fixes the official 32×32 input, 4×4 patch stem, 384 channels,
12 attention heads, and 4 transformer blocks while retaining the same TP, PP, and
FSDP2 implementation. ``mixup_alpha`` enables serializable batch-level mixup;
``0`` disables it.
Rank zero prints one JSON record after every epoch containing the optimizer step,
train loss, validation loss, and validation accuracy. The returned ``metrics``
dictionary contains the final values and throughput statistics.

``input_layout`` explicitly declares the DataLoader batch layout. ``"NCHW"``
accepts static ``[N, C, H, W]`` images; single-step calls the model ``T`` times
with the same batch, while multi-step constructs contiguous
``[T, N, C, H, W]`` input. ``"NTCHW"`` accepts default-collated
``[N, T, C, H, W]`` frames from datasets such as CIFAR10-DVS and DVS Gesture,
validates T, and converts them to time-first layout. Tensor rank is never used
to infer the declared layout.

Before parallel wrapping, the entry point calls ``functional.set_step_mode`` and
resets the model with ``functional.reset_net`` after each complete time window.
Single-step currently does not support PP, memopt, or the Triton neuron backend.
The built-in SEW-ResNet34 supports both modes. Spikformer's architecture and
attention are intrinsically multi-step and are not wrapped to simulate a
single-step interface.
Single-step DDP disables per-forward buffer broadcasts so repeated calls do not
modify BatchNorm buffers needed by backward. It instead broadcasts buffers once
before each complete T window, keeping replicas synchronized without mutating a
saved buffer between single-step forwards.

``loss_function`` is the full import path of a callable receiving reduced
``[N, C]`` logits and class targets. It must return the batch-mean scalar used
for backward and loss reporting. ``loss_kwargs`` supplies keyword arguments to
each call. The same function is used by non-pipeline and pipeline training and
validation; top-1 accuracy remains the fixed classification metric.

The repository's synthetic-data entry point can verify the installation and
parallel configuration directly:

.. code-block:: bash

    torchrun --standalone --nproc-per-node=4 benchmark/vision_distributed.py \
        --model sew-resnet34 \
        --data-parallel fsdp2 \
        --tensor-parallel-size 2 \
        --precision bf16 \
        --max-steps 10

Custom models use ``vision.ModelConfig`` and ``vision.ModelBuilder``. ``build``
returns the model for the current rank, FSDP2 shard roots, and PP input/output
shapes. See :class:`spikingjelly.activation_based.distributed.vision.ModelBuilder`
for the complete signature.

LLMs
~~~~

``spikingjelly.activation_based.distributed.llm`` provides SNN language-model
training on Megatron Core. It requires Python 3.12 or newer. Install the optional
dependency first:

.. code-block:: bash

    uv pip install --editable ".[megatron]"

``llm.TrainingConfig`` combines:

* an ``llm.ModelConfig`` containing the MCore ``TransformerConfig``, vocabulary,
  context, and SNN time steps;
* an MCore ``OptimizerConfig``;
* the dataset builder, micro/global batch, training progress, evaluation, and
  checkpoint settings;
* optional SpikingJelly memopt.

When no topology is specified, ``llm.plan_training`` selects TP, PP, CP, and
recomputation from the GPU count, memory budget, and objective. For a known
topology, set ``TransformerConfig`` directly. The complete SpikeLM model,
optimizer, and dataset configuration lives in ``benchmark/snn_llm/cli.py``:

.. code-block:: bash

    torchrun --standalone --nproc-per-node=4 \
        benchmark/snn_llm/train_spikelm.py \
        --data /datasets/tokens \
        --output checkpoints/spikelm \
        --train-steps 200 \
        --global-batch-size 128

``llm.train`` currently supports only complete, independent fixed-T windows.
The architecture-specific ``forward_step`` owns temporal encoding, state
isolation, and reduction. Its MCore ``T*B`` envelope is not the generic
SpikingJelly ``step_mode="m"`` interface, so LLM configs intentionally do not
expose a ``step_mode`` field yet.

Offline distributed inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interface roles
^^^^^^^^^^^^^^^

Inference interfaces are divided by whether execution belongs to the training
lifecycle and whether ground truth is required. Validation and test use the same
evaluation computation and differ only in invocation time and dataset, so there
are no duplicate ``validate`` and ``test`` functions.

.. list-table:: Vision and LLM inference interface roles
    :header-rows: 1

    * - Scenario
      - Ground truth
      - Vision
      - LLM
      - Output
    * - Validation during training
      - Required
      - ``train_classification`` evaluates the validation dataset every epoch
      - ``train`` evaluates according to ``eval_interval`` / ``eval_steps``
      - Validation loss/accuracy or LM loss
    * - Post-training evaluation/test
      - Required
      - ``evaluate_classification``
      - MCore ``evaluate``
      - Aggregate loss, accuracy/perplexity, and performance metrics
    * - Post-training prediction/generation
      - Not required
      - ``predict_classification``
      - MCore ``generate`` or SGLang ``generate_sglang``
      - Per-sample logits or generated tokens; no evaluation metrics

``evaluate_classification`` requires every dataset item to be
``(image, target)``. Likewise, ``llm.evaluate`` requires ``input_ids``,
``labels``, and an optional ``loss_mask``. Prediction and generation do not read
ground truth: Vision ignores targets even if items are ``(image, target)``, while
LLM generation accepts prompts only. All three roles belong to training or
offline workflows; the SGLang path includes no HTTP server, router, or other
online-serving control plane.

Vision
^^^^^^

Vision inference remains native PyTorch. First export a training checkpoint to
a TP/PP-independent canonical artifact, then evaluate it under a different DP,
FSDP2, TP, or PP topology:

.. code-block:: bash

    torchrun --standalone --nproc-per-node=4 benchmark/vision_inference.py \
        --artifact artifacts/sew-resnet34.pt \
        --export-checkpoint checkpoints/step_00000010 \
        --model sew-resnet34

    torchrun --standalone --nproc-per-node=4 benchmark/vision_inference.py \
        --artifact artifacts/sew-resnet34.pt \
        --model sew-resnet34 \
        --data-parallel replicate \
        --batch-size 32

``vision.evaluate_classification`` returns global loss, accuracy, images/s, and
the busiest rank's peak memory. ``vision.predict_classification`` computes and
returns none of those metrics; it merges rank outputs by dataset index into one
HDF5 file containing only ``index`` and ``logits``. Classes can be derived with
``logits.argmax(axis=1)``. Padding never appears in the final output, so dataset
size need not divide DP or batch size.

LLMs
^^^^

LLMs expose two backends with different roles:

* MCore reuses the training model provider and sharded checkpoint.
  ``llm.evaluate(EvaluationConfig(...))`` runs complete DP/TP/PP/CP loss and
  perplexity evaluation. ``llm.generate(MCoreGenerationConfig(...), input_ids)``
  adds DP prompt sharding to TP/PP static-KV-cache generation. MCore cached
  generation requires CP=1.
* SGLang handles high-throughput post-training offline generation. It consumes a
  ``config.json`` plus safetensors artifact from a separate Python environment
  and starts no HTTP server or router.

SGLang 0.5.17 owns its PyTorch and Transformers runtime, so do not force it into
the main training environment:

.. code-block:: bash

    uv venv --python 3.12 .venv-sglang
    source .venv-sglang/bin/activate
    uv pip install --editable ".[sglang]"

The SpikeLM and Qwen2 export and generation references live in
``benchmark/snn_llm/inference.py``, ``qwen_distributed_inference.py``, and
``sglang_inference.py``. The SGLang models retain hidden state as
``[token, T, hidden]`` and fold T into the head dimension only at the
RadixAttention/KV-cache seam. The scheduler still owns one semantic request.

SGLang DCP can remove only KV-head replicas already created by TP. For a
SpikingJelly artifact, ``TP / effective_KV_heads`` must be at least the DCP
size. The high-level interface rejects any smaller topology before starting
the Engine rather than producing incorrect tokens.
The SpikeLM and Qwen2 reference adapters use SGLang's native layer staging and
``PPProxyTensors`` protocol. The Engine owns TP, PP, DP, and DCP configurations
satisfying the constraint above. SGLang 0.5.17 automatically disables its
overlap schedule when PP is greater than one.

Low-level APIs
--------------

Custom vision models
~~~~~~~~~~~~~~~~~~~~

``vision.ModelBuilder.build`` constructs the current PP stage, applies model
parallelism, and returns the FSDP2 shard roots. See ``SEWResNet34Builder`` and
``SpikformerBuilder`` for working implementations.

A minimal declaration has this form:

.. code-block:: python

    from dataclasses import dataclass
    from typing import ClassVar

    from spikingjelly.activation_based.distributed import vision

    @dataclass(frozen=True)
    class MyModelConfig(vision.ModelConfig):
        builder: ClassVar[str] = "my_package.model.MyModelBuilder"
        width: int = 128

    class MyModelBuilder(vision.ModelBuilder):
        def build(
            self,
            *,
            process_group,
            pipeline_rank,
            pipeline_size,
            pipeline_microbatches,
            device,
            micro_batch_size,
            memopt_level,
            memopt_compress_inputs,
        ):
            if pipeline_size != 1:
                raise ValueError("MyModelBuilder does not define PP stages.")
            model = build_my_model(self.config)
            model = parallelize_my_model(model, process_group)
            model.to(device)
            return model, ("blocks",), None, None

The model author supplies ``parallelize_my_model``. This example replaces model
layers with public components from
``spikingjelly.activation_based.distributed.tensor_parallel``:

.. code-block:: python

    from spikingjelly.activation_based.distributed.tensor_parallel import (
        ChannelShardBatchNorm2d,
        ChannelShardConv2d,
    )

    def parallelize_my_model(model, process_group):
        block = model.block
        block.conv1 = ChannelShardConv2d(block.conv1, process_group, "colwise")
        block.bn1 = ChannelShardBatchNorm2d(block.bn1, process_group)
        block.conv2 = ChannelShardConv2d(block.conv2, process_group, "rowwise")
        return model

The neuron consumes the local-channel tensor produced by the colwise layer
directly; no wrapper is required. The model author only needs to ensure that the
following rowwise layer consumes that local tensor.

Custom training loops
~~~~~~~~~~~~~~~~~~~~~

When the predefined ``train`` does not fit the task, compose the PyTorch
distributed interfaces with the SpikingJelly components above. The following
omits the task-specific ``build_my_model``, ``dataset``, and hyperparameters and
shows only the assembly order:

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    from spikingjelly.activation_based import functional

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    dp_group = mesh["dp"].get_group()
    tp_group = mesh["tp"].get_group()

    model = build_my_model()
    model = parallelize_my_model(model, tp_group).cuda(local_rank)
    functional.set_step_mode(model, step_mode)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], process_group=dp_group
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=mesh.get_local_rank("dp"),
        shuffle=True,
    )
    loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for images, labels in loader:
            images = images.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)
            sequence = (
                images.unsqueeze(0)
                .expand(time_steps, *images.shape)
                .contiguous()
            )

            optimizer.zero_grad(set_to_none=True)
            if step_mode == "s":
                logits = torch.stack([model(x_t) for x_t in sequence]).mean(0)
            else:
                logits = model(sequence).mean(0)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

Validation, mixed precision, scheduling, metric reduction, and checkpoints are
task-specific. The following calling contracts are required:

* world size must be divisible by the selected model-parallel size;
* data is sharded only over DP, and ranks in one TP group receive the same batch;
* create the optimizer after model parallelism and DDP/FSDP2 wrapping;
* reset SNN state after every independent batch or pipeline microbatch;
* global batch excludes TP, PP, CP, and SNN time steps.

Custom LLMs
~~~~~~~~~~~

An LLM subclasses ``llm.ModelConfig`` and points its ``builder`` class variable
to an ``llm.ModelBuilder``. The builder's ``build`` method returns the MCore
``model_provider`` and ``forward_step`` callbacks:

.. code-block:: python

    from spikingjelly.activation_based.distributed import llm

    class MyModelBuilder(llm.ModelBuilder):
        def build(self, *, use_snn_memopt: bool, resume: bool):
            return model_provider, forward_step

``model_provider`` builds the current PP stage. ``forward_step`` reads one
microbatch from the data iterator, invokes the model, and returns the MCore loss
callback. These callbacks can be passed to ``llm.train`` or used in a custom
MCore training loop. Complete SpikeLM and Qwen2 implementations are available in
``benchmark/snn_llm/spikelm.py`` and ``benchmark/snn_llm/qwen2.py``.

The SNN temporal layout is ``[T, B, S, H] -> [S, T*B, H]``. ``T`` is folded only
into the MCore batch dimension. It is not folded into the token dimension and
does not contribute to global batch size.

Measured results
----------------

The following results were measured on four RTX 4090 24-GiB GPUs. The machine
had no NVLink, and CUDA peer access was ``False`` across GPUs. The software stack
used PyTorch 2.8.0, Megatron Core 0.18.2, and Triton 3.4.0. These results are
relative references for a PCIe multi-GPU machine and should not be extrapolated
directly to an NVLink cluster.

Vision benchmarks
~~~~~~~~~~~~~~~~~

The Vision benchmarks fixed BF16, ``T=4``, 128 x 128 inputs, and 1000 classes.
The plots retain only one GPU, DP4, FSDP4, TP4, and PP4. Each curve labels its
largest successful global batch size (``G``). For one GPU, TP4, and PP4, global batch equals the
per-rank batch; for DP4 and FSDP4 it is four times the per-rank batch.

All topologies at a fixed global batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table restores the complete topology comparison at ``G=32``. Each
configuration started in a fresh process, warmed up for 10 optimizer steps,
measured 50 steps, and was repeated independently three times. Values are the
three-run medians. Throughput covers the whole job; memory is the highest CUDA
peak allocated memory among all ranks.

.. list-table:: Vision results across all topologies at fixed ``G=32``
    :header-rows: 1

    * - Topology
      - GPUs
      - SEW-ResNet34 images/s
      - SEW-ResNet34 GiB/GPU
      - Spikformer-S images/s
      - Spikformer-S GiB/GPU
    * - One GPU
      - 1
      - 269.0
      - 1.67
      - 270.1
      - 3.15
    * - DP2
      - 2
      - 261.8
      - 1.15
      - 264.3
      - 1.73
    * - FSDP2
      - 2
      - 146.9
      - 0.86
      - 172.6
      - 1.59
    * - TP2
      - 2
      - 257.6
      - 1.30
      - 262.6
      - 2.03
    * - PP2
      - 2
      - 83.9
      - 1.01
      - 85.9
      - 2.20
    * - DP4
      - 4
      - 259.9
      - 0.82
      - 258.8
      - 1.00
    * - FSDP4
      - 4
      - 145.7
      - 0.45
      - 169.4
      - 0.81
    * - TP4
      - 4
      - 240.3
      - 1.10
      - 246.9
      - 1.45
    * - PP4
      - 4
      - 122.2
      - 0.75
      - 148.5
      - 1.71
    * - TP2 + DP2
      - 4
      - 244.8
      - 0.82
      - 251.3
      - 1.10
    * - TP2 + FSDP2
      - 4
      - 143.3
      - 0.65
      - 166.7
      - 1.01
    * - PP2 + DP2
      - 4
      - 150.9
      - 0.53
      - 146.3
      - 1.18
    * - PP2 + FSDP2
      - 4
      - 74.4
      - 0.52
      - 86.2
      - 1.13
    * - TP2 + PP2
      - 4
      - 81.9
      - 0.86
      - 82.0
      - 1.45

At the same ``G=32``, multiple GPUs primarily reduce per-GPU memory rather than
automatically raising throughput: as compute per GPU shrinks, PCIe communication,
synchronization, and pipeline bubbles dominate more easily. The capacity curves
below answer a different question: how far each strategy extends the
throughput-memory frontier when it may use a larger global batch.

Batch size was increased by powers of two until the first candidate that could
not complete. SEW-ResNet34 succeeded through per-rank batch 256 on one GPU, DP4,
and FSDP4, and through 512 on TP4 and PP4. Each configuration started in a fresh
process, warmed up for 10 optimizer steps, and measured 40 steps. Spikformer-S
succeeded through per-rank batch 128 on one GPU, DP4, and FSDP4, and through 256
on TP4 and PP4, with 5 warmup and 25 measured steps. Every successful point was
repeated independently three times; the plots show the median and three-run
range. Timing includes H2D, forward, backward, communication, and the optimizer,
but excludes initialization, DataLoader work, validation, and checkpoints.

The vertical axis is aggregate job throughput: the global batch completed by all
ranks divided by the slowest rank's measured time. The horizontal axis is the
highest CUDA peak allocated memory among all ranks. They respectively answer how
many images the whole job processes per second and the minimum per-GPU memory
capacity it needs. Both axes are logarithmic. Failed candidates are not plotted
as throughput points.

.. figure:: ../../_static/tutorials/distributed/sew-resnet34-tradeoff.png
    :width: 720px
    :alt: SEW-ResNet34 aggregate throughput and per-GPU peak memory at different global batches

    SEW-ResNet34 aggregate training throughput versus the busiest GPU's peak allocated memory.

.. figure:: ../../_static/tutorials/distributed/spikformer-tradeoff.png
    :width: 720px
    :alt: Spikformer-S aggregate throughput and per-GPU peak memory at different global batches

    Spikformer-S aggregate training throughput versus the busiest GPU's peak allocated memory.

SEW-ResNet34 DP4 reached 3616.2 images/s and 11.25 GiB/GPU at ``G=1024``;
FSDP4 reached 3482.5 images/s and 10.86 GiB/GPU, while PP4 reached 1636.5
images/s at ``G=512``. The largest successful Spikformer-S points for DP4 and
FSDP4 both used ``G=512`` and reached 2503.8 and 2334.6 images/s; PP4 reached
1028.2 images/s at ``G=256``. TP4 plateaued early on both models: increasing
batch primarily raised memory, indicating that TP communication dominates on
this PCIe host. These numbers describe the throughput-capacity frontier under
more total work, not fixed-batch speedup.

.. list-table:: Vision capacity search (largest success → first failed candidate)
    :header-rows: 1

    * - Model
      - Topology
      - Largest successful ``B/G``
      - First failed ``B/G``
      - Result
    * - SEW-ResNet34
      - One GPU
      - 256/256
      - 512/512
      - CUDA OOM
    * - SEW-ResNet34
      - DP4
      - 256/1024
      - 512/2048
      - CUDA OOM
    * - SEW-ResNet34
      - FSDP4
      - 256/1024
      - 512/2048
      - CUDA OOM
    * - SEW-ResNet34
      - TP4
      - 512/512
      - 1024/1024
      - CUDA OOM
    * - SEW-ResNet34
      - PP4
      - 512/512
      - 1024/1024
      - NCCL collective timeout
    * - Spikformer-S
      - One GPU
      - 128/128
      - 256/256
      - CUDA OOM
    * - Spikformer-S
      - DP4
      - 128/512
      - 256/1024
      - CUDA OOM
    * - Spikformer-S
      - FSDP4
      - 128/512
      - 256/1024
      - CUDA OOM
    * - Spikformer-S
      - TP4
      - 256/256
      - 512/512
      - NCCL collective timeout
    * - Spikformer-S
      - PP4
      - 256/256
      - 512/512
      - NCCL collective timeout

``B`` is the per-rank batch. A collective timeout means that the candidate
produced no training metrics; it is neither a slow successful point nor labeled
as OOM without an OOM traceback.

LLM benchmarks
~~~~~~~~~~~~~~

The LLM benchmark used an approximately 1.41B-parameter SpikeLM with 24 layers,
hidden size 2048, 16 heads, FFN size 8192, vocabulary 50304, BF16, sequence 128,
and ``T=4``. Every capacity-search point below disabled SpikingJelly memopt and
gradient accumulation, so
``global_batch_size = micro_batch_size × data_parallel_size`` and each optimizer
step executes one micro batch on every DP rank.

All topologies at a fixed global batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table is the earlier fixed-work comparison across every tested
two-GPU, four-GPU, and hybrid topology: ``micro batch=1``, ``G=8``, 10 warmup
optimizer steps, 30 measured steps, and three independent repeats. Values are
the three-run medians. One GPU OOMed during distributed-optimizer initialization,
so DP2 is the relative-throughput baseline. Holding ``G=8`` across different DP
sizes required ``8 / DP`` accumulation steps in this fixed-work experiment:
four for DP2, two for DP4, and eight for each DP1 topology. It is a different
protocol from the no-accumulation capacity search below.

.. list-table:: 1.41B SpikeLM results across all topologies at fixed ``G=8``
    :header-rows: 1

    * - Topology
      - GPUs
      - Semantic tokens/s
      - GiB/GPU
      - Relative to DP2
    * - DP2
      - 2
      - 746.3
      - 17.35
      - 1.00x
    * - TP2
      - 2
      - 679.9
      - 12.86
      - 0.91x
    * - PP2
      - 2
      - 1008.3
      - 13.15
      - 1.35x
    * - CP2
      - 2
      - 417.7
      - 16.56
      - 0.56x
    * - DP4
      - 4
      - 585.6
      - 13.40
      - 0.78x
    * - TP4
      - 4
      - 673.1
      - 6.65
      - 0.90x
    * - PP4
      - 4
      - 1379.9
      - 8.09
      - 1.85x
    * - CP4
      - 4
      - 289.5
      - 12.34
      - 0.39x
    * - TP2 + DP2
      - 4
      - 817.1
      - 8.91
      - 1.09x
    * - PP2 + DP2
      - 4
      - 997.9
      - 9.20
      - 1.34x
    * - CP2 + DP2
      - 4
      - 446.0
      - 12.61
      - 0.60x
    * - TP2 + PP2
      - 4
      - 989.1
      - 6.81
      - 1.33x
    * - TP2 + CP2
      - 4
      - 427.2
      - 8.39
      - 0.57x
    * - PP2 + CP2
      - 4
      - 612.6
      - 8.44
      - 0.82x

At fixed ``G=8``, PP4 has the highest aggregate throughput, while TP4 and
TP2 + PP2 have the lowest per-GPU peak memory. CP cannot amortize its
communication at sequence length 128. This table compares topologies directly;
the following no-accumulation experiment compares their batch-capacity and
throughput limits.

The plot retains DP2, DP4, TP4, PP4, and CP4. DP2 succeeds only through micro
batch 1 (``G=2``), DP4 through micro batch 4 (``G=16``), and TP4, PP4, and CP4
through micro batch 16 (``G=16``). Each configuration started in a fresh process,
warmed up for 5 steps, measured 15 steps, and was repeated independently three
times. One GPU OOMed during distributed-optimizer initialization and is therefore
omitted. The LLM path uses MCore DDP and its distributed optimizer rather than
PyTorch FSDP2.

.. figure:: ../../_static/tutorials/distributed/spikelm-1.41b-tradeoff.png
    :width: 720px
    :alt: 1.41B SpikeLM aggregate throughput and per-GPU peak memory at different global batches

    1.41B SpikeLM aggregate training throughput versus the busiest GPU's peak allocated memory, without gradient accumulation or memopt.

PP4 at ``G=16`` reached 2997.4 semantic tokens/s and 14.85 GiB/GPU, the highest
throughput in this set; it is already close to the 2897.0 tokens/s measured at
``G=8``. TP4 reached 1684.4 tokens/s and 16.47 GiB/GPU at ``G=16`` and likewise
flattened noticeably after ``G=4``. DP4's largest successful point remains
``G=16`` at 1284.3 tokens/s and 17.55 GiB/GPU. CP4 reached 865.5 tokens/s and
16.52 GiB/GPU after scaling to ``G=16``, but remained below TP4 and PP4. DP2
retains only ``G=2`` at 303.0 tokens/s and 17.35 GiB/GPU.

.. list-table:: LLM capacity search (largest success → first failed candidate)
    :header-rows: 1

    * - Topology
      - Largest successful ``micro/G``
      - First failed ``micro/G``
      - Result
    * - DP2
      - 1/2
      - 2/4
      - CUDA OOM
    * - DP4
      - 4/16
      - 8/32
      - CUDA OOM
    * - TP4
      - 16/16
      - 32/32
      - CUDA OOM
    * - PP4
      - 16/16
      - 32/32
      - stalled, no training metrics
    * - CP4
      - 16/16
      - 32/32
      - stalled, no training metrics

The PP4 and CP4 ``micro=32`` candidates remained in fixed rank-wait states for
several normal run durations and were terminated as stalled. They are neither
throughput points nor mislabeled as OOM.

Points on different curves can have different global batches, so this plot shows
the throughput-capacity frontier rather than fixed-batch speedup. Complete
medians, three-run ranges, and batch configurations are available in the
:download:`summary CSV <../../_static/tutorials/distributed/distributed-tradeoff.csv>`.

Functional tests also covered BF16 TP4, PP4, TP2 x PP2, CP4, TP2 x CP2, and PP2
x CP2, plus FP8 TP4, PP4, and CP4. Every case produced finite loss and gradients
and nonzero gradients in the SNN modules. Under a 7-GiB memory budget, the planner
selected TP4, SpikingJelly memopt, and MCore selective ``core_attn``
recomputation; two training steps used 6.28 GiB. A TP2 x PP2 sharded
model/optimizer checkpoint also resumed successfully from step 1 to step 2.

Distributed inference benchmarks
--------------------------------

The inference benchmark used the same single-host 4 x RTX 4090 24-GiB environment
as training. ``nvidia-smi topo -m`` reports ``SYS`` from GPU0 to every other GPU
and ``NODE`` among GPUs 1--3, with no ``NV#`` link. ``nvidia-smi topo -p2p r/w``
returns ``CNS`` for every GPU pair. The host therefore has neither NVLink nor a
usable CUDA peer read/write path; NCCL traffic traverses PCIe/CPU interconnects.
The software stack matches training: PyTorch 2.8.0, Megatron Core 0.18.2, and
Triton 3.4.0.

Vision used BF16, ``T=4``, 1000 classes, and cached 224 x 224 synthetic images.
The non-PP SEW-ResNet34 per-rank grid is
``16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024``; Spikformer-S stops
between 384 and 1024 according to OOM. PP4 also measures ``1536, 2048`` and then
uses single-batch probes for its capacity boundary. Each successful throughput
point starts in a fresh process and uses four
DataLoader workers, runs five untimed batches and ten measured batches, and is repeated independently three
times. Timing includes H2D, forward, communication, and metric reduction but
excludes DataLoader work, model/artifact loading, and initialization. The plots
show medians and complete ranges. Only three-run points from the regular
protocol participate in the throughput-memory Pareto frontiers.
For the PP4 capacity tail at ``L >= 4096``, one high-level batch already contains
at least 256 pipeline microbatches. Each fresh process therefore measures one
batch without an additional warmup, while retaining three independent process
runs. CSV notes/status distinguish these points from the regular throughput
segment; they remain capacity-table evidence and are not connected to the
formal throughput lines.
This section uses ``L`` for the local batch on each DP rank/replica and ``G``
for the whole-job global batch. Always ``G = L × DP``; TP, PP, CP, and SNN time
steps ``T`` do not multiply G. PP additionally uses ``K`` for the number of
pipeline microbatches, each of size ``L / K``. Figure endpoints report only
global batch G.

Every capacity search uses the same growth rule: from the largest successful
point ``x``, try ``2x``; if ``2x`` fails, try ``1.5x``; if ``1.5x`` succeeds,
make it the new x and repeat. Non-streaming topologies continue to CUDA OOM.
When streaming PP or request queuing makes peak memory independent of G, the
search instead stops at a runtime boundary where a candidate cannot finish
within its own full timeout budget.

Inference PP uses a dedicated forward-only streaming schedule instead of the
``ScheduleGPipe`` backward state machine needed by training. It synchronizes the
pipeline group before returning each high-level batch, preventing cross-batch
work accumulation. SEW downsampling blocks sit before stage boundaries, while
Spikformer distributes blocks as ``0/2/2/2``. In the public configuration,
``pipeline_microbatches`` is the number of chunks cut from one DP rank's local
batch. It defaults to one, requires divisibility, and gives
``samples per chunk = batch_size / pipeline_microbatches``. The Vision
benchmark alone sets it to 4 for ``L < 64`` and to ``L / 16`` otherwise,
keeping 16 images per
pipeline microbatch at large B. The framework does not silently apply this rule
to user calls. The summary CSV records ``per_rank_batch_size``,
``global_batch_size``, ``pipeline_microbatches``, and
``pipeline_microbatch_size`` separately.
Vision uses exact-memory Pareto frontiers for regular-protocol points. SGLang
frontier lines merge measurements within the same 0.05-GiB horizontal bin and
retain its highest throughput. MCore instead connects every successful
batch-sweep point through OOM. The CSV keeps exact, unquantized memory and every
measurement for all three cases.

.. figure:: ../../_static/tutorials/distributed/sew-resnet34-inference-tradeoff.png
    :width: 720px
    :alt: SEW-ResNet34 distributed evaluation throughput and per-GPU peak memory

    SEW-ResNet34 regular-protocol aggregate evaluation throughput-memory Pareto frontiers.

.. figure:: ../../_static/tutorials/distributed/spikformer-inference-tradeoff.png
    :width: 720px
    :alt: Spikformer-S distributed evaluation throughput and per-GPU peak memory

    Spikformer-S regular-protocol aggregate evaluation throughput-memory Pareto frontiers.

At per-rank batch 128, SEW-ResNet34 reaches 845.7, 3368.9, 3109.3, 548.3, and
1404.1 images/s on one GPU, DP4, FSDP4, TP4, and PP4. Spikformer-S reaches
516.6, 2060.4, 2000.2, 412.2, and 1273.1 images/s. DP/FSDP approach linear
four-GPU throughput. PP reaches 1.66x and 2.46x single-GPU throughput and then
enters a stable large-batch plateau.

Pure TP4 remains below one GPU but is now stable; this is a model
compute-to-communication limit rather than scheduler variance. SEW executes 16
rowwise all-reduces totaling about 1.41 GB per batch, while Spikformer executes
12 totaling about 0.92 GB. Two TP2 replicas on four GPUs reach 1226.1 and 858.6
images/s, so practical deployments use TP to fit the model and DP to scale
throughput.

.. list-table:: Vision inference capacity boundaries
    :header-rows: 1

    * - Model
      - Topology
      - Largest three-run ``L/G``
      - First failure and final capacity evidence
    * - SEW-ResNet34
      - One GPU
      - 512/512
      - 768/768: sustained multi-batch CUDA OOM
    * - SEW-ResNet34
      - DP4
      - 512/2048
      - 768/3072: sustained multi-batch CUDA OOM
    * - SEW-ResNet34
      - FSDP4
      - 512/2048
      - 768/3072: sustained multi-batch CUDA OOM
    * - SEW-ResNet34
      - TP4
      - 512/512
      - 768/768: sustained multi-batch CUDA OOM
    * - SEW-ResNet34
      - PP4
      - 32768/32768
      - 49152/49152: runtime timeout
    * - Spikformer-S
      - One GPU
      - 256/256
      - 384/384: CUDA OOM
    * - Spikformer-S
      - DP4
      - 384/1536
      - 512/2048: CUDA OOM
    * - Spikformer-S
      - FSDP4
      - 256/1024
      - 384/1536: CUDA OOM
    * - Spikformer-S
      - TP4
      - 512/512
      - 768/768: CUDA OOM
    * - Spikformer-S
      - PP4
      - 32768/32768
      - 49152/49152: runtime timeout

Vision correctness tests also covered FSDP2, PP2, and exporting a TP2 x PP2
training checkpoint before restoring it under TP1 x DP4. The latter reported
validation losses 2.310132205 and 2.310132384.

MCore loss/perplexity evaluation uses Qwen2.5-0.5B QCFS, BF16, ``T=2``, and
sequence length 16. The TP1/DP4 baseline segment uses a fixed 128-sample
dataset; the newly measured TP2/PP2/PP4 points set dataset samples equal to G so
padding cannot depress throughput. It compares TP1, DP4, TP2, PP2, and PP4.
Because this model has 14 attention heads, TP2 is the valid pure-TP topology
above one on the four-GPU host; TP4 violates head divisibility.
Every point restores the same initialized state from a sharded checkpoint in a
fresh process, runs five untimed schedule batches, measures a complete schedule,
and is repeated independently three times; checkpoint/model initialization is
excluded. The new sweep explicitly sets ``NCCL_P2P_DISABLE=1``,
``NCCL_IB_DISABLE=1``, and
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. In the MCore API,
``micro_batch_size`` is the chunk size, while this section's L equals
``micro_batch_size × pipeline_microbatches``. Non-PP points use K=1. Every
PP2/PP4 point fixes K=4, so each chunk is ``L/4`` and grows with L through OOM;
the benchmark no longer fixes the chunk size while merely lengthening its queue.
The fine grid is ``16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024``, followed
by the ``2x/1.5x`` boundary search. Every three-run curve point uses five warmup
schedules. The PP4 L=3072 debug capacity probe uses one to avoid allocator
fragmentation from repeated warmups and serves only as capacity evidence.

.. figure:: ../../_static/tutorials/distributed/mcore-inference.png
    :width: 720px
    :alt: Qwen2.5-0.5B QCFS MCore distributed evaluation throughput and per-GPU peak memory

    Complete MCore loss/perplexity evaluation batch sweeps: aggregate semantic-token throughput versus the busiest GPU's peak allocated memory.

At small-batch L=16, TP1, TP2, PP2, and PP4 reach 4636.2, 3611.2, 1823.5, and
2203.6 semantic tokens/s. With K fixed at 4, each PP chunk contains only four
samples, so kernel and schedule overheads are not yet amortized. By L=384, TP1,
TP2, PP2, and PP4 reach 23145.8, 28975.2, 24707.7, and 28348.2 tokens/s; all
three model-parallel topologies exceed one GPU at the same L.

The best TP1, TP2, PP2, and PP4 points reach 24549.7, 29767.5, 30217.9, and
34317.2 tokens/s. The latter three are 1.21x, 1.23x, and 1.40x the best one-GPU
throughput. TP2 peaks at L=256 and 3.95 GiB/GPU; PP2 and PP4 peak at L=1024
and 7.57/7.40 GiB/GPU. Both three-run PP curves extend through L=2048 and about
14.5 GiB/GPU. Their capacity-tail drop repeats across all three runs and is the
measured cost of growing each chunk to 512 samples. A single debug capacity
probe completes PP4 L=3072 at 21.35 GiB/GPU, but the non-debug formal run times
out, so that point is excluded from the curve; L=4096 is a confirmed CUDA OOM.

.. list-table:: MCore capacity tail (largest completion → first failure)
    :header-rows: 1

    * - Topology
      - Largest ``L/G``
      - First failed ``L/G``
      - Status
    * - TP1
      - 384/384
      - 512/512
      - CUDA OOM
    * - DP4
      - 384/1536
      - 512/2048
      - CUDA OOM
    * - TP2
      - 1024/1024
      - 1536/1536
      - CUDA OOM
    * - PP2
      - 2048/2048
      - 2304/2304
      - CUDA OOM
    * - PP4
      - 2048/2048
      - 3072/3072
      - one debug probe completed but the formal run timed out; 4096 CUDA OOM

LLM generation used Qwen2.5-0.5B QCFS, BF16, ``T=2``, 8-token prompts, and
8-token outputs. SGLang compares TP1, DP2, DP4, TP2, PP2, PP4, and DP2 x TP2.
The global prompt batch G starts at 16 and follows the ``2x/1.5x`` growth rule.
The regular grid runs sequentially within one Engine per topology. Every
capacity-boundary ``2x/1.5x`` candidate owns a separate Engine lifetime and a
360-second budget. Regular points have three same-G warmups followed by three
measurements; scheduler points whose three-run max/min exceeds 1.3 remain
``unstable``. This update adds non-power-of-two fine grids near the TP1, PP2,
and PP4 frontiers. Those points also use three warmups, followed by seven timed
runs whose median resists periodic scheduler slow samples while the plot retains
the complete min/max range. Engine startup was excluded
and Radix cache was disabled. SGLang workers do not expose PyTorch allocator peak,
so the horizontal axis uses the busiest GPU's post-generation NVML device-memory
usage. It includes the ``memory_fraction_static=0.5`` KV pool and is not directly
comparable to Vision peak allocated memory. The static KV pool and
MiB-resolution NVML readings can give different G
values the same horizontal coordinate. The CSV retains every measurement, while
lines connect only the throughput-memory Pareto frontier. Only the
highest-throughput point at equal memory enters a line, avoiding vertical
segments and unexplained detached markers.

.. figure:: ../../_static/tutorials/distributed/sglang-inference.png
    :width: 720px
    :alt: Qwen2.5-0.5B QCFS SGLang offline-generation throughput

    SGLang offline-generation throughput-memory Pareto frontiers; frontier points show the three- or seven-run median and full range.

The best TP1, DP2, DP4, TP2, PP2, PP4, and DP2 x TP2 frontier points reach
15758.7, 18636.8, 25743.8, 9097.7, 12733.8, 9885.6, and 14355.1 generated
tokens/s at G=2048, 16384, 32768, 8192, 1024, 2048, and 32768. None should be
misread as per-GPU batch. The seven-run TP1, PP2, and PP4 fine grids continue to
G=3072, 12288, and 8192 without improving median throughput. Their plotted
endpoints are therefore measured plateau frontiers rather than prematurely
terminated rising segments.
SGLang caps in-flight
tokens and queues excess requests, so user request batch has no traditional OOM
point; the complete boundary is a throughput plateau rather than a manufactured
OOM. TP1 remains fastest at small and medium request batches while this 0.5B
model fits on one GPU; with a sufficiently large queue, pure DP2 and DP4 reach
1.18x and 1.63x TP1's best throughput. PP2 and PP4 remain below TP1 because PP
disables the overlap schedule and stage traffic crosses PCIe.

.. list-table:: SGLang capacity tail (largest completion → first independent timeout)
    :header-rows: 1

    * - Topology
      - Largest ``L/G``
      - First timeout ``L/G``
    * - TP1
      - 65536/65536
      - 98304/98304
    * - DP2
      - 49152/98304
      - 65536/131072
    * - DP4
      - 32768/131072
      - 49152/196608
    * - TP2
      - 49152/49152
      - 65536/65536
    * - PP2
      - 65536/65536
      - 98304/98304
    * - PP4
      - 49152/49152
      - 65536/65536
    * - DP2 x TP2
      - 32768/65536
      - 49152/98304

Correctness acceptance covered SpikeLM TP2 x PP2 checkpoint evaluation and
generation, CP2 x TP2 evaluation, and exact token equality between MCore and
SGLang TP1/TP2/DP2 x TP2 for Qwen2 and SpikeLM. Qwen2 SGLang PP2/PP4 and
SpikeLM SGLang PP2 also match their TP1 baselines token for token. Complete
medians, ranges, memory,
and failed statuses are available in the
:download:`throughput-memory CSV <../../_static/tutorials/distributed/distributed-inference-tradeoff.csv>`.
Regenerate all inference plots directly from the summary:

.. code-block:: bash

    python benchmark/plot_distributed_inference.py \
        docs/source/_static/tutorials/distributed/distributed-inference-tradeoff.csv \
        docs/source/_static/tutorials/distributed
