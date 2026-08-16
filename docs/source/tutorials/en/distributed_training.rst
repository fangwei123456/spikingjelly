Distributed SNN Training
========================

中文版： :doc:`../cn/distributed_training`

The high-level interfaces launch predefined training workflows. The low-level
interfaces support custom models and training loops. The final section reports
measured throughput and memory.

High-level APIs
---------------

Vision models
~~~~~~~~~~~~~

``spikingjelly.activation_based.distributed.vision`` provides image
classification training with PyTorch DDP, FSDP2, tensor parallelism, and
pipeline parallelism. ``vision.TrainingConfig`` describes the job and
``vision.train`` executes it:

.. code-block:: python

    from pathlib import Path

    from spikingjelly.activation_based.distributed import vision

    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(time_steps=4, num_classes=1000),
        dataset_builder=(
            "spikingjelly.activation_based.distributed.vision."
            "build_imagefolder_datasets"
        ),
        dataset_kwargs={"root": Path("/datasets/imagenet")},
        batch_size=32,
        tensor_parallel_size=2,
        data_parallel="fsdp2",
        precision="bf16",
        memopt_level=1,
    )
    metrics = vision.train(config)

``batch_size`` is the batch size on each DP rank. The global batch is
``batch_size * DP`` and does not include TP, PP, or SNN time steps.
``tensor_parallel_size`` and ``pipeline_parallel_size`` select TP and PP; the
remaining ranks become DP replicas. The built-in models are
``SEWResNet34Config`` and ``SpikformerConfig``.

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
