Distributed SNN Training (DTensor / FSDP2)
==========================================

中文版： :doc:`../cn/distributed_training`

This tutorial explains the distributed-training helpers in
``spikingjelly.activation_based.distributed``. It is written for two reading
paths:

* New users can follow **Analyze -> Plan -> Apply** and the official training
  script.
* Advanced users can use ``SNNDistributedConfig`` directly when they need manual
  mesh dimensions, tensor-parallel roots, FSDP roots, or pipeline controls.

Before running the examples, you should know the minimum PyTorch distributed
vocabulary: ``torchrun`` starts one process per rank, ``world_size`` is the
number of participating ranks, and ``init_process_group`` creates the process
group used by DeviceMesh, DTensor, DDP, FSDP2, and pipeline schedules.

Why SNN Distributed Training Needs Special Handling
++++++++++++++++++++++++++++++++++++++++++++++++++++

SNN modules carry neuron state across timesteps. A distributed wrapper must keep
that state consistent with the activation shard owned by each rank. For example,
Linear tensor parallelism shards the feature dimension, while Conv/BN/neuron
channel tensor parallelism shards the channel dimension. Stateful neurons should
therefore keep local state for the local shard instead of silently replicating a
full global state.

Pipeline parallelism adds another SNN-specific concern: microbatches must not
share neuron state. The pipeline runtime resets state inside each stage between
microbatches, so one microbatch cannot leak voltage or other state into the next.

Parallel Modes
++++++++++++++

.. list-table::
    :header-rows: 1

    * - Mode
      - Best fit
      - Mesh
      - Notes
    * - ``dp``
      - Simple throughput scaling
      - 1D data mesh
      - Uses DDP-style replication. ``ZeroRedundancyOptimizer`` is optional.
    * - ``tp``
      - Reducing per-rank activation/state memory
      - 1D tensor mesh
      - Linear TP is stable; Conv/BN and Spikformer TP are experimental flags.
    * - ``fsdp2``
      - Parameter, gradient, and optimizer-state sharding
      - 1D data mesh
      - Uses DTensor/FSDP2 and is the recommended memory baseline.
    * - ``fsdp2_tp``
      - Hybrid memory reduction and model parallelism
      - 2D ``(dp, tp)`` mesh
      - Recommended hybrid path. Avoids unsupported DDP + TP synchronization.
    * - ``pp``
      - Stage-level memory pressure or pipeline experiments
      - pipeline ranks, optional virtual stages
      - Uses dedicated pipeline builders, not the unified ``apply()`` path.

DeviceMesh gives names and coordinates to the ranks. DTensor records how a
tensor is placed on that mesh. The SNN helpers use those two concepts to keep
model weights, gradients, optimizer state, activations, and neuron state aligned
with the selected strategy.

Beginner Path: Analyze -> Plan -> Apply
+++++++++++++++++++++++++++++++++++++++

Use the public package root for the high-level workflow:

.. code:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from spikingjelly.activation_based import distributed as sjdist
    from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG

    model = CIFAR10DVSVGG(dropout=0.0, backend="torch")
    dataset = TensorDataset(
        torch.randn(4, 2, 2, 48, 48),
        torch.tensor([0, 1, 2, 3]),
    )

    analysis = sjdist.analyze(model, model_family="cifar10dvs_vgg")
    plan = sjdist.plan(
        analysis=analysis,
        objective="memory",
        topology={"dp": 1},
        backend="torch",
        batch_size=2,
        model_family="cifar10dvs_vgg",
        mode="none",
        features=sjdist.DistributedFeatureSet(
            allow_experimental_conv_tp=False,
        ),
    )
    runtime = sjdist.apply(model=model, plan=plan, device_type="cpu")

    optimizer = runtime.build_optimizer(torch.optim.SGD, lr=1e-3)
    loader = runtime.prepare_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    criterion = nn.CrossEntropyLoss()

    runtime.model.train()
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = runtime.model(images.float())
        logits, labels = runtime.prepare_classification_output(logits, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        runtime.reset_state()

A multi-process launch uses the same code, but it must run under ``torchrun`` and
initialize the process group before creating the distributed runtime:

.. code:: bash

    torchrun --nproc_per_node=4 train.py

Internal Workflow
+++++++++++++++++

The high-level path keeps the public interface small while the implementation is
split into focused modules:

.. code:: text

    analyze(model) -> analysis.py scans stateful modules and TP candidates
        |
        v
    plan(...) -> planner.py chooses mode, mesh, roots, and notes
        |
        v
    apply(...) -> api.py selects an adapter when a model family needs one
        |
        v
    build_eager_config(...) -> execution.py assembles SNNDistributedConfig
        |
        v
    configure_snn_distributed(...) -> TP, FSDP2, or DDP modules are applied

Model adapters only provide model-family policy such as classifier roots,
Conv/BN roots, Spikformer roots, and FSDP shard roots. The shared eager config
builder is the single place that expands ``mode + topology + policy + feature
flags`` into ``SNNDistributedConfig``.

Official Training Script
++++++++++++++++++++++++

The repository includes a CIFAR10-DVS training entry:

.. code:: bash

    torchrun --nproc_per_node=4 \
      spikingjelly/activation_based/examples/memopt/train_distributed.py \
      --data-dir /path/to/cifar10dvs \
      --distributed-mode fsdp2_tp \
      --mesh-shape 2 2 \
      --backend inductor \
      --batch-size 16 \
      --epochs 1 \
      --print-summary

Common mode choices:

* Start with ``fsdp2`` for memory reduction on a 1D mesh.
* Use ``fsdp2_tp --mesh-shape DP TP`` when the model also benefits from tensor
  parallelism.
* Use ``tp`` only when you explicitly want tensor parallelism without FSDP2.
* Use ``pp`` through the dedicated pipeline path for stage-level experiments.

Advanced Path: SNNDistributedConfig
+++++++++++++++++++++++++++++++++++

Advanced users can still bypass the planner and call the compatibility low-level
entry through ``distributed.dtensor``. This path is useful when you need exact
TP/FSDP roots or manual 2D mesh dimensions.

.. code:: python

    from spikingjelly.activation_based.distributed.dtensor import (
        SNNDistributedConfig,
        configure_snn_distributed,
    )

    model, mesh, analysis = configure_snn_distributed(
        model,
        SNNDistributedConfig(
            device_type="cuda",
            mesh_shape=(2, 2),
            enable_fsdp2=True,
            fsdp_shard_roots=["features"],
            fsdp_shard_module_root=False,
            tensor_parallel_roots=["classifier"],
            auto_tensor_parallel=True,
            experimental_conv_tensor_parallel=True,
            conv_tensor_parallel_roots=["features"],
            dp_mesh_dim=0,
            tp_mesh_dim=1,
        ),
    )

This low-level path is stable for compatibility, but most users should prefer
``analyze`` / ``plan`` / ``apply`` unless they need direct control over roots or
mesh dimensions.

Pipeline Parallelism
++++++++++++++++++++

Pipeline parallelism still uses dedicated builders because it requires an
``example_input`` for stage construction and cost measurement. The unified
``apply()`` API intentionally rejects ``mode='pp'``.

The pipeline modules own stage partitioning, schedule selection, microbatch
reset, and optional stage-level memory optimization. Supported controls include
``--pp-schedule``, ``--pp-microbatches``, ``--pp-virtual-stages``,
``--pp-layout``, and ``--pp-delay-wgrad``. The important SNN invariant is that
stage-local neuron state is reset between microbatches.

Limits and Troubleshooting
++++++++++++++++++++++++++

* ``DDP + TP`` is not supported because DDP state synchronization mixes regular
  ``Tensor`` parameters and ``DTensor`` parameters. Use ``fsdp2_tp`` instead.
* ``fsdp2_tp`` should use an explicit 2D mesh such as ``--mesh-shape 2 4``.
* Pipeline batch size must be compatible with the selected microbatch count.
* Some features depend on optional PyTorch APIs. Availability flags such as
  ``DTENSOR_AVAILABLE``, ``FSDP2_AVAILABLE``, and ``PIPELINING_AVAILABLE`` are
  exported at the package root.
* Outputs from DTensor paths may need materialization before ordinary loss or
  metric code. ``SNNDistributedRuntime.prepare_classification_output`` handles
  the common classification case.

Benchmark Usage and Result Interpretation
+++++++++++++++++++++++++++++++++++++++++

Use the benchmark script for smoke tests and for comparing modes under the same
hardware, model, and batch regime:

.. code:: bash

    torchrun --nproc_per_node=4 \
      benchmark/benchmark_snn_distributed.py \
      --model cifar10dvs_vgg \
      --mode fsdp2_tp \
      --mesh-shape 2 2 \
      --backend torch \
      --steps 2 \
      --warmup 1 \
      --batch-size 2 \
      --T 4

A short smoke run proves startup, forward, backward, optimizer step, state reset,
and clean shutdown. It does not prove scaling efficiency. For scaling claims,
compare longer runs with identical benchmark regimes and report both throughput
and peak memory.

Read the headline metrics together:

.. list-table::
    :header-rows: 1

    * - Metric
      - Meaning
      - Compare by
    * - ``global_samples/s``
      - End-to-end throughput for the whole distributed job.
      - Higher is better only under the same model, backend, batch regime, and step count.
    * - ``peak_memory_mb``
      - Peak memory observed on a rank.
      - Lower is better when the run still completes the same workload.
    * - ``step_ms``
      - Per-step latency after warmup.
      - Lower is better for latency runs; use throughput for weak-scaling runs.

Benchmark Appendix
++++++++++++++++++

Server Benchmarks (Triton, compile disabled)
+++++++++++++++++++++++++++++++++++++++++++++++++

The following numbers were collected on ``g3``, a 7-GPU RTX 4090 server, with PyTorch ``2.7.1+cu118`` and Triton ``3.3.1``. The benchmark used ``backend='triton'``, ``NCCL_P2P_DISABLE=1``, ``TORCH_COMPILE_DISABLE=1``, ``TORCHDYNAMO_DISABLE=1``, and ``memopt_level=0``. No ``torch.compile`` path was enabled. The tables therefore focus on the effect of distributed parallel strategies, not memory-optimization rewrites.

All rows use ``benchmark_regime='throughput_weak_scaling'``. ``global_samples/s`` is the end-to-end throughput of the whole distributed job, and ``peak_allocated_mb`` is the maximum CUDA allocation observed on any rank.

``CIFAR10DVSVGG``, ``batch_size=2``, ``T=10``:

.. list-table::
    :header-rows: 1

    * - Mode
      - #GPUs
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
      - Notes
    * - ``none``
      - 1
      - 10.96
      - 182.40
      - 576.46
      - single-GPU baseline
    * - ``dp``
      - 2
      - 13.89
      - 287.89
      - 609.34
      - pure DDP weak scaling
    * - ``dp`` + ``zero``
      - 2
      - 16.54
      - 241.85
      - 609.34
      - DDP with ``ZeroRedundancyOptimizer``
    * - ``tp``
      - 2
      - 17.93
      - 111.57
      - 503.46
      - tensor parallelism reduces per-GPU memory but lowers throughput here
    * - ``fsdp2``
      - 2
      - 18.66
      - 214.37
      - 595.94
      - parameter/gradient/optimizer-state sharding
    * - ``fsdp2_tp``
      - 4
      - 30.21
      - 132.42
      - 510.52
      - hybrid FSDP2 + TP on a ``(2, 2)`` mesh
    * - ``hybrid`` (``DDP + TP``)
      - 4
      - -
      - -
      - -
      - explicitly unsupported; use ``fsdp2_tp`` instead

On this small convolutional workload, ``dp`` gives the best throughput because the per-rank compute is small and the model is easy to replicate. ``tp`` and ``fsdp2_tp`` reduce peak memory, but their communication and sharded execution overhead outweigh the memory benefit at this scale.

Pipeline Parallelism
++++++++++++++++++++

The pipeline runtime supports cost-aware stage balancing, automatic microbatch selection, ``gpipe`` / ``1f1b`` / ``interleaved`` / ``zero_bubble`` schedules, optional virtual stages, manual ``pp_layout`` overrides, and stage-local neuron-state reset between microbatches.

``CIFAR10DVSVGG``, ``backend='triton'``, 2 GPUs, ``batch_size=8``, ``T=4``, ``memopt_level=0``:

.. list-table::
    :header-rows: 1

    * - Schedule
      - ``pp_virtual_stages``
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
    * - ``gpipe``
      - 1
      - 64.36
      - 124.31
      - 566.74
    * - ``1f1b``
      - 1
      - 64.03
      - 124.95
      - 398.76
    * - ``interleaved``
      - 2
      - 46.11
      - 173.48
      - 466.24
    * - ``zero_bubble``
      - 2
      - 57.79
      - 138.44
      - 475.04

In this run, ``interleaved`` is the best PP schedule for throughput, while ``1f1b`` uses the least peak memory. ``zero_bubble`` runs successfully but is not the fastest option for this workload.

Spikformer Strategy Benchmark
+++++++++++++++++++++++++++++

``spikformer_ti``, ``backend='triton'``, ``batch_size=4``, ``T=8``, ``image_size=224``, ``memopt_level=0``:

.. list-table::
    :header-rows: 1

    * - Mode
      - #GPUs
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
      - Notes
    * - ``none``
      - 1
      - 34.71
      - 115.22
      - 2290.53
      - single-GPU baseline
    * - ``dp``
      - 2
      - 38.87
      - 205.81
      - 2307.49
      - best throughput in this no-memopt run
    * - ``dp`` + ``zero``
      - 2
      - 40.05
      - 199.77
      - 2307.49
      - optimizer sharding does not reduce peak activation memory here
    * - ``fsdp2``
      - 2
      - 50.09
      - 159.71
      - 2289.38
      - lower throughput than DP for this short run
    * - ``tp``
      - 2
      - 58.24
      - 68.68
      - 1557.10
      - clear per-GPU memory reduction
    * - ``fsdp2_tp``
      - 4
      - 90.75
      - 88.16
      - 1561.62
      - hybrid path with similar memory to pure TP

For ``spikformer_ti``, tensor-parallel modes reduce per-GPU peak allocation from about ``2.29 GB`` to about ``1.56 GB`` without using memopt. The cost is lower throughput in this short weak-scaling run. Plain ``dp`` remains the strongest throughput baseline when memory is sufficient.

Recommended Combinations
++++++++++++++++++++++++

If you already know your main objective, the following rules of thumb work well:

* **Throughput first, memory is not the main bottleneck**:

  * start with ``dp`` for straightforward weak scaling;
  * use ``dp + zero`` when optimizer state is expected to matter, but benchmark it because the benefit is workload-dependent;
  * for small models, sharding can easily cost more than it saves.

* **Per-GPU memory first, especially for Transformer-style SNNs**:

  * try ``tp`` when activation and neuron-state memory dominate;
  * use ``fsdp2_tp`` when you also need FSDP2-style sharding, but keep an explicit 2D mesh such as ``--mesh-shape 2 2``.

* **Pipeline experiments or stage-level memory pressure**:

  * use ``pp`` through the dedicated pipeline runtime;
  * in the current CIFAR10DVSVGG PP benchmark, ``interleaved`` is the best throughput default and ``1f1b`` is the lower-memory schedule.

* **Safest and simplest distributed entry point**:

  * begin with ``dp``;
  * move to ``fsdp2``, ``tp``, ``fsdp2_tp``, or ``pp`` only when the model size or memory profile justifies the extra machinery.

If you do not want to hand-pick the mode yourself, the training script and benchmark now also expose a high-level recommender:

.. code:: bash

    torchrun --nproc_per_node=4 \
      spikingjelly/activation_based/examples/memopt/train_distributed.py \
      --data-dir /path/to/cifar10dvs \
      --distributed-mode auto \
      --prefer memory \
      --backend inductor \
      --batch-size 16

The current high-level intents are:

* ``--prefer speed`` for throughput-oriented defaults,
* ``--prefer memory`` for lower per-GPU memory defaults,
* ``--prefer capacity`` for configurations that are more likely to fit larger models, typically prioritizing ``PP``.

When ``prefer=capacity`` and the environment supports it, the auto recommender now prefers:

* ``mode=pp``
* ``pp_virtual_stages=2``
* ``pp_schedule=interleaved``
* ``memopt level=1``

``zero_bubble`` still remains available as an explicit command-line option. It now runs stably, but the default recommendation still prefers the faster and more predictable ``interleaved`` schedule; ``zero_bubble`` is better treated as a manual experimental or capacity-oriented tuning path.

If you explicitly set ``--distributed-mode``, the ``prefer`` hint can still fill in defaults such as ``memopt`` or ``optimizer_sharding``, but it will not override the manually selected mode.

Automatic Benchmark Logging and Comparison
++++++++++++++++++++++++++++++++++++++++++

``benchmark/benchmark_snn_distributed.py`` now appends results to ``benchmark/results/benchmark_snn_distributed.jsonl`` by default and automatically compares each run against the most recent earlier run with the same configuration. The newer records also make the benchmark regime and batch semantics explicit. Each record stores:

* ``benchmark_regime``: ``throughput_weak_scaling`` / ``latency_strong_scaling`` / ``memory_capacity``
* ``global_batch_size``
* ``per_rank_batch_size``
* ``data_replicas``
* ``pp_memopt_stages``
* ``step_latency_ms``
* ``global_throughput_sps``
* ``per_device_throughput_sps``
* ``peak_allocated_mb``
* ``optimize_ms``
* ``forward_ms``
* ``backward_ms``
* ``optimizer_ms``
* ``reset_ms``
* ``materialize_ms``
* ``tp_all_reduce_calls``
* ``tp_all_reduce_mb``
* ``warning_count``
* ``recompile_count``
* ``graph_break_count``

For example:

.. code:: bash

    torchrun --nproc_per_node=4 \
      benchmark/benchmark_snn_distributed.py \
      --mode auto \
      --prefer speed \
      --model spikformer_ti \
      --backend inductor \
      --batch-size 4 \
      --T 8

Combinations that should still be avoided for now:

* ``hybrid`` (``DDP + TP``): still unsupported;
* running high-level ``memopt`` (``level >= 2``) online on large ``Spikformer``-like workloads: it now works functionally, but the search cost is still high and it is more likely to trigger extra ``inductor`` recompiles, so it is best treated as an offline tuning workflow for now.
