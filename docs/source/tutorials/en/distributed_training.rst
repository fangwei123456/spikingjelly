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

Server Benchmarks (small-network smoke benchmark)
+++++++++++++++++++++++++++++++++++++++++++++++++

The following numbers were collected on a multi-GPU RTX 4090 server with ``CIFAR10DVSVGG``, ``backend='inductor'``, ``batch_size=2``, and ``T=10`` using a short training-step benchmark. The ``global_samples/s`` column is the unified global throughput of the whole distributed job. This workload is intentionally tiny and should be read as a smoke benchmark plus a memory-trend probe rather than a definitive scaling study.

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
      - 12.86
      - 155.52
      - 401.63
      - single-GPU baseline
    * - ``dp``
      - 2
      - 83.71
      - 47.78
      - 434.25
      - pure DDP; communication dominates at this tiny batch size
    * - ``dp`` + ``zero``
      - 2
      - 96.79
      - 41.33
      - 410.22
      - pure DDP + ``ZeroRedundancyOptimizer``
    * - ``tp``
      - 2
      - 86.58
      - 23.10
      - 308.88
      - pure TP with neuron states kept on local feature/channel shards
    * - ``fsdp2``
      - 2
      - 97.11
      - 41.19
      - 400.61
      - pure FSDP2
    * - ``fsdp2_tp``
      - 4
      - 26.68
      - 149.91
      - 316.27
      - recommended ``FSDP2 + TP`` strategy
    * - ``hybrid`` (``DDP + TP``)
      - 4
      - -
      - -
      - -
      - explicitly unsupported for now; use ``fsdp2_tp`` instead

This small-network smoke benchmark shows that:

* both ``TP`` and ``FSDP2 + TP`` can now execute real SNN training steps with standard neurons running on ``backend='inductor'``;
* explicit neuron sharding keeps neuron states aligned with local feature/channel shards instead of fully replicating them;
* even at this tiny scale, ``TP`` / ``FSDP2 + TP`` already reduce per-GPU peak allocated memory;
* ``DDP + TP`` is still not recommended, and ``fsdp2_tp`` should be used instead.

Experimental PP benchmark (server rerun)
++++++++++++++++++++++++++++++++++++++++

``PP`` now includes:

* cost-aware stage balancing based on dry-run module timings rather than simple layer counts;
* a more aggressive automatic ``pp_microbatches`` heuristic that prefers values dividing ``batch_size`` cleanly;
* multiple schedules: ``gpipe``, ``1f1b``, ``interleaved``, and ``zero_bubble``;
* explicit ``pp_layout`` overrides for manual stage placement;
* lighter microbatch reset handling to avoid repeated full-module tree scans.

The following numbers come from rerunning the larger PP benchmarks on the server. They are more useful for deciding which schedule should be the default recommendation than for claiming that ``PP`` is already the primary throughput path.

``CIFAR10DVSVGG``, ``backend='inductor'``, 2 GPUs, ``batch_size=8``, ``T=4``:

.. list-table::
    :header-rows: 1

    * - Schedule
      - ``pp_virtual_stages``
      - ``memopt``
      - ``optimize_ms``
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
    * - ``gpipe``
      - 1
      - 0
      - 0.0
      - 93.70
      - 85.38
      - 507.84
    * - ``1f1b``
      - 1
      - 0
      - 0.0
      - 102.65
      - 77.93
      - 259.09
    * - ``interleaved``
      - 2
      - 0
      - 0.0
      - 87.63
      - 91.29
      - 361.45
    * - ``interleaved``
      - 2
      - 1
      - 1521.40
      - 84.39
      - 94.79
      - 361.45
    * - ``zero_bubble``
      - 2
      - 0
      - 0.0
      - 145.17
      - 55.11
      - 452.67
    * - ``zero_bubble``
      - 2
      - 1
      - 1535.38
      - 118.00
      - 67.80
      - 452.67

``spikformer_ti``, ``backend='inductor'``, 2 GPUs, ``batch_size=4``, ``T=8``, ``image_size=224``:

.. list-table::
    :header-rows: 1

    * - Schedule
      - ``pp_virtual_stages``
      - ``memopt``
      - ``optimize_ms``
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
    * - ``gpipe``
      - 1
      - 0
      - 0.0
      - 423.64
      - 9.44
      - 1286.03
    * - ``1f1b``
      - 1
      - 0
      - 0.0
      - 461.92
      - 8.66
      - 679.22
    * - ``interleaved``
      - 2
      - 0
      - 0.0
      - 394.63
      - 10.14
      - 1389.71
    * - ``interleaved``
      - 2
      - 1
      - 112.83
      - 423.73
      - 9.44
      - 541.91
    * - ``zero_bubble``
      - 2
      - 0
      - 0.0
      - 455.79
      - 8.78
      - 1356.73
    * - ``zero_bubble``
      - 2
      - 1
      - 164.35
      - 473.41
      - 8.45
      - 483.31

These rerun results show that:

* ``PP`` already works with standard neurons running on ``backend='inductor'``;
* for a small convolutional SNN such as ``CIFAR10DVSVGG``, ``interleaved`` is currently the best default schedule for throughput, while ``1f1b`` is more attractive when memory is the main concern;
* for ``spikformer_ti``, ``interleaved`` is also the strongest default schedule, and adding ``memopt level=1`` can reduce ``peak_allocated_mb`` from about ``1.39 GB`` to about ``0.54 GB``;
* ``zero_bubble`` now runs successfully on both ``CIFAR10DVSVGG`` and ``spikformer_ti``, but it is still not the strongest throughput option on either workload;
* for ``spikformer_ti``, ``zero_bubble + memopt level=1`` now also works and reduces ``peak_allocated_mb`` to about ``0.48 GB``;
* however, ``zero_bubble`` still comes with extra ``inductor`` recompilation warnings, so it is best viewed as a manual experimental or capacity-oriented option rather than the default recommendation today.

Spikformer + memopt Results
++++++++++++++++++++++++++++

On ``spikformer_ti`` in a more ImageNet-like setting, ``TP`` and ``FSDP2 + TP`` can now also be combined with ``memopt level=1``. The following experiment uses:

* model: ``spikformer_ti``
* input resolution: ``224x224``
* ``batch_size=4``
* ``T=8``
* backend: ``inductor``
* GPU: RTX 4090

.. list-table::
    :header-rows: 1

    * - Mode
      - ``optimizer_sharding``
      - ``memopt``
      - ``optimize_ms``
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
    * - ``none``
      - ``none``
      - ``0``
      - 0.0
      - 36.70
      - 109.00
      - 2070.34
    * - ``none``
      - ``none``
      - ``1``
      - 26852.97
      - 57.35
      - 69.74
      - 1298.16
    * - ``dp``
      - ``none``
      - ``0``
      - 0.0
      - 126.56
      - 63.21
      - 2070.93
    * - ``dp``
      - ``zero``
      - ``0``
      - 0.0
      - 122.28
      - 65.42
      - 2055.70
    * - ``dp``
      - ``none``
      - ``1``
      - 22591.25
      - 134.48
      - 59.49
      - 1315.71
    * - ``dp``
      - ``zero``
      - ``1``
      - 23030.79
      - 149.21
      - 53.61
      - 1297.59
    * - ``fsdp2``
      - ``none``
      - ``0``
      - 0.0
      - 111.08
      - 72.02
      - 2033.86
    * - ``fsdp2``
      - ``none``
      - ``1``
      - 22919.87
      - 132.65
      - 60.31
      - 1272.13
    * - ``tp``
      - ``none``
      - ``0``
      - 0.0
      - 196.41
      - 20.37
      - 1321.38
    * - ``tp``
      - ``none``
      - ``1``
      - 26913.14
      - 173.65
      - 23.03
      - 767.51
    * - ``fsdp2_tp``
      - ``none``
      - ``0``
      - 0.0
      - 131.90
      - 60.65
      - 1319.68
    * - ``fsdp2_tp``
      - ``none``
      - ``1``
      - 26403.47
      - 103.95
      - 76.96
      - 761.26

These numbers highlight that:

* ``memopt level=1`` now works with ``none / dp / fsdp2 / tp / fsdp2_tp``;
* higher ``memopt`` levels on ``tp / fsdp2_tp / pp`` (``level >= 2``) are now functional as well by running split-search before TP/FSDP2/PP materialization; however, this path is still expensive and is better suited to offline tuning or smoke validation than to frequent online retuning;
* for larger SNNs such as ``Spikformer``, ``TP`` / ``FSDP2 + TP`` already give a clear per-GPU memory reduction when the neurons use the standard ``inductor`` backend;
* adding ``memopt level=1`` reduces ``tp`` / ``fsdp2_tp`` further to about ``0.76 GB`` per GPU in this benchmark;
* in this particular benchmark, ``fsdp2_tp + memopt level=1`` improves both memory usage and throughput;
* whether ``dp + zero`` beats plain ``dp`` remains workload-dependent, but it is still a useful option when optimizer state becomes meaningful.

Recommended Combinations
++++++++++++++++++++++++

If you already know your main objective, the following rules of thumb work well:

* **Throughput first, memory is not the main bottleneck**:

  * for tiny models or single-GPU work, start by checking whether ``none`` is already sufficient;
  * for larger distributed workloads, prefer ``fsdp2`` or ``fsdp2_tp``;
  * ``dp + zero`` is a useful pure-DP enhancement, but its benefit depends strongly on the workload.

* **Per-GPU memory first, especially for ImageNet-scale or Transformer-style SNNs**:

  * prefer ``tp + memopt level=1`` or ``fsdp2_tp + memopt level=1``;
  * in the current measurements, both combinations reduce ``Spikformer`` peak per-GPU memory to about ``0.76 GB``.

* **Balanced speed/memory tradeoff**:

  * ``fsdp2_tp`` is currently the most balanced primary recommendation;
  * if your workload is close to the ``Spikformer`` benchmark above, it is worth trying ``fsdp2_tp + memopt level=1`` directly;
  * if memory is already sufficient, keep ``fsdp2_tp`` without ``memopt`` to avoid the extra optimization pass.

* **Safest and simplest distributed entry point**:

  * begin with ``dp``;
  * move to ``fsdp2`` or ``fsdp2_tp`` only when you need to scale to larger models or tighter per-GPU memory.

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
