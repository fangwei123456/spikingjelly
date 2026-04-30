Distributed SNN Training (DTensor / FSDP2)
==========================================

中文版： :doc:`../cn/distributed_training`

This tutorial introduces the experimental distributed-training helpers in ``spikingjelly.activation_based.distributed``. The current implementation focuses on:

* `DP`: conventional data parallelism
* `TP`: simple SNN-oriented tensor parallelism
* `FSDP2`: DTensor-based parameter, gradient, and optimizer-state sharding
* `FSDP2 + TP`: the recommended hybrid distributed strategy
* `PP`: experimental pipeline parallelism (currently implemented first for ``CIFAR10DVSVGG`` and ``Spikformer``)

In contrast, the traditional ``DDP + TP`` combination still runs into ``Tensor`` / ``DTensor`` synchronization issues in the current PyTorch version, so this implementation reports an actionable error and recommends ``FSDP2 + TP`` instead.

Quick Start
++++++++++++++++++++++++

The low-level entry is :func:`configure_snn_distributed <spikingjelly.activation_based.distributed.configure_snn_distributed>`. High-level helpers include:

* :func:`configure_cifar10dvs_vgg_distributed <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_distributed>`
* :func:`configure_cifar10dvs_vgg_fsdp2 <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_fsdp2>`
* :func:`configure_cifar10dvs_vgg_pipeline <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_pipeline>`
* :func:`configure_spikformer_pipeline <spikingjelly.activation_based.distributed.configure_spikformer_pipeline>`

For example, pure FSDP2 on ``CIFAR10DVSVGG``:

.. code:: python

    from spikingjelly.activation_based.distributed import configure_cifar10dvs_vgg_fsdp2
    from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG

    model = CIFAR10DVSVGG(dropout=0.0, backend='inductor')
    model, mesh, analysis = configure_cifar10dvs_vgg_fsdp2(
        model,
        device_type='cuda',
        mesh_shape=(world_size,),
        enable_classifier_tensor_parallel=False,
        enable_experimental_conv_tensor_parallel=False,
    )

To enable ``FSDP2 + TP``, use a 2D mesh:

.. code:: python

    model, mesh, analysis = configure_cifar10dvs_vgg_fsdp2(
        model,
        device_type='cuda',
        mesh_shape=(2, 2),   # (dp, tp)
        enable_classifier_tensor_parallel=True,
        enable_experimental_conv_tensor_parallel=True,
        dp_mesh_dim=0,
        tp_mesh_dim=1,
    )

Training Script
++++++++++++++++++++++++

The repository now includes a real distributed training entry:

* ``spikingjelly/activation_based/examples/memopt/train_distributed.py``

Example:

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

Supported modes in the script:

* ``none``
* ``dp``
* ``tp``
* ``fsdp2``
* ``fsdp2_tp``
* ``pp``: currently exposed as an experimental training path aimed first at smoke benchmarks and structural validation

``PP`` also exposes a set of scheduling and layout controls that move it closer to Megatron-style pipeline tuning:

* ``--pp-schedule gpipe``: the simplest GPipe schedule;
* ``--pp-schedule 1f1b``: standard 1F1B;
* ``--pp-schedule interleaved``: interleaved / VPP-style scheduling;
* ``--pp-schedule zero_bubble``: an experimental zero-bubble schedule driven by delayed ``wgrad``;
* ``--pp-virtual-stages N``: keep ``N`` virtual chunks on each physical stage;
* ``--pp-layout``: explicitly describe the contiguous logical-stage split, for example ``1|2|2|1``;
* ``--pp-delay-wgrad``: explicitly enable delayed-``wgrad`` style optimization when the chosen schedule supports it.

The historical ``hybrid`` (``DDP + TP``) combination is still unsupported and is no longer exposed in the script; use ``fsdp2_tp`` instead.

If you want to further shard optimizer state on the pure ``dp`` path, you can also enable ``ZeroRedundancyOptimizer``:

.. code:: bash

    torchrun --nproc_per_node=2 \
      benchmark/benchmark_snn_distributed.py \
      --model cifar10dvs_vgg \
      --mode dp \
      --optimizer-sharding zero \
      --backend inductor \
      --batch-size 2 \
      --T 10

Current Scope
++++++++++++++++++++++++

1. Linear layers use PyTorch's official tensor-parallel APIs.
2. Element-wise spiking neurons now explicitly follow upstream sharding:

   * for ``[T, N, C]`` activations, they shard along the last feature dimension ``C``;
   * for ``[T, N, C, H, W]`` activations, they shard along the channel dimension ``C``.

   In other words, the neuron state ``v`` now only keeps the local shard rather than a full replicated global state.
3. The ``Conv + BN + Neuron`` backbone in ``CIFAR10DVSVGG`` supports experimental channel tensor parallelism.
4. Under ``FSDP2 + TP``, the current implementation prioritizes FSDP2 sharding on ``features``; when the ``classifier`` is already tensor-parallelized, it is not additionally root-sharded, avoiding repeated cross-mesh sharding.
5. Traditional ``hybrid`` (that is, ``DDP + TP``) is explicitly unsupported for now and will recommend ``fsdp2_tp`` instead.
6. ``PP`` currently uses manual stage splitting rather than ``torch.export`` full-graph partitioning, which makes it compatible with standard spiking neurons that mutate internal state.
7. ``PP`` explicitly resets neuron state between microbatches inside each stage so that different microbatches do not leak state into each other.

Server Benchmarks (small-network smoke benchmark)
++++++++++++++++++++++++

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
+++++++++++++++++++++++

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
++++++++++++++++++++++++

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
+++++++++++++++++++++++++++++++++++++++++

``benchmark/benchmark_snn_distributed.py`` now appends results to ``benchmark/results/benchmark_snn_distributed.jsonl`` by default and automatically compares each run against the most recent earlier run with the same configuration. Each record stores:

* ``global_samples_per_second``
* ``peak_allocated_mb``
* ``optimize_ms``
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
* running high-level ``memopt`` (``level >= 2``) online on large ``Spikformer``-like workloads: it now works functionally, but the search cost is still high and it is more likely to trigger extra `inductor` recompiles, so it is best treated as an offline tuning workflow for now.
