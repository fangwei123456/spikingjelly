SNN 分布式训练（DTensor / FSDP2）
=========================================

English version: :doc:`../en/distributed_training`

本教程介绍 ``spikingjelly.activation_based.distributed`` 中的分布式训练工具。它有两条阅读路线：

* 新手可以沿着 **Analyze -> Plan -> Apply** 和官方训练脚本使用。
* 高级用户在需要手工 mesh 维度、tensor-parallel roots、FSDP roots 或 pipeline 控制时，可以直接使用 ``SNNDistributedConfig``。

运行示例前，需要了解最小 PyTorch 分布式概念：``torchrun`` 会为每个 rank 启动一个进程，``world_size`` 是参与训练的 rank 数，``init_process_group`` 会创建 DeviceMesh、DTensor、DDP、FSDP2 和 pipeline schedule 使用的进程组。

为什么 SNN 分布式训练需要特殊处理
++++++++++++++++++++++++++++++++++

SNN 模块会在时间步之间携带神经元状态。分布式包装必须让这些状态和每个 rank 持有的 activation shard 保持一致。例如，Linear tensor parallelism 切分特征维，而 Conv/BN/neuron channel tensor parallelism 切分通道维。因此，有状态神经元应该只保存本地 shard 对应的状态，而不是悄悄复制完整全局状态。

Pipeline parallelism 还有另一个 SNN 特有问题：不同 microbatch 不能共享神经元状态。pipeline runtime 会在每个 stage 内的 microbatch 之间重置状态，避免一个 microbatch 的电压或其它状态泄漏到下一个 microbatch。

并行模式
++++++++

.. list-table::
    :header-rows: 1

    * - 模式
      - 适用目标
      - Mesh
      - 说明
    * - ``dp``
      - 简单吞吐扩展
      - 1D data mesh
      - 使用 DDP 风格复制；可选 ``ZeroRedundancyOptimizer``。
    * - ``tp``
      - 降低单 rank activation/state 显存
      - 1D tensor mesh
      - Linear TP 较稳定；Conv/BN 和 Spikformer TP 是实验性开关。
    * - ``fsdp2``
      - 参数、梯度、优化器状态分片
      - 1D data mesh
      - 使用 DTensor/FSDP2，是推荐的显存基线。
    * - ``fsdp2_tp``
      - 混合显存优化与模型并行
      - 2D ``(dp, tp)`` mesh
      - 推荐的混合路径；避开不支持的 DDP + TP 同步问题。
    * - ``pp``
      - stage 级显存压力或 pipeline 实验
      - pipeline ranks，可选 virtual stages
      - 使用专用 pipeline builder，不走统一 ``apply()`` 路径。

DeviceMesh 为 rank 提供名字和坐标。DTensor 记录 tensor 在 mesh 上的 placement。SNN helpers 用这两个概念让模型权重、梯度、优化器状态、activation 和神经元状态与所选策略对齐。

新手主线：Analyze -> Plan -> Apply
++++++++++++++++++++++++++++++++++

高层工作流使用包根公开接口：

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
        mode="fsdp2",
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

多进程启动使用同一份代码，但要在 ``torchrun`` 下运行，并在创建 distributed runtime 前初始化进程组：

.. code:: bash

    torchrun --nproc_per_node=4 train.py

内部工作流程
++++++++++++

高层路径保持较小的公开接口，内部实现拆到聚焦模块：

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

模型 adapter 只提供模型族 policy，例如 classifier roots、Conv/BN roots、Spikformer roots 和 FSDP shard roots。共享 eager config builder 是唯一把 ``mode + topology + policy + feature flags`` 展开成 ``SNNDistributedConfig`` 的位置。

官方训练脚本
++++++++++++

仓库提供 CIFAR10-DVS 训练入口：

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

常用模式选择：

* 先用 ``fsdp2`` 在 1D mesh 上降低显存。
* 当模型也适合 tensor parallelism 时，使用 ``fsdp2_tp --mesh-shape DP TP``。
* 只有明确需要无 FSDP2 的 tensor parallelism 时，才使用 ``tp``。
* stage 级实验通过专用 pipeline 路径使用 ``pp``。

高级用户：SNNDistributedConfig
++++++++++++++++++++++++++++++

高级用户仍可绕过 planner，通过 ``distributed.dtensor`` 兼容低层入口直接调用。这个路径适合需要精确控制 TP/FSDP roots 或手工 2D mesh 维度的场景。

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

低层路径会保持兼容，但除非需要直接控制 roots 或 mesh 维度，多数用户应优先使用 ``analyze`` / ``plan`` / ``apply``。

Pipeline Parallelism
++++++++++++++++++++

Pipeline parallelism 仍使用专用 builder，因为它需要 ``example_input`` 来构造 stage 并测量 cost。统一 ``apply()`` API 会有意拒绝 ``mode='pp'``。

pipeline 模块负责 stage partition、schedule selection、microbatch reset 和可选 stage-level memory optimization。可用控制包括 ``--pp-schedule``、``--pp-microbatches``、``--pp-virtual-stages``、``--pp-layout`` 和 ``--pp-delay-wgrad``。关键 SNN 不变量是 stage-local 神经元状态会在 microbatch 之间重置。

限制与故障排查
++++++++++++++

* ``DDP + TP`` 不支持，因为 DDP 状态同步会混合普通 ``Tensor`` 参数和 ``DTensor`` 参数。请使用 ``fsdp2_tp``。
* ``fsdp2_tp`` 应显式使用 2D mesh，例如 ``--mesh-shape 2 4``。
* Pipeline batch size 必须与选择的 microbatch 数兼容。
* 一些功能依赖可选 PyTorch API。``DTENSOR_AVAILABLE``、``FSDP2_AVAILABLE`` 和 ``PIPELINING_AVAILABLE`` 等 availability flags 在包根导出。
* DTensor 路径的输出在进入普通 loss 或 metric 代码前可能需要 materialization。``SNNDistributedRuntime.prepare_classification_output`` 会处理常见分类任务。

Benchmark 使用和结果解释
+++++++++++++++++++++++++

benchmark 脚本可用于 smoke test，也可在相同硬件、模型和 batch regime 下比较不同模式：

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

短 smoke run 只能证明启动、前向、反向、optimizer step、状态 reset 和正常退出。它不能证明扩展效率。若要讨论扩展性，应使用相同 benchmark regime 的更长运行，并同时报告吞吐和峰值显存。

Benchmark 附录
++++++++++++++

服务器实测结果（小网络 smoke benchmark）
+++++++++++++++++++++++++++++++++++++++++++++++++

以下数据来自单机多卡服务器（RTX 4090），网络为 ``CIFAR10DVSVGG``，后端为 ``inductor``，输入配置为 ``batch_size=2``、``T=10``，指标为短步数训练 benchmark。表中的 ``global_samples/s`` 统一表示整个分布式作业的全局吞吐。这个工作负载非常小，更多用于 smoke test 和显存趋势对比，不适合作为最终扩展效率结论。

.. list-table::
    :header-rows: 1

    * - 模式
      - GPU 数
      - ``step_ms``
      - ``global_samples/s``
      - ``peak_allocated_mb``
      - 备注
    * - ``none``
      - 1
      - 12.86
      - 155.52
      - 401.63
      - 单卡基线
    * - ``dp``
      - 2
      - 83.71
      - 47.78
      - 434.25
      - 纯 DDP，小 batch 下通信开销占主导
    * - ``dp`` + ``zero``
      - 2
      - 96.79
      - 41.33
      - 410.22
      - 纯 DDP + ``ZeroRedundancyOptimizer``
    * - ``tp``
      - 2
      - 86.58
      - 23.10
      - 308.88
      - 纯 TP，神经元按 shard 后特征/通道局部执行
    * - ``fsdp2``
      - 2
      - 97.11
      - 41.19
      - 400.61
      - 纯 FSDP2
    * - ``fsdp2_tp``
      - 4
      - 26.68
      - 149.91
      - 316.27
      - 推荐的 ``FSDP2 + TP`` 方案
    * - ``hybrid``（``DDP + TP``）
      - 4
      - -
      - -
      - -
      - 当前显式不支持；请改用 ``fsdp2_tp``

从这组小网络 smoke benchmark 可以看出：

* ``TP`` 和 ``FSDP2 + TP`` 都已经可以在标准神经元 ``backend='inductor'`` 下完成真实 SNN 训练 step。
* 显式 neuron shard 后，神经元状态会随特征/通道切分，而不再保持完整复制。
* 即使在很小的网络和 batch 上，``TP`` / ``FSDP2 + TP`` 也已经能带来可见的单卡显存下降。
* ``DDP + TP`` 目前仍不推荐，建议直接使用 ``fsdp2_tp``。

实验性 PP benchmark（服务器复测）
++++++++++++++++++++++++++++++++++++++++

当前 ``PP`` 已经支持：

* 基于 dry-run 实际耗时的 stage balance，而不是简单按层数均分；
* 自动选择更积极的 ``pp_microbatches``（优先选择 ``batch_size`` 的可整除值）；
* ``gpipe / 1f1b / interleaved / zero_bubble`` 多种调度；
* 显式 ``pp_layout`` 覆盖自动切分；
* 更轻量的 microbatch reset 逻辑，减少每次调度的遍历开销。

下面的结果来自重新在服务器上跑的 schedule 对比。它们更适合回答“当前哪种 PP 调度更值得默认推荐”，而不是把 ``PP`` 说成已经是吞吐主力。

``CIFAR10DVSVGG``，``backend='inductor'``，2 张 GPU，``batch_size=8``，``T=4``：

.. list-table::
    :header-rows: 1

    * - 调度
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

``spikformer_ti``，``backend='inductor'``，2 张 GPU，``batch_size=4``，``T=8``，``image_size=224``：

.. list-table::
    :header-rows: 1

    * - 调度
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

这组服务器复测说明：

* ``PP`` 与标准神经元 ``backend='inductor'`` 已经能够真实训练；
* 对 ``CIFAR10DVSVGG`` 这类小型卷积 SNN，``interleaved`` 目前是最好的默认调度，吞吐最好；``1f1b`` 的优势更多体现在显存；
* 对 ``spikformer_ti``，``interleaved`` 同样是当前最好的默认调度；如果叠加 ``memopt level=1``，可以把 ``peak_allocated_mb`` 从约 ``1.39 GB`` 压到约 ``0.54 GB``；
* ``zero_bubble`` 已经能在 ``CIFAR10DVSVGG`` 和 ``spikformer_ti`` 上功能跑通，但当前吞吐都还不占优；
* 对 ``spikformer_ti``，``zero_bubble + memopt level=1`` 现在也已经可用，并能把 ``peak_allocated_mb`` 压到约 ``0.48 GB``；
* 不过 ``zero_bubble`` 仍会伴随额外的 ``inductor`` 重编译告警，因此当前更适合手动实验和容量优先场景，而不是默认推荐。

Spikformer 与 memopt 组合结果
+++++++++++++++++++++++++++++

在更接近 ImageNet 训练设置的 ``spikformer_ti`` 上，``TP`` 和 ``FSDP2 + TP`` 也已经可以和 ``memopt level=1`` 结合使用。下面的实验使用：

* 模型：``spikformer_ti``
* 输入：``224x224``
* ``batch_size=4``
* ``T=8``
* 后端：``inductor``
* GPU：RTX 4090

.. list-table::
    :header-rows: 1

    * - 模式
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

可以看到：

* ``memopt level=1`` 与 ``none / dp / fsdp2 / tp / fsdp2_tp`` 都已经可以组合使用；
* ``tp / fsdp2_tp / pp`` 上更高 level 的 ``memopt``（``level >= 2``）现在也已经打通，做法是在 TP/FSDP2/PP 物化之前先完成 split-search；不过这类搜索开销很大，更适合离线调优或小规模 smoke 验证；
* 对 ``Spikformer`` 这类更大的 SNN，``TP`` / ``FSDP2 + TP`` 在 ``inductor`` 神经元下已经能明显降低单卡峰值显存；
* 再叠加 ``memopt level=1`` 后，``tp`` 与 ``fsdp2_tp`` 的单卡峰值显存都可以压到约 ``0.76 GB``；
* 这组 benchmark 里，``fsdp2_tp + memopt level=1`` 同时拿到了更低显存和更好的吞吐；
* ``dp + zero`` 是否优于纯 ``dp`` 取决于工作负载，在较大模型上更值得尝试。

推荐组合
+++++++++++++++++++++++

如果你的目标比较明确，可以按下面的经验规则选择：

* **吞吐优先，显存压力不大**：

  * 对小模型或单卡训练，先看 ``none``；
  * 对更大的分布式 workload，优先尝试 ``fsdp2`` 或 ``fsdp2_tp``；
  * ``dp + zero`` 可以作为纯数据并行路线的一个可选增强，但收益和 workload 强相关。

* **单卡显存优先，尤其是 ImageNet / Transformer 型 SNN**：

  * 优先尝试 ``tp + memopt level=1`` 或 ``fsdp2_tp + memopt level=1``；
  * 当前实测里，这两种组合都能把 ``Spikformer`` 的单卡峰值显存压到约 ``0.76 GB``。

* **希望在速度和显存之间取得折中**：

  * ``fsdp2_tp`` 仍然是最稳妥的主推荐；
  * 如果你的工作负载与这里的 ``Spikformer`` benchmark 接近，可以直接试 ``fsdp2_tp + memopt level=1``；
  * 如果显存已经足够，则保留 ``fsdp2_tp`` 而不开 ``memopt``，可以减少优化前处理时间。

* **只想要最省心、最稳妥的分布式训练入口**：

  * 从 ``dp`` 开始；
  * 如果要进一步扩展到更大模型，再迁移到 ``fsdp2`` 或 ``fsdp2_tp``。

如果你不想自己手工挑模式，现在训练脚本和 benchmark 也支持高层自动推荐器：

.. code:: bash

    torchrun --nproc_per_node=4 \
      spikingjelly/activation_based/examples/memopt/train_distributed.py \
      --data-dir /path/to/cifar10dvs \
      --distributed-mode auto \
      --prefer memory \
      --backend inductor \
      --batch-size 16

其中：

* ``--prefer speed`` 倾向于选择吞吐优先的组合；
* ``--prefer memory`` 倾向于选择单卡显存更低的组合；
* ``--prefer capacity`` 倾向于选择更容易放下大模型的组合（优先考虑 ``PP``）。

当 ``prefer=capacity`` 且环境允许时，自动推荐器会优先选择：

* ``mode=pp``
* ``pp_virtual_stages=2``
* ``pp_schedule=interleaved``
* ``memopt level=1``

``zero_bubble`` 仍然作为显式可选项保留在命令行里。它现在已经能稳定跑通，但当前默认仍建议优先使用更稳、更快的 ``interleaved``；``zero_bubble`` 更适合手动实验和容量优先场景。

如果显式指定了 ``--distributed-mode``，那么 ``prefer`` 仍然可以帮你补默认的 ``memopt`` / ``optimizer_sharding`` 等参数，但不会覆盖你手工指定的模式。

Benchmark 自动记录与对比
++++++++++++++++++++++++++++

``benchmark/benchmark_snn_distributed.py`` 现在会默认把结果追加到 ``benchmark/results/benchmark_snn_distributed.jsonl``，并自动和同配置的上一条记录做对比。新版记录会显式区分 benchmark 口径与 batch 语义，统一保存：

* ``benchmark_regime``：``throughput_weak_scaling`` / ``latency_strong_scaling`` / ``memory_capacity``
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

例如：

.. code:: bash

    torchrun --nproc_per_node=4 \
      benchmark/benchmark_snn_distributed.py \
      --mode auto \
      --prefer speed \
      --model spikformer_ti \
      --backend inductor \
      --batch-size 4 \
      --T 8

当前建议避免的组合：

* ``hybrid``（``DDP + TP``）：当前仍不支持；
* 在大尺寸 ``Spikformer`` 工作负载上直接使用高 level ``memopt``（``level >= 2``）做在线搜索：虽然功能上已经可用，但 ``optimize_ms`` 仍然很高，并且更容易触发 ``inductor`` 的额外重编译，建议先离线搜索、再固定策略。
