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

核心指标需要一起解读：

.. list-table::
    :header-rows: 1

    * - 指标
      - 含义
      - 比较方式
    * - ``global_samples/s``
      - 整个分布式作业的端到端吞吐。
      - 只有模型、后端、batch regime 和步数一致时才可比较；越高越好。
    * - ``peak_memory_mb``
      - 单个 rank 观测到的峰值显存。
      - 同一 workload 能正常完成时越低越好。
    * - ``step_ms``
      - warmup 后的单步延迟。
      - latency run 看越低越好；weak-scaling run 优先看吞吐。

Benchmark 附录
++++++++++++++

服务器实测结果（Triton，关闭 compile）
+++++++++++++++++++++++++++++++++++++++++++++++++

以下数据来自 ``g3``，一台 7 卡 RTX 4090 服务器；环境为 PyTorch ``2.7.1+cu118`` 和 Triton ``3.3.1``。benchmark 使用 ``backend='triton'``、``NCCL_P2P_DISABLE=1``、``TORCH_COMPILE_DISABLE=1``、``TORCHDYNAMO_DISABLE=1`` 和 ``memopt_level=0``。本轮没有启用 ``torch.compile`` 路径；这些表只考察分布式并行策略的影响，不比较 memory optimization rewrite 的作用。

所有结果都使用 ``benchmark_regime='throughput_weak_scaling'``。``global_samples/s`` 表示整个分布式作业的端到端吞吐，``peak_allocated_mb`` 表示所有 rank 中观测到的最大 CUDA allocation。

``CIFAR10DVSVGG``，``batch_size=2``，``T=10``：

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
      - 10.96
      - 182.40
      - 576.46
      - 单卡基线
    * - ``dp``
      - 2
      - 13.89
      - 287.89
      - 609.34
      - 纯 DDP weak scaling
    * - ``dp`` + ``zero``
      - 2
      - 16.54
      - 241.85
      - 609.34
      - DDP + ``ZeroRedundancyOptimizer``
    * - ``tp``
      - 2
      - 17.93
      - 111.57
      - 503.46
      - TP 降低单卡显存，但这组配置下吞吐下降
    * - ``fsdp2``
      - 2
      - 18.66
      - 214.37
      - 595.94
      - 参数、梯度和优化器状态分片
    * - ``fsdp2_tp``
      - 4
      - 30.21
      - 132.42
      - 510.52
      - ``(2, 2)`` mesh 上的 FSDP2 + TP
    * - ``hybrid``（``DDP + TP``）
      - 4
      - -
      - -
      - -
      - 当前显式不支持；请改用 ``fsdp2_tp``

在这个小型卷积 workload 上，``dp`` 的吞吐最好，因为单 rank 计算量较小，模型复制成本也低。``tp`` 和 ``fsdp2_tp`` 能降低峰值显存，但在这个规模下通信与分片执行开销超过了显存收益。

Pipeline Parallelism
++++++++++++++++++++

pipeline runtime 支持基于 cost 的 stage balance、自动 microbatch 选择、``gpipe`` / ``1f1b`` / ``interleaved`` / ``zero_bubble`` 调度、可选 virtual stages、手工 ``pp_layout`` 覆盖，以及 microbatch 之间的 stage-local 神经元状态 reset。

``CIFAR10DVSVGG``，``backend='triton'``，2 张 GPU，``batch_size=8``，``T=4``，``memopt_level=0``：

.. list-table::
    :header-rows: 1

    * - 调度
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

这组结果里，``interleaved`` 是吞吐最好的 PP 调度，``1f1b`` 的峰值显存最低。``zero_bubble`` 可以正常运行，但不是这个 workload 上最快的选项。

Spikformer 策略 benchmark
+++++++++++++++++++++++++

``spikformer_ti``，``backend='triton'``，``batch_size=4``，``T=8``，``image_size=224``，``memopt_level=0``：

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
      - 34.71
      - 115.22
      - 2290.53
      - 单卡基线
    * - ``dp``
      - 2
      - 38.87
      - 205.81
      - 2307.49
      - 本轮无 memopt benchmark 中吞吐最高
    * - ``dp`` + ``zero``
      - 2
      - 40.05
      - 199.77
      - 2307.49
      - 这组配置里 optimizer sharding 不降低 activation 峰值显存
    * - ``fsdp2``
      - 2
      - 50.09
      - 159.71
      - 2289.38
      - 这个短 benchmark 中吞吐低于 DP
    * - ``tp``
      - 2
      - 58.24
      - 68.68
      - 1557.10
      - 明显降低单卡显存
    * - ``fsdp2_tp``
      - 4
      - 90.75
      - 88.16
      - 1561.62
      - 混合路径，显存与纯 TP 接近

对 ``spikformer_ti``，不使用 memopt 时，tensor-parallel 模式已经能把单卡峰值显存从约 ``2.29 GB`` 降到约 ``1.56 GB``。代价是在这个短 weak-scaling run 里吞吐较低。当显存足够时，纯 ``dp`` 仍然是最强吞吐基线。

推荐组合
+++++++++++++++++++++++

如果你的目标比较明确，可以按下面的经验规则选择：

* **吞吐优先，显存压力不大**：

  * 先用 ``dp`` 做直接 weak scaling；
  * 如果优化器状态可能成为瓶颈，可以尝试 ``dp + zero``，但收益和 workload 强相关，需要实测；
  * 对小模型，分片开销很容易超过节省的收益。

* **单卡显存优先，尤其是 Transformer 型 SNN**：

  * activation 和神经元状态显存占主导时，优先尝试 ``tp``；
  * 如果还需要 FSDP2 风格的分片，再尝试 ``fsdp2_tp``，并显式使用 2D mesh，例如 ``--mesh-shape 2 2``。

* **pipeline 实验或 stage 级显存压力**：

  * 通过专用 pipeline runtime 使用 ``pp``；
  * 当前 CIFAR10DVSVGG PP benchmark 中，``interleaved`` 是吞吐最好的默认调度，``1f1b`` 是显存更低的调度。

* **只想要最省心、最稳妥的分布式训练入口**：

  * 从 ``dp`` 开始；
  * 只有当模型规模或显存曲线确实需要时，再迁移到 ``fsdp2``、``tp``、``fsdp2_tp`` 或 ``pp``。

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
