SNN 分布式训练（DTensor / FSDP2）
=========================================

English version: :doc:`../en/distributed_training`

本教程介绍 ``spikingjelly.activation_based.distributed`` 中新增的实验性分布式训练工具。当前实现重点支持：

* `DP`：常规数据并行
* `TP`：面向 SNN 的简单 tensor parallel
* `FSDP2`：基于 DTensor 的参数、梯度与优化器状态分片
* `FSDP2 + TP`：推荐的混合分布式训练方案
* `PP`：实验性的 pipeline parallel（当前先支持 ``CIFAR10DVSVGG`` 和 ``Spikformer``）

其中，传统的 ``DDP + TP`` 组合在当前 PyTorch 版本下仍会在参数同步阶段遇到 ``Tensor`` / ``DTensor`` 混用问题，因此本实现会直接提示用户改用 ``FSDP2 + TP``。

快速开始
++++++++++++++++++++++++

低层入口是 :func:`configure_snn_distributed <spikingjelly.activation_based.distributed.configure_snn_distributed>`，高层入口包括：

* :func:`configure_cifar10dvs_vgg_distributed <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_distributed>`
* :func:`configure_cifar10dvs_vgg_fsdp2 <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_fsdp2>`
* :func:`configure_cifar10dvs_vgg_pipeline <spikingjelly.activation_based.distributed.configure_cifar10dvs_vgg_pipeline>`
* :func:`configure_spikformer_pipeline <spikingjelly.activation_based.distributed.configure_spikformer_pipeline>`

例如，对 ``CIFAR10DVSVGG`` 启用纯 FSDP2：

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

若想启用 ``FSDP2 + TP``，可使用 2D mesh：

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

训练脚本
++++++++++++++++++++++++

仓库中提供了一个真实训练入口：

* ``spikingjelly/activation_based/examples/memopt/train_distributed.py``

示例：

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

该脚本支持的模式有：

* ``none``
* ``dp``
* ``tp``
* ``fsdp2``
* ``fsdp2_tp``
* ``pp``：当前是实验性训练入口，优先面向 smoke benchmark 和结构验证

``PP`` 还支持一组更接近 Megatron 风格的调度与布局参数：

* ``--pp-schedule gpipe``：最简单的 GPipe 调度；
* ``--pp-schedule 1f1b``：标准 1F1B；
* ``--pp-schedule interleaved``：interleaved / VPP 风格调度；
* ``--pp-schedule zero_bubble``：基于 delayed-``wgrad`` 的实验性 zero-bubble 调度；
* ``--pp-virtual-stages N``：每个物理 stage 持有 ``N`` 个虚拟 chunk；
* ``--pp-layout``：显式指定逻辑 stage 的连续切分，例如 ``1|2|2|1``；
* ``--pp-delay-wgrad``：在可用调度下显式打开 delayed-``wgrad`` 风格优化。

历史上的 ``hybrid``（``DDP + TP``）组合当前仍不支持，也没有在脚本里继续暴露；推荐直接使用 ``fsdp2_tp``。

如果希望在纯 ``dp`` 路径上进一步压缩优化器状态，还可以启用 ``ZeroRedundancyOptimizer``：

.. code:: bash

    torchrun --nproc_per_node=2 \
      benchmark/benchmark_snn_distributed.py \
      --model cifar10dvs_vgg \
      --mode dp \
      --optimizer-sharding zero \
      --backend inductor \
      --batch-size 2 \
      --T 10

当前实现范围
++++++++++++++++++++++++

1. 线性层使用官方 tensor-parallel API。
2. 逐元素脉冲神经元现在会显式跟随上游 shard：

   * 对 ``[T, N, C]`` 激活，按最后一维 ``C`` 切分；
   * 对 ``[T, N, C, H, W]`` 激活，按通道维 ``C`` 切分。

   这意味着神经元内部状态 ``v`` 只保留本地 shard，而不是完整复制一份全局状态。
3. ``CIFAR10DVSVGG`` 的 ``Conv + BN + Neuron`` 主干支持实验性的 channel tensor parallel。
4. ``FSDP2 + TP`` 当前优先对 ``features`` 做 FSDP2 分片；当 ``classifier`` 已经启用 TP 时，不再额外对其做 root fully-shard，以避免跨 mesh 维度重复切分。
5. 传统 ``hybrid``（即 ``DDP + TP``）当前显式不支持，接口会直接提示改用 ``fsdp2_tp``。
6. ``PP`` 当前通过手工 stage 切分实现，而不是依赖 ``torch.export`` 整图切分；这样可以兼容标准脉冲神经元的内部状态写入。
7. ``PP`` 在 microbatch 之间会显式重置每个 stage 内的神经元状态，避免不同样本的状态串扰。

服务器实测结果（小网络 smoke benchmark）
++++++++++++++++++++++++

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
+++++++++++++++++++++++

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
++++++++++++++++++++++++

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
+++++++++++++++++++++++

``benchmark/benchmark_snn_distributed.py`` 现在会默认把结果追加到 ``benchmark/results/benchmark_snn_distributed.jsonl``，并自动和同配置的上一条记录做对比。记录里会统一保存：

* ``global_samples_per_second``
* ``peak_allocated_mb``
* ``optimize_ms``
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
* 在大尺寸 ``Spikformer`` 工作负载上直接使用高 level ``memopt``（``level >= 2``）做在线搜索：虽然功能上已经可用，但 ``optimize_ms`` 仍然很高，并且更容易触发 `inductor` 的额外重编译，建议先离线搜索、再固定策略。
