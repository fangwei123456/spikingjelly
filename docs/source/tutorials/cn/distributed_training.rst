SNN 分布式训练
==============

English version: :doc:`../en/distributed_training`

高层接口用于直接启动训练；底层接口用于接入模型或自行编写训练循环。最后一节给出
不同并行策略的实测吞吐和显存。

高层 API
--------

视觉模型
~~~~~~~~

``spikingjelly.activation_based.distributed.vision`` 提供基于 PyTorch DDP、FSDP2、
张量并行和流水线并行的图像分类训练入口：``vision.TrainingConfig`` 描述任务，
``vision.train`` 执行训练。

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

``batch_size`` 是每个 DP rank 的 batch size；global batch 为
``batch_size * DP``，不乘 TP、PP 或 SNN 时间步。``tensor_parallel_size`` 和
``pipeline_parallel_size`` 分别控制 TP 和 PP，剩余 rank 自动作为 DP。内置模型包括
``SEWResNet34Config`` 和 ``SpikformerConfig``。

仓库中的合成数据入口可以直接验证安装和并行配置：

.. code-block:: bash

    torchrun --standalone --nproc-per-node=4 benchmark/vision_distributed.py \
        --model sew-resnet34 \
        --data-parallel fsdp2 \
        --tensor-parallel-size 2 \
        --precision bf16 \
        --max-steps 10

自定义模型通过 ``vision.ModelConfig`` 和 ``vision.ModelBuilder`` 接入。
``build`` 返回当前 rank 的模型、FSDP2 分片模块路径，以及 PP 输入输出形状；完整签名见
:class:`spikingjelly.activation_based.distributed.vision.ModelBuilder`。

LLM
~~~

``spikingjelly.activation_based.distributed.llm`` 提供基于 Megatron Core 的 SNN
语言模型训练。该路径要求 Python 3.12 或更高版本；安装可选依赖后使用：

.. code-block:: bash

    uv pip install --editable ".[megatron]"

``llm.TrainingConfig`` 组合以下配置：

* ``llm.ModelConfig``：MCore ``TransformerConfig``、词表、context 和 SNN 时间步；
* MCore ``OptimizerConfig``；
* dataset builder、micro/global batch、训练步数、验证和 checkpoint；
* 可选的 SpikingJelly memopt。

未指定并行拓扑时，``llm.plan_training`` 根据 GPU 数量、显存预算和目标选择 TP、PP、
CP 与重计算配置；已有明确拓扑时直接设置 ``TransformerConfig``。SpikeLM 的完整
config、optimizer 和 dataset 配置位于 ``benchmark/snn_llm/cli.py``，可直接启动：

.. code-block:: bash

    torchrun --standalone --nproc-per-node=4 \
        benchmark/snn_llm/train_spikelm.py \
        --data /datasets/tokens \
        --output checkpoints/spikelm \
        --train-steps 200 \
        --global-batch-size 128

底层 API
--------

自定义视觉模型
~~~~~~~~~~~~~~

``vision.ModelBuilder.build`` 构建当前 PP stage、配置模型并行并返回 FSDP2 分片位置。
实现可参考 ``SEWResNet34Builder`` 和 ``SpikformerBuilder``。

最小声明形式如下：

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

``parallelize_my_model`` 由模型作者定义。下面的例子直接使用
``spikingjelly.activation_based.distributed.tensor_parallel`` 中的公开组件替换模型层：

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

神经元直接接收 colwise 层产生的本地通道张量，不需要额外包装；模型作者只需保证后续
rowwise 层消费该本地张量。

自定义训练流程
~~~~~~~~~~~~~~

若预定义 ``train`` 不适合任务，可以直接组合 PyTorch 分布式接口和上述 SpikingJelly
组件。下面省略任务相关的 ``build_my_model``、``dataset`` 和超参数，只展示组装顺序：

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

验证、混合精度、scheduler、指标归约和 checkpoint 由具体任务补充。以下调用约束不能省略：

* world size 能被所选模型并行大小整除；
* 数据只沿 DP 维切分，同一 TP group 使用相同 batch；
* 完成模型并行和 DDP/FSDP2 包装后再创建 optimizer；
* 每个相互独立的 batch 或 pipeline microbatch 后重置 SNN 状态；
* global batch 不包含 TP、PP、CP 或 SNN 时间步。

自定义 LLM
~~~~~~~~~~

LLM 模型继承 ``llm.ModelConfig``，并通过 ``builder`` 类变量指向一个
``llm.ModelBuilder``。builder 的 ``build`` 方法返回 MCore 需要的
``model_provider`` 和 ``forward_step``：

.. code-block:: python

    from spikingjelly.activation_based.distributed import llm

    class MyModelBuilder(llm.ModelBuilder):
        def build(self, *, use_snn_memopt: bool, resume: bool):
            return model_provider, forward_step

``model_provider`` 构建当前 PP stage；``forward_step`` 从 data iterator 读取一个
microbatch，调用模型并返回 MCore loss callback。用户可以复用 ``llm.train``，也可以
在自己的 MCore 训练流程中使用这两个 callback。SpikeLM 和 Qwen2 的完整实现分别位于
``benchmark/snn_llm/spikelm.py`` 和 ``benchmark/snn_llm/qwen2.py``。

SNN 时间布局为 ``[T, B, S, H] -> [S, T*B, H]``。``T`` 只并入 MCore batch 维，
不会并入 token 维，也不用于计算 global batch。

实测效果
--------

以下结果在 4 张 RTX 4090 24 GiB 上测得。机器没有 NVLink，跨卡 CUDA peer access
均为 ``False``；软件栈为 PyTorch 2.8.0、Megatron Core 0.18.2 和 Triton 3.4.0。
因此结果适合作为 PCIe 多卡机器上的相对参考，不应直接外推到 NVLink 集群。

Vision 基准
~~~~~~~~~~~

Vision 基准固定 BF16、``T=4``、128 × 128 输入和 1000 类。图中只保留单卡、
DP4、FSDP4、TP4 和 PP4；曲线末端标出最大的成功 global batch size（``G``）。单卡、TP4
和 PP4 的 global batch 等于每 rank batch，DP4 和 FSDP4 则是每 rank batch 的
4 倍。

固定 global batch 的全拓扑对比
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下表恢复了 ``G=32`` 时的完整拓扑对比。每个配置从新进程启动，预热 10 个
optimizer step，再测量 50 个 step，并独立重复三次；表中为三次中位数。吞吐是整个
作业的总吞吐，显存是所有 rank 中最高的 CUDA peak allocated memory。

.. list-table:: Vision 固定 ``G=32`` 的全拓扑结果
    :header-rows: 1

    * - 拓扑
      - GPU
      - SEW-ResNet34 images/s
      - SEW-ResNet34 GiB/卡
      - Spikformer-S images/s
      - Spikformer-S GiB/卡
    * - 单卡
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

在相同 ``G=32`` 下，多卡主要降低单卡显存，并不会自动提高吞吐：每张卡分到的计算量
变小后，PCIe 通信、同步和 pipeline bubble 更容易占主导。下面的容量曲线回答另一个
问题：允许并行方案使用更大的 global batch 时，吞吐—显存边界能扩展到哪里。

batch 从已有点开始按 2 的幂增加，直到首个不能完成的候选。SEW-ResNet34 的单卡、
DP4、FSDP4 成功到每 rank batch 256，TP4、PP4 成功到 512；每个配置从新进程启动，
预热 10 个 optimizer step，再测量 40 个 step。Spikformer-S 的单卡、DP4、FSDP4
成功到每 rank batch 128，TP4、PP4 成功到 256；预热 5 个 step，再测量 25 个
step。每个成功点均独立重复三次，图中绘制中位数和三次范围。计时包含 H2D、forward、
backward、通信与 optimizer，不包含初始化、DataLoader、验证和 checkpoint。

纵轴是整个作业的总吞吐：所有 rank 完成的 global batch 除以最慢 rank 的计时时间；
横轴是所有 rank 中最高的 CUDA peak allocated memory。它们分别回答“整个作业每秒
处理多少图”和“至少需要多大的单卡显存”。两轴使用对数尺度，失败候选不绘制为吞吐点。

.. figure:: ../../_static/tutorials/distributed/sew-resnet34-tradeoff.png
    :width: 720px
    :alt: SEW-ResNet34 不同 global batch 下的总吞吐与单卡峰值显存

    SEW-ResNet34：总训练吞吐与最繁忙 GPU 峰值分配显存。

.. figure:: ../../_static/tutorials/distributed/spikformer-tradeoff.png
    :width: 720px
    :alt: Spikformer-S 不同 global batch 下的总吞吐与单卡峰值显存

    Spikformer-S：总训练吞吐与最繁忙 GPU 峰值分配显存。

SEW-ResNet34 的 DP4 在 ``G=1024`` 达到 3616.2 images/s 和 11.25 GiB/卡，
FSDP4 为 3482.5 images/s 和 10.86 GiB/卡；PP4 在 ``G=512`` 达到
1636.5 images/s。Spikformer-S 的 DP4 和 FSDP4 最大成功点均为 ``G=512``，
分别达到 2503.8 和 2334.6 images/s；PP4 在 ``G=256`` 达到 1028.2 images/s。
TP4 在两个模型上均早早进入吞吐平台，继续增大 batch 主要增加显存，说明该 PCIe
机器上的 TP 通信已经主导。这些数字描述扩大总工作量后的吞吐—容量边界，不是固定
batch 加速比。

.. list-table:: Vision 容量搜索边界（最大成功点 → 首个失败候选）
    :header-rows: 1

    * - 模型
      - 拓扑
      - 最大成功 ``B/G``
      - 首个失败 ``B/G``
      - 结果
    * - SEW-ResNet34
      - 单卡
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
      - 单卡
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

``B`` 是每 rank batch。collective timeout 表示候选没有产出训练指标，不能当作
慢速成功点，也不能在没有 OOM 栈时标成 OOM。

LLM 基准
~~~~~~~~

LLM 基准使用约 1.41B 参数的 SpikeLM：24 层、hidden 2048、16 heads、FFN 8192、
词表 50304、BF16、sequence 128 和 ``T=4``。所有点均关闭 SpikingJelly memopt 和
梯度累计，因此 ``global_batch_size = micro_batch_size × data_parallel_size``；每个
optimizer step 在每个 DP rank 上只执行一个 micro batch。

固定 global batch 的全拓扑对比
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下表是此前遍历全部 2 卡、4 卡及混合拓扑的固定工作量实验：``micro batch=1``、
``G=8``，预热 10 个 optimizer step，再测量 30 个 step，并独立重复三次；表中为
三次中位数。单卡在 distributed optimizer 初始化时 OOM，因此以 DP2 作为相对吞吐
基线。为让不同 DP size 都保持 ``G=8``，这组固定工作量实验使用了梯度累计；它与
下方明确关闭梯度累计的容量搜索是两套不同口径。

.. list-table:: 1.41B SpikeLM 固定 ``G=8`` 的全拓扑结果
    :header-rows: 1

    * - 拓扑
      - GPU
      - semantic tokens/s
      - GiB/卡
      - 相对 DP2 吞吐
    * - DP2
      - 2
      - 746.3
      - 17.35
      - 1.00×
    * - TP2
      - 2
      - 679.9
      - 12.86
      - 0.91×
    * - PP2
      - 2
      - 1008.3
      - 13.15
      - 1.35×
    * - CP2
      - 2
      - 417.7
      - 16.56
      - 0.56×
    * - DP4
      - 4
      - 585.6
      - 13.40
      - 0.78×
    * - TP4
      - 4
      - 673.1
      - 6.65
      - 0.90×
    * - PP4
      - 4
      - 1379.9
      - 8.09
      - 1.85×
    * - CP4
      - 4
      - 289.5
      - 12.34
      - 0.39×
    * - TP2 + DP2
      - 4
      - 817.1
      - 8.91
      - 1.09×
    * - PP2 + DP2
      - 4
      - 997.9
      - 9.20
      - 1.34×
    * - CP2 + DP2
      - 4
      - 446.0
      - 12.61
      - 0.60×
    * - TP2 + PP2
      - 4
      - 989.1
      - 6.81
      - 1.33×
    * - TP2 + CP2
      - 4
      - 427.2
      - 8.39
      - 0.57×
    * - PP2 + CP2
      - 4
      - 612.6
      - 8.44
      - 0.82×

固定 ``G=8`` 时，PP4 的总吞吐最高，TP4 和 TP2 + PP2 的单卡峰值显存最低；CP
在 sequence 只有 128 时通信收益不足。这个表用于横向比较拓扑，下面的无梯度累计
实验用于比较各拓扑扩大 batch 后的容量和吞吐上限。

图中保留 DP2、DP4、TP4、PP4 和 CP4。DP2 只成功到 micro batch 1（``G=2``）；
DP4 成功到 micro batch 4（``G=16``）；TP4、PP4 和 CP4 均成功到 micro batch
16（``G=16``）。每个配置从新进程启动，预热 5 个 step，再测量 15 个 step，并
独立重复三次。单卡在 distributed optimizer 初始化时 OOM，因此没有强行加入单卡点。
LLM 路径使用 MCore DDP 与其 distributed optimizer，不叠加 PyTorch FSDP2。

.. figure:: ../../_static/tutorials/distributed/spikelm-1.41b-tradeoff.png
    :width: 720px
    :alt: 1.41B SpikeLM 不同 global batch 下的总吞吐与单卡峰值显存

    1.41B SpikeLM：无梯度累计、无 memopt 时的总训练吞吐与最繁忙 GPU 峰值分配显存。

PP4 的 ``G=16`` 点达到 2997.4 semantic tokens/s 和 14.85 GiB/卡，是本组最高
吞吐；从 ``G=8`` 的 2897.0 tokens/s 到 ``G=16`` 已接近平台。TP4 在 ``G=16``
达到 1684.4 tokens/s 和 16.47 GiB/卡，同样在 ``G=4`` 后明显变平。DP4 的最大
成功点仍为 ``G=16``：1284.3 tokens/s 和 17.55 GiB/卡。CP4 扩大到 ``G=16``
后达到 865.5 tokens/s 和 16.52 GiB/卡，但仍低于 TP4/PP4。DP2 只保留
``G=2``：303.0 tokens/s 和 17.35 GiB/卡。

.. list-table:: LLM 容量搜索边界（最大成功点 → 首个失败候选）
    :header-rows: 1

    * - 拓扑
      - 最大成功 ``micro/G``
      - 首个失败 ``micro/G``
      - 结果
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
      - stalled，无训练指标
    * - CP4
      - 16/16
      - 32/32
      - stalled，无训练指标

PP4/CP4 的 ``micro=32`` 候选持续多个正常运行时长保持固定的 rank 等待状态，因而
终止并记为 stalled；它们不是吞吐点，也没有被误标成 OOM。

不同曲线的点可能具有不同 global batch，因此这张图表达吞吐—容量边界，而不是固定
batch 下的加速比。完整中位数、三次运行范围和 batch 配置见
:download:`汇总 CSV <../../_static/tutorials/distributed/distributed-tradeoff.csv>`。

功能测试还覆盖了 BF16 的 TP4、PP4、TP2 × PP2、CP4、TP2 × CP2、PP2 × CP2，
以及 TP4/PP4/CP4 的 FP8。所有组合都得到有限 loss、有限梯度，且 SNN 模块上存在非零梯度。
在 7 GiB 显存预算下，planner 选择 TP4、SpikingJelly memopt 和 MCore selective
``core_attn`` 重计算，两步训练使用 6.28 GiB。TP2 × PP2 的 sharded model/optimizer
checkpoint 也已验证从 step 1 恢复到 step 2。
