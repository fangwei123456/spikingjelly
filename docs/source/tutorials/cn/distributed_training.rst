SNN 分布式训练与推理
====================

本教程作者： `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_、`Wei Fang (fangwei123456) <https://github.com/fangwei123456>`_

English version: :doc:`../en/distributed_training`

高层接口用于直接启动训练、评测与离线推理；底层接口用于接入模型
或自行编写运行循环。最后一节给出与训练相同 4 卡 RTX 4090 环境下的
吞吐和显存结果。

API 设计动机
------------

这套 API 首先按工作负载划分为 ``vision`` 和 ``llm``，而不是假设所有 SNN 都适用
同一种并行策略。Spiking CNN 的通道、特征图和流水线边界与 LLM 的 token、attention
和 context parallel 有不同语义；强行共用一套模型描述只会把这些差异藏进大量分支。
两条路径因此只在确实相同的外层概念上保持对称：都使用 ``ModelConfig`` 描述模型、
``ModelBuilder`` 接入 architecture-specific 实现、``TrainingConfig`` 描述训练，
``EvaluationConfig`` 描述有标签评测；Vision ``PredictionConfig`` 与 LLM generation
config 描述无标签输出。各高层入口拥有对应资源生命周期。

高层接口受到 Megatron Core 当前模型接入方式的启发。MCore 将声明式的
``TransformerConfig``、模型专项的 ``ModuleSpec`` / ``model_provider`` / ``forward_step``
与通用 pipeline schedule、optimizer 和 checkpoint 生命周期分开。SpikingJelly 沿用
这种“配置描述事实、builder 适配架构、训练入口拥有生命周期”的边界，而没有要求用户
修改一个巨大的预定义训练函数。LLM builder 返回 MCore 原生需要的 ``model_provider``
和 ``forward_step``；Vision builder 则返回 PyTorch pipeline 所需的 stage、FSDP2 分片
位置和边界形状。外层风格一致，内层契约服从各自运行时。

底层接口遵循“复用成熟运行时，只补 SNN 特有语义”的原则。DP、FSDP2、device mesh
和通用 pipeline 来自 PyTorch；LLM 的 TP、PP、CP、distributed optimizer 和 sharded
checkpoint 来自 Megatron Core。SpikingJelly 只提供这些运行时没有表达的部分，例如
SNN 时间布局与状态重置、适合通道型网络的分片层，以及脉冲压缩 memopt。memopt 与
MCore 重计算也保持职责分离：前者处理 SNN 激活和脉冲表示，后者仅在需要时处理不重叠
的 Transformer 子计算。因此，高层 ``train`` 适合标准流程，而需要新任务、模型或
调度方式的用户仍可直接组合下面的底层组件。

高层 API
--------

视觉模型
~~~~~~~~

``spikingjelly.activation_based.distributed.vision`` 提供基于 PyTorch DDP、FSDP2、
张量并行和流水线并行的图像分类训练入口：``vision.TrainingConfig`` 描述任务，
``vision.train_classification`` 执行训练。

.. code-block:: python

    from pathlib import Path

    from spikingjelly.activation_based.distributed import vision
    from spikingjelly.activation_based.model.sew_resnet import SEWResNet34Config

    config = vision.TrainingConfig(
        model=SEWResNet34Config(
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

``batch_size`` 是每个 DP rank 的 batch size；global batch 为
``batch_size * DP``，不乘 TP、PP 或 SNN 时间步。``tensor_parallel_size`` 和
``pipeline_parallel_size`` 分别控制 TP 和 PP，剩余 rank 自动作为 DP。模型专属的
distributed recipe 从 ``model.sew_resnet`` 和 ``model.spikformer`` 导入，而不是由
``distributed.vision`` 持有。仓库示例包括 ``SEWResNet34Config``、
``SpikformerConfig`` 和 ``SpikformerCIFAR10Config``。
CIFAR-10 变体固定使用官方的 32×32 输入、4×4 patch stem、384 维、12 个 attention
heads 和 4 个 Transformer blocks，并复用相同的 TP、PP 与 FSDP2 实现。
``mixup_alpha`` 配置可序列化的 batch-level mixup；``0`` 表示禁用。
rank 0 在每个 epoch 后输出一行 JSON，其中包含 optimizer step、train loss、
validation loss 和 validation accuracy；返回的 ``metrics`` 字典包含最终指标与吞吐统计。

``input_layout`` 显式声明 DataLoader batch 的布局。``"NCHW"`` 接收静态图像
``[N, C, H, W]``；single-step 直接对同一 batch 调用模型 ``T`` 次，multi-step
使用连续的 ``[T, N, C, H, W]``。``"NTCHW"`` 接收 CIFAR10-DVS、DVS Gesture
等默认 collate 产生的 ``[N, T, C, H, W]``，训练函数校验 T 后转为 time-first。
输入布局不根据 tensor 维数自动推断。

训练入口在并行包装前调用 ``functional.set_step_mode``，并在完整时间窗结束后调用
``functional.reset_net``。single-step 当前不支持 PP、memopt 或 Triton 神经元后端。
内置 SEW-ResNet34 支持 ``"s"`` 和 ``"m"``；Spikformer 的 architecture 与
attention 原生只支持 ``"m"``，不会通过 wrapper 模拟单步接口。
single-step DDP 会关闭逐次 forward 的 buffer 广播，避免 T 次调用期间原地修改
BatchNorm buffer；训练函数改为在每个完整 T 窗口前只广播一次 buffer，从而在不修改
单步 forward 已保存 buffer 的前提下同步各 DP rank。

``loss_function`` 是 loss callable 的完整导入路径；它接收经时间维归约后的
``[N, C]`` logits 和分类 target，并返回用于反向传播和 loss 统计的 batch-mean
标量张量。``loss_kwargs`` 是每次调用时传入的关键字参数。普通训练、PP、验证
共用同一 loss 函数；top-1 accuracy 仍是固定的分类指标。

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

``llm.train`` 当前只支持完整、相互独立的固定 T 时间窗。模型专项
``forward_step`` 负责时间编码、状态隔离与时间归约；该 MCore ``T*B`` envelope
不等同于通用 SpikingJelly ``step_mode="m"``，因此 LLM config 暂不提供
``step_mode`` 字段。

离线分布式推理
~~~~~~~~~~~~~~~~~~

接口分工
^^^^^^^^

推理接口按是否处于训练生命周期以及是否需要 ground truth 分为三类。validation
与 test 使用相同的评测计算，区别只是调用时机和数据集，因此不增加重复的
``validate`` / ``test`` 函数。

.. list-table:: Vision 与 LLM 推理接口分工
    :header-rows: 1

    * - 场景
      - Ground truth
      - Vision
      - LLM
      - 输出
    * - 训练期间 validation
      - 需要
      - ``train_classification`` 每个 epoch 评测 validation dataset
      - ``train`` 按 ``eval_interval`` / ``eval_steps`` 评测
      - validation loss/accuracy 或 LM loss
    * - 训练后 evaluation/test
      - 需要
      - ``evaluate_classification``
      - MCore ``evaluate``
      - 聚合 loss、accuracy/perplexity 和性能指标
    * - 训练后 direct prediction/generation
      - 不需要
      - ``predict_classification``
      - MCore ``generate``
      - 逐样本 logits 或生成 token；不返回评测指标
    * - 训练后 scheduler-backed offline generation
      - 不需要
      - —
      - SGLang ``open_sglang_engine``
      - 按请求生成 token；不返回评测指标

``evaluate_classification`` 要求 dataset 的每个元素都是 ``(image, target)``；
``llm.evaluate`` 同样要求 ``input_ids``、``labels`` 和可选 ``loss_mask``。
相反，prediction/generation 不读取 ground truth：Vision 即使收到
``(image, target)`` 也会忽略 target，LLM generation 只接收 prompt。
这四类均属于训练或离线工作流。SGLang ``Engine`` 包含管理 request/KV pool 的
内部执行 scheduler，但 SpikingJelly 路径不包含 HTTP server、router 或其他在线
serving 控制面。

Vision
^^^^^^

Vision 推理继续使用 PyTorch。训练 checkpoint 先导出为与 TP/PP 拓扑无关的
canonical artifact，之后可在不同 DP、FSDP2、TP 或 PP 拓扑上评测：

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

``vision.evaluate_classification`` 返回全局 loss、accuracy、images/s 和最繁忙
rank 的峰值显存。``vision.predict_classification`` 不计算或返回这些指标，而是把
各 rank 的输出按 dataset index 合并为一个 HDF5 文件；文件只包含 ``index`` 和
``logits``，类别可用 ``logits.argmax(axis=1)`` 得到。填充样本不会写入结果，因而
数据集大小无需整除 DP 或 batch size。

LLM
^^^

LLM 提供两个目的不同的 backend：

* MCore 与训练使用相同 model provider 和 sharded checkpoint，适合 validation、
  loss/perplexity evaluation 和具有直接 tensor-batch 语义的同步 generation。
  ``llm.evaluate(EvaluationConfig(...))`` 支持 DP/TP/PP/CP 的完整 loss/perplexity
  评测；``llm.generate(MCoreGenerationConfig(...), input_ids)`` 支持 DP prompt
  切分、TP/PP 和 static KV-cache decode。MCore cached generation 要求 CP=1。
* SGLang 用于训练完成后的 scheduler-backed 高吞吐离线生成。
  :func:`open_sglang_engine` 管理原生 Engine 生命周期；采样、变长 token IDs、异步
  生成和 streaming 直接使用原生 Engine 接口。该路径不启动 HTTP server 或 router。

.. list-table:: LLM 推理 backend 选择
    :header-rows: 1

    * - 需求
      - 后端
    * - validation、loss/perplexity、训练 checkpoint 直接恢复
      - MCore
    * - 可解释的 local/global batch、pipeline microbatch 和 CUDA OOM
      - MCore
    * - 大规模 prompt corpus、continuous batching 和 KV-cache 调度
      - SGLang
    * - HTTP/router、多租户或 SLA serving
      - 本轮不支持

MCore evaluation 与 SGLang generation 回答不同问题，不能用同一套 batch、显存或
吞吐 protocol 横向比较。

SGLang 0.5.17 固定自己的 PyTorch/Transformers 运行栈，不应与主训练
环境强行合并：

.. code-block:: bash

    uv venv --python 3.12 .venv-sglang
    source .venv-sglang/bin/activate
    uv pip install --editable ".[sglang]"

``llm.export_sglang_artifact`` 负责分布式 checkpoint 加载、逐 stage 分片写入、
index、失败同步和原子发布。模型自己的 ``stage_tensors`` 回调负责权重名称和变换，
``artifact_config`` 负责 SGLang/Hugging Face 配置。导出时用 ``torchrun`` 启动
checkpoint 的 TP x PP x CP 拓扑；任意 GPU 都不会聚合完整模型，输出 artifact 可在
推理时改用其他 TP/PP/DP 拓扑。

仓库中的 SpikeLM、Qwen2 recipe 和 external runtime model 位于
``benchmark/snn_llm``，用于演示该 seam，并不作为 wheel 内置模型支持。其 adapter
在内部保留 ``[token, T, hidden]``，仅在 RadixAttention/KV cache 边界并入 head 维。
自定义模型必须同时提供 export 回调和可导入的 SGLang external model package。

首个支持范围是单节点 NVIDIA BF16 TP、PP 和 DP；prefill CP 与 DCP 不受支持，
也不会出现在 ``SGLangEngineConfig`` 中。CUDA Graph 同样禁用：SGLang 0.5.17
graph capture 所需的 attention metadata 尚未由 temporal adapter 提供。SpikeLM 与
Qwen2 adapter 使用 SGLang 原生 stage 分层和 ``PPProxyTensors`` 协议。

.. code-block:: python

    from pathlib import Path

    from spikingjelly.activation_based.distributed import llm

    def main():
        config = llm.SGLangEngineConfig(
            artifact=Path("artifacts/qwen2-snn"),
            external_model_package="benchmark.snn_llm.sglang_models",
            tensor_parallel_size=2,
        )
        with llm.open_sglang_engine(config) as engine:
            outputs = engine.generate(
                input_ids=[[1, 2, 3], [1, 2, 3, 4, 5]],
                sampling_params={"temperature": 0, "max_new_tokens": 32},
            )
        print(outputs)

    if __name__ == "__main__":
        main()

使用 ``benchmark/snn_llm/sglang_benchmark.py`` 产生可复现的 offline Engine
测量。它支持变长 token 和共享前缀 workload，报告 request/input/output 吞吐、
TTFT、TPOT、端到端延迟的 median/p99，以及每卡峰值显存。

正式 SGLang 验收使用一台 on-demand 4 × RTX 4090、无 NVLink 的主机，软件为
PyTorch 2.11.0、CUDA 13.0、SGLang 0.5.17、BF16、Triton attention，并关闭
CUDA Graph。每个性能点先执行一个 warmup request；Engine 在 warmup 后及每次
测量前清空 Radix cache，再重复三次并报告中位数。Qwen artifact 使用
Qwen2.5-0.5B 权重和确定性的全一 QCFS scale，因此这里是系统测量，不是模型质量结论。

.. figure:: ../../_static/tutorials/distributed/sglang-inference.png
    :width: 900px
    :alt: SGLang Qwen2.5-0.5B 扩展吞吐与共享前缀延迟

    左图：固定 workload 下 TP1、PP2 和 DP4 的 output 吞吐与 p99 TPOT。
    右图：TP1 在无共享和 2048-token 共享前缀下的 input/output 吞吐与 p99 TTFT。
    右图的 prompt 长度差异用于刻画 Radix cache，不是拓扑横向对比。

DP4 的 output 吞吐为 TP1 的 3.93 倍，而 TPOT 基本不变。该 PCIe 主机上的 TP2
与 PP2 均慢于 TP1；模型并行应用于容量，之后再用 DP 扩吞吐。12.6B SpikeLM
导出包含 564 个 tensor、25,173,851,048 artifact bytes，并成功在 PP4 上加载和生成，
没有任何 GPU 持有完整模型。

Qwen2 在固定 parity prompts 上与 MCore 的 32 个 greedy tokens 完全一致。
SpikeLM 的所有首个 decode token 一致；PP2 的四个 prompt 中三个完整 8-token 一致，
一个在第三个生成 token 后因不同 BF16 执行次序分叉。因此，接近并列的 logits 下，
跨 backend 的 free-running token 完全相同不属于契约。

完整中位数见 :download:`SGLang 结果 CSV
<../../_static/tutorials/distributed/sglang-inference-results.csv>`。

可直接从该 CSV 重新生成图片：

.. code-block:: bash

    python benchmark/plot_sglang_inference.py \
        docs/source/_static/tutorials/distributed/sglang-inference-results.csv \
        docs/source/_static/tutorials/distributed/sglang-inference.png

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
词表 50304、BF16、sequence 128 和 ``T=4``。下方容量搜索中的所有点均关闭
SpikingJelly memopt 和梯度累计，因此
``global_batch_size = micro_batch_size × data_parallel_size``；每个 optimizer step
在每个 DP rank 上只执行一个 micro batch。

固定 global batch 的全拓扑对比
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下表是此前遍历全部 2 卡、4 卡及混合拓扑的固定工作量实验：``micro batch=1``、
``G=8``，预热 10 个 optimizer step，再测量 30 个 step，并独立重复三次；表中为
三次中位数。单卡在 distributed optimizer 初始化时 OOM，因此以 DP2 作为相对吞吐
基线。为让不同 DP size 都保持 ``G=8``，这组固定工作量实验使用了梯度累计：累计
次数为 ``8 / DP``，所以 DP2 为 4、DP4 为 2，其余 DP1 拓扑为 8。它与下方明确关闭
梯度累计的容量搜索是两套不同口径。

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

分布式推理基准
----------------

共同实验环境
~~~~~~~~~~~~

推理基准使用与训练相同的单机 4 × RTX 4090 24 GiB 环境。``nvidia-smi topo -m``
显示 GPU0 到其余 GPU 为跨 NUMA 的 ``SYS``，GPU1--3 之间为 ``NODE``，没有
``NV#`` 链路；``nvidia-smi topo -p2p r/w`` 对所有跨卡组合均返回 ``CNS``。
因此本机既没有 NVLink，也没有可用的 CUDA P2P read/write，NCCL 跨卡通信经过
PCIe/CPU interconnect。Vision/MCore 软件栈与训练相同：PyTorch 2.8.0、Megatron
Core 0.18.2 和 Triton 3.4.0。

Vision evaluation
~~~~~~~~~~~~~~~~~

下方历史曲线使用 BF16、``T=4``、1000 类和缓存的全零 224 × 224 合成图像。
当前正式 benchmark 必须通过 ``--cifar10-data`` 或 ``--data`` 使用固定真实图像
子集；seeded-random 默认值只用于 smoke。历史点与替换后的新点不能混在同一条曲线。
SEW-ResNet34
各非 PP 曲线的 per-rank batch grid 为
``16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024``；Spikformer-S 根据
OOM 在 384--1024 之间停止。PP4 固定 K=4；SEW-ResNet34 的 L grid 为
``16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072``，
Spikformer-S 使用同一 grid 到 ``1536``。每个成功吞吐点从新进程启动，使用
4 个 DataLoader workers，
预热 5 个 batch、测量 10 个 batch，并独立重复三次。计时包含 H2D、forward、通信和指标归约，不包含 DataLoader、
模型/artifact 加载和初始化。图中连接同一 protocol 下所有三次完成点，并显示三次
中位数与完整范围；不做 Pareto 抽点或人工平滑。
Vision 与 MCore 小节用 ``L`` 表示每个 DP rank/replica 的本地 batch，用 ``G``
表示整个作业的 global batch；始终有 ``G = L × DP``，TP、PP、CP 和 SNN 时间步
``T`` 均不乘入 ``G``。PP 另用 ``K`` 表示 pipeline microbatch 数，每块大小为
``L / K``。Vision 图端点只标 global batch ``G``。

所有容量搜索使用相同的增长规则：从最大成功点 ``x`` 测 ``2x``；若 ``2x``
失败，再测 ``1.5x``；若 ``1.5x`` 成功，就以它为新的 ``x`` 重复上述过程。
Vision 的所有拓扑均搜索到 CUDA OOM。

推理 PP 使用专门的 forward-only streaming schedule，而不是训练所需的
``ScheduleGPipe`` backward 状态机。每个高层 batch 返回前同步 pipeline group，避免
跨 batch 堆积；SEW 下采样 block 被放到 stage 边界前，Spikformer 的 blocks 按
``0/2/2/2`` 分配。公共配置中的 ``pipeline_microbatches`` 表示每个 DP rank 的
本地 batch 被切成的块数，默认 1，并要求 ``batch_size`` 能被它整除；每块样本数为
``batch_size / pipeline_microbatches``。Vision 与 MCore 的 PP benchmark 都固定
``pipeline_microbatches = 4``，因此 L 增长时每块样本数始终按 ``L / 4`` 等比例
增长，直到 CUDA OOM。该规则只属于实验 protocol，框架不会
在用户调用中自动改写这个参数。汇总 CSV 分别记录 ``per_rank_batch_size``、
``global_batch_size``、``pipeline_microbatches`` 和
``pipeline_microbatch_size``。
Vision 图连接统一 protocol 下直到 OOM 的全部成功 batch sweep 点；CSV 还保留
失败候选的状态。

.. figure:: ../../_static/tutorials/distributed/sew-resnet34-inference-tradeoff.png
    :width: 720px
    :alt: SEW-ResNet34 分布式评测吞吐与单卡峰值显存

    SEW-ResNet34：固定 PP K=4 的完整 batch sweep。

.. figure:: ../../_static/tutorials/distributed/spikformer-inference-tradeoff.png
    :width: 720px
    :alt: Spikformer-S 分布式评测吞吐与单卡峰值显存

    Spikformer-S：固定 PP K=4 的完整 batch sweep。

在每 rank batch 128 时，SEW-ResNet34 的单卡、DP4、FSDP4、TP4、PP4 分别为
845.7、3368.9、3109.3、548.3、1320.9 images/s；Spikformer-S 分别为
516.6、2060.4、2000.2、412.2、1088.5 images/s。DP/FSDP 接近四卡线性吞吐，
PP 分别达到单卡的 1.56 倍和 2.11 倍。固定 K 后 PP 吞吐在中等 batch 达峰，随后
随每块样本和显存继续增大而平滑进入容量尾部。

纯 TP4 在两个模型上仍低于单卡，但曲线已经稳定；这是模型计算/通信比限制，不是调度
波动。SEW 每 batch 需要 16 次、合计约 1.41 GB 的 rowwise all-reduce，Spikformer
需要 12 次、约 0.92 GB。使用两个 TP2 replica 的四卡 ``TP2 × DP2`` 可达到
1226.1 和 858.6 images/s，说明实际部署应以 TP 容纳模型、再以 DP 扩展吞吐。

.. list-table:: Vision 推理容量边界
    :header-rows: 1

    * - 模型
      - 拓扑
      - 最大三次完成 ``L/G``
      - 首个失败与最终容量证据
    * - SEW-ResNet34
      - 单卡
      - 512/512
      - 768/768：正式多 batch 运行 CUDA OOM
    * - SEW-ResNet34
      - DP4
      - 512/2048
      - 768/3072：正式多 batch 运行 CUDA OOM
    * - SEW-ResNet34
      - FSDP4
      - 512/2048
      - 768/3072：正式多 batch 运行 CUDA OOM
    * - SEW-ResNet34
      - TP4
      - 512/512
      - 768/768：正式多 batch 运行 CUDA OOM
    * - SEW-ResNet34
      - PP4
      - 3072/3072
      - 4096/4096：CUDA OOM
    * - Spikformer-S
      - 单卡
      - 256/256
      - 384/384：CUDA OOM
    * - Spikformer-S
      - DP4
      - 384/1536
      - 512/2048：CUDA OOM
    * - Spikformer-S
      - FSDP4
      - 256/1024
      - 384/1536：CUDA OOM
    * - Spikformer-S
      - TP4
      - 512/512
      - 768/768：CUDA OOM
    * - Spikformer-S
      - PP4
      - 1536/1536
      - 2048/2048：CUDA OOM

Vision 正确性测试还覆盖 FSDP2、PP2，以及 TP2 × PP2 训练 checkpoint 导出后在
TP1 × DP4 上恢复；后者的 validation loss 为 2.310132205 和 2.310132384。

MCore loss/perplexity evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MCore loss/perplexity 评测使用 Qwen2.5-0.5B QCFS、BF16、``T=2`` 和序列长度16。
TP1/DP4 的基线段使用固定 128-sample 数据集；新测的 TP2/PP2/PP4 点令数据集
样本数等于 G，避免 padding 污染吞吐。拓扑为 TP1、DP4、TP2、
PP2 和 PP4；该模型有 14 个 attention heads，故四卡节点上大于 1 的合法纯 TP
拓扑为 TP2，TP4 不满足 head 整除约束。每个点从新进程恢复同一初始化状态的
sharded checkpoint，先执行 5 个不计时 schedule batch，再计时完整 schedule，
并独立重复三次；checkpoint/model 初始化、dataset indexing 和 collation 不计时。
计时包含 H2D、模型执行、通信和指标归约。Vision 采用相同的计时边界，但其图像吞吐
与 LLM token 吞吐属于不同 workload，不能横向比较。新测 sweep 显式设置
``NCCL_P2P_DISABLE=1``、``NCCL_IB_DISABLE=1``
和 ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``。MCore API 中
``micro_batch_size`` 表示每块大小，而本节的 L 是
``micro_batch_size × pipeline_microbatches``。非 PP 点 K=1；PP2/PP4 的所有点
固定 K=4，因此每块为 ``L/4``，会随 L 一起增长到 OOM，而不是固定每块大小后只增加
排队块数。细网格为 ``16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024``，
再按 ``2x/1.5x`` 规则搜索各拓扑边界。三次正式曲线点均预热 5 个 schedule；
PP4 的 L=3072 debug 容量 probe 为避免重复预热造成 allocator 碎片，只预热 1 个，
并仅作为容量证据。

.. figure:: ../../_static/tutorials/distributed/mcore-inference.png
    :width: 720px
    :alt: Qwen2.5-0.5B QCFS 的 MCore 分布式评测吞吐与单卡峰值显存

    MCore loss/perplexity 评测的完整 batch sweep：总 semantic-token 吞吐与最繁忙 GPU 的 peak allocated memory。

在小 batch 的 L=16，TP1、TP2、PP2 和 PP4 分别为 4636.2、3611.2、1823.5
和 2203.6 semantic tokens/s；固定 K=4 后 PP 每块仅有 4 个样本，kernel 与 schedule
开销尚未摊薄。到 L=384，TP1、TP2、PP2 和 PP4 分别达到 23145.8、28975.2、
24707.7 和 28348.2 tokens/s，三种模型并行拓扑均已超过同 L 单卡。

TP1、TP2、PP2 和 PP4 的最佳点分别为 24549.7、29767.5、30217.9 和
34317.2 tokens/s；后三者为单卡最佳点的 1.21、1.23 和 1.40 倍。TP2 最佳点在
L=256、3.95 GiB/卡；PP2/PP4 最佳点都在 L=1024，分别为 7.57/7.40 GiB/卡。
PP2/PP4 的三次正式曲线均延伸到 L=2048、约 14.5 GiB/卡，容量尾部的吞吐下降在
三次运行中重复出现，是每块增至 512 后的真实代价。PP4 的 L=3072 debug 容量
probe 单次通过（21.35 GiB/卡），但无 debug 的正式运行 timeout，因此不进入曲线；
L=4096 明确 CUDA OOM。

.. list-table:: MCore 容量尾部（最大完成点 → 首个失败点）
    :header-rows: 1

    * - 拓扑
      - 最大完成 ``L/G``
      - 首个失败 ``L/G``
      - 状态
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
      - 单次 debug probe 通过但正式运行 timeout；4096 CUDA OOM

完整中位数、运行范围、显存和失败状态见
:download:`推理结果 CSV <../../_static/tutorials/distributed/distributed-inference-tradeoff.csv>`。
该 CSV 只包含 Vision 和 MCore evaluation 结果；图可直接从汇总 CSV 重新生成：

.. code-block:: bash

    python benchmark/plot_distributed_inference.py \
        docs/source/_static/tutorials/distributed/distributed-inference-tradeoff.csv \
        docs/source/_static/tutorials/distributed
