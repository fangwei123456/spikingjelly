训练与推理精度
==============

English version: :doc:`../en/precision`

SpikingJelly 有三条精度配置路径，应根据使用场景选择入口：

* 自定义 PyTorch 流程使用 ``PrecisionConfig`` 和
  ``prepare_model_for_precision``；
* ``distributed.vision`` 将 ``PrecisionConfig`` 放进训练、评测或预测配置；
* ``distributed.llm`` 使用 Megatron Core 自己的 ``TransformerConfig`` 和
  ``OptimizerConfig``，不使用 ``PrecisionConfig``。

模型级 FP16、BF16 和 FP8 与 Triton 神经元精度是两个独立维度：
普通层可以使用 BF16，同时让 Triton 神经元保持 FP32。

安装
----

BF16 和 FP16 只需要 PyTorch。模型级 FP8 需要 Transformer Engine：

.. code-block:: bash

    uv pip install --editable ".[fp8]"

Triton 神经元 mixed precision 另外需要：

.. code-block:: bash

    uv pip install --editable ".[triton]"

在已经预装 PyTorch 的 pip 环境中，添加 ``--no-build-isolation``，使 Transformer
Engine 根据现有 ``torch.version.cuda`` 选择 CUDA wheel。否则隔离构建环境可能临时
安装另一版 PyTorch，导致 CUDA wheel 与实际环境不一致：

.. code-block:: bash

    python -m pip install --no-build-isolation \
        "transformer-engine[pytorch]>=2.16,<3"

路径一：自定义训练或推理流程
----------------------------

调用顺序不能颠倒：先把模型移到目标设备，再调用
``prepare_model_for_precision``，最后创建 optimizer。FP8 准备过程会替换部分模块；
如果 optimizer 已经持有旧参数，后续训练不会更新新模块。

.. code-block:: python

    import torch
    from torch import nn

    from spikingjelly.activation_based.precision import (
        PrecisionConfig,
        prepare_model_for_precision,
    )

    device = torch.device("cuda")
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    precision = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode="fp8", fp8_recipe="auto"),
    )
    model = precision.model
    optimizer = torch.optim.AdamW(model.parameters())

    optimizer.zero_grad(set_to_none=True)
    with precision.autocast_context():
        output = model(torch.randn(256, 4096, device=device))
        loss = output.square().mean()
    precision.backward(loss, optimizer)

``mode`` 可取 ``fp32``、``fp16``、``bf16`` 或 ``fp8``。返回的
``PrecisionArtifacts`` 会在 FP16 下创建 GradScaler，所以四种 mode 可以共用该
训练循环。推理时仍在 ``autocast_context`` 中调用模型，但不调用 ``backward``：

.. code-block:: python

    model.eval()
    with torch.inference_mode(), precision.autocast_context():
        output = model(input_tensor)

检查 FP8 转换结果
~~~~~~~~~~~~~~~~~

FP8 当前转换对齐的 ``torch.nn.Linear``、SpikingJelly ``layer.Linear``、pointwise
Conv1d，以及支持的 LayerNorm 和相邻融合模式。Linear 输入维必须是 16 的倍数，输出维
必须是 8 的倍数。不满足对齐的层保留高精度并出现在诊断报告中：

.. code-block:: python

    report = precision.describe()
    print(report["conversion_report"])

Transformer Engine、硬件或 recipe 不可用时，准备过程直接报错，不会改用 BF16。
模型中至少要有一个可转换模块。``fp8_recipe="auto"`` 使用当前 Transformer Engine
的默认 recipe；需要固定数值策略时，再显式选择 ``delayed``、``current``、``block``
或 ``mxfp8``。

部分 Transformer Engine recipe 会把 FP8 metadata 序列化为 pickle。恢复可信来源的
checkpoint 时，需要显式设置 ``NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1``；不要为未知
checkpoint 开启该选项。

配置 Triton 神经元
~~~~~~~~~~~~~~~~~~

Triton 精度也在 ``prepare_model_for_precision`` 中设置，不需要给每个神经元增加构造
参数。例如，普通层可以使用 BF16，Triton 神经元使用 BF16 存储和前向、FP32
反向：

.. code-block:: python

    config = PrecisionConfig(
        mode="bf16",
        triton_storage="bf16",
        triton_fwd="bf16",
        triton_bwd="fp32",
    )
    precision = prepare_model_for_precision(model, device, config)

只有 ``backend="triton"``、``step_mode="m"`` 的 IFNode、LIFNode 和
ParametricLIFNode 会使用这些设置；函数不会替模型切换 backend。
``triton_fwd`` 和 ``triton_bwd`` 可分别取 ``fp8``、``fp16``、``bf16`` 或
``fp32``。FP8 算术要求
``triton_storage`` 为 ``float8_e4m3fn`` 或 ``float8_e5m2``。指数和敏感 surrogate
计算固定在 kernel 内部使用 FP32，不是用户选项。

路径二：``distributed.vision``
--------------------------------

视觉高层接口在并行包装和 optimizer 创建前读取配置中的 ``precision`` 字段。训练使用
``TrainingConfig``；评测和预测分别使用 ``EvaluationConfig`` 和
``PredictionConfig``，三者都接收同一个 ``PrecisionConfig``：

.. code-block:: python

    from spikingjelly.activation_based import distributed
    from spikingjelly.activation_based.precision import PrecisionConfig

    config = distributed.vision.TrainingConfig(
        model=model_config,
        dataset_builder=dataset_builder,
        precision=PrecisionConfig(
            mode="bf16",
            triton_storage="bf16",
            triton_fwd="bf16",
            triton_bwd="fp32",
        ),
    )
    result = distributed.vision.train_classification(config)

使用仓库的命令行示例时，``--precision`` 映射到 ``mode``，其余字段由
``--fp8-recipe``、``--triton-storage``、``--triton-fwd`` 和
``--triton-bwd`` 设置：

.. code-block:: bash

    uv run torchrun --standalone --nproc-per-node=2 \
        benchmark/vision_distributed.py \
        --model spikformer --dataset synthetic \
        --data-parallel ddp --precision fp8 \
        --image-size 128 --classes 1024 \
        --batch-size 32 --max-steps 10

``distributed.vision`` 会在 DDP 包装和 optimizer 创建前准备精度，并用 DDP 进程组
同步 Transformer Engine 的 scaling metadata。模型 FP8 和 Triton 神经元 mixed
precision 目前只支持 DDP，且 TP=PP=1。Vision PP 只支持 FP32 和 BF16；普通
FP32/BF16 不受这个实验性精度限制。

路径三：``distributed.llm``
-----------------------------

``distributed.llm`` 不使用 ``PrecisionConfig``。模型与 optimizer 的精度分别由
MCore ``TransformerConfig`` 和 ``OptimizerConfig`` 设置，二者必须一致：

.. code-block:: python

    import torch
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import TransformerConfig

    transformer = TransformerConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        ffn_hidden_size=8192,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    optimizer = OptimizerConfig(
        lr=3e-4,
        min_lr=3e-5,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )

将 ``transformer`` 放入具体的 ``distributed.llm.ModelConfig``，将 ``optimizer``
放入 ``distributed.llm.TrainingConfig``。FP16 时，两个配置都设置 ``fp16=True``、
``bf16=False`` 和 ``params_dtype=torch.float16``。启用 MCore FP8 时，通常在 BF16
基线上给 ``TransformerConfig`` 设置 ``fp8="hybrid"`` 和相同的
``fp8_recipe``，同时让 ``OptimizerConfig.fp8_recipe`` 与之匹配。PP 开启时，
``pipeline_dtype`` 必须与 ``params_dtype`` 一致。

独立评测和 cached generation 复用模型中的 ``TransformerConfig``。SGLang artifact
导出目前要求 MCore BF16；完整的 LLM 模型、数据和训练配置见
:doc:`./distributed_training`。

FP8 何时更快
------------

FP8 的收益来自大矩阵，不来自 dtype 名称本身。普通 FC-SNN 训练和当前
Spikformer DDP 都没有受益；足够宽的 Linear 和 FC-SNN 推理才越过了 FP16/BF16。

以下结果测于 2026-08-29。Linear 和 FC-SNN 使用一张 RTX 5090 32 GiB，DDP 使用
两张；软件版本为 PyTorch 2.11.0+cu128 和 Transformer Engine 2.18.0。Linear 和
FC-SNN 运行三次并轮换精度顺序，表中取中位数。吞吐至少提高 5% 才算有优势，计时
不包含模型转换。

Linear/GEMM 交叉点
~~~~~~~~~~~~~~~~~~

表中的两个比值依次为 ``FP8 / FP16`` 和 ``FP8 / BF16``：

.. list-table::
    :header-rows: 1

    * - workload（batch, width, depth）
      - 训练吞吐
      - 推理吞吐
      - 结论
    * - 4096, 2304, 8
      - 0.553x / 0.527x
      - 0.915x / 0.966x
      - FP8 更慢
    * - 4096, 2560, 8
      - 0.720x / 0.705x
      - 1.117x / 1.100x
      - 推理跨过门槛
    * - 4096, 3072, 8
      - 0.978x / 1.026x
      - 1.454x / 1.629x
      - 训练尚未跨过门槛
    * - 4096, 3200, 8
      - 1.060x / 1.067x
      - 1.464x / 1.614x
      - 训练和推理均跨过门槛
    * - 3072, 4096, 8
      - 1.390x / 1.472x
      - 1.513x / 1.684x
      - 训练和推理均明显更快

在这组 dense Linear 测试中，固定 batch=4096、depth=8 后，推理交叉点位于
width 2304--2560，训练交叉点位于 width 3072--3200。固定 width=4096 时，训练
交叉点位于 batch 2048--3072。

FP8 也不等于固定省显存。在 4096×3200×8 的交叉点，FP8 训练/推理 allocated memory
比 FP16 低约 9%/14%，但比 BF16 高约 24%/34%。

端到端 SNN 与 DDP
~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - workload
      - 训练 FP8 / FP16、BF16
      - 推理 FP8 / FP16、BF16
    * - FC-SNN：T16, batch 256, width 4096, depth 20
      - 0.954x / 0.942x
      - 0.901x / 0.896x
    * - FC-SNN：T16, batch 256, width 8192, depth 10
      - 0.908x / 0.887x
      - 1.310x / 1.299x

width=8192 时，FC-SNN 推理中的大矩阵已经足以抵消 FP8 的额外开销；训练还包含反向
和神经元计算，因此依然更慢。此时 FP8 训练显存比 FP16/BF16 高约 46%--47%。

2 卡 Spikformer DDP 使用 5 个计时 step：

.. list-table::
    :header-rows: 1

    * - 每卡 batch
      - FP8 images/s
      - FP16 / BF16 images/s
      - FP8 / 16-bit allocated memory 每卡
    * - 32
      - 505.9
      - 566.2 / 562.6
      - 5517 MiB / 3513 MiB
    * - 144
      - 1005.3
      - 1591.8 / 1488.1
      - 23807 MiB / 14896 MiB

这个 Spikformer 配置直到接近显存上限仍没有出现交叉点。FP8 与 DDP 可以一起工作，
但这里应该选择 FP16 或 BF16。

如何选择
--------

默认先用 BF16。profile 显示对齐的 Linear/MLP 占主要耗时，并且矩阵维度接近表中的
交叉区间时，再测试 FP8。CNN、神经元或通信占比较高时，FP8 通常无法弥补转换和
metadata 开销。FP8 的显存也可能高于 BF16，不应把它当作显存优化开关。

使用仓库 benchmark 在目标 GPU 上重测交叉点：

.. code-block:: bash

    uv run python benchmark/benchmark_fp8_training_inference.py \
        --batch-size 4096 --width 3200 --depth 8 --num-classes 3200 \
        --warmup 8 --training-steps 20 --inference-steps 40 --trials 3 \
        --precisions fp16 bf16 fp8 --baseline-precision bf16 \
        --output benchmark/output/fp8-vs-bf16.json

将 ``--baseline-precision`` 改为 ``fp16``，可分别检查两种 16-bit baseline。正式
训练还要在真实数据上比较收敛曲线；这个 benchmark 只检查 loss、输出和参数更新是否
有限，以及 steady-state 性能。
