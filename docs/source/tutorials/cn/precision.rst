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

控制未转换算子的 dtype
~~~~~~~~~~~~~~~~~~~~~~~

Transformer Engine 只改变已转换模块。FP8 默认给其余普通 CUDA 算子增加 BF16
autocast，避免 Conv2d、BatchNorm 或神经元沿 FP32 边界执行：

.. code-block:: python

    config = PrecisionConfig(
        mode="fp8",
        fp8_recipe="auto",
    )

需要复现实验时，可以用 ``fp8_fallback_dtype="fp16"`` 或 ``"fp32"`` 显式覆盖。
这里的 fallback 指未被 FP8 覆盖的算子，不是错误降级。该字段只控制普通 CUDA
autocast 和 TE 模块输出边界，不表示这些算子已使用 FP8 kernel。

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

FP8 的收益来自大矩阵，不来自 dtype 名称本身。FC-SNN 是否受益取决于神经元后端和
Linear--神经元边界；Spikformer 的卷积、BatchNorm 和神经元占比更高，目前单卡仍未
越过 FP16/BF16。

Dense Linear 和 2 卡 DDP 的历史结果测于 2026-08-29，使用 Transformer Engine
2.18.0；下面替换后的 FC-SNN 以及新增的单卡 Spikformer 结果测于 2026-08-30，
使用 PyTorch 2.11.0+cu128、Transformer Engine 2.17.1 和一张 RTX 5090 32 GiB。
后者是因为该镜像中的 TE 2.18.0 extension 无法加载，不能把两套软件栈的数字混为一谈。
FC-SNN/Spikformer 每个稳态 case 运行三次并取独立进程 median；吞吐至少提高 5% 才算
有优势，计时不包含模型转换。

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
    * - FC-SNN：T16, batch 256, width 4096, depth 20（Triton LIF，FP16 fallback）
      - 1.533x / 1.677x
      - 1.578x / 1.786x
    * - FC-SNN：T16, batch 256, width 8192, depth 10（Triton LIF，FP32 fallback）
      - 1.544x / 1.468x
      - 1.648x / 1.471x

这里的比值仍依次为 ``FP8 / FP16`` 和 ``FP8 / BF16``，且是端到端吞吐比值。两组
FC-SNN 都使用现有 Triton LIF；FP8 神经元 ``triton_storage`` 未启用，神经元计算仍
保持高精度。W4096 使用 ``fp8_fallback_dtype="fp16"``，训练/推理峰值 allocated
memory 为 4279.1/1460.3 MiB；W8192 是未启用外层 autocast 的既有结果。因此，这些
结果说明“FP8 Linear + Triton LIF”已经在该规模上超过 FP16/BF16，而不是说明所有
神经元都已经用 FP8，也不能据此假定 FP8 固定节省显存。

旧教程中的慢速 FC-SNN 数字来自 Torch LIF，只保留在 nsys 根因报告中，不再作为
FC-SNN 的推荐 benchmark。

W4096/depth20 的 requested FP8 tracked dense-MAC coverage 为 100%。

复现 FC-SNN profile：

.. code-block:: bash

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o fcsnn-fp8 \
      uv run python benchmark/benchmark_train_precision_snn_fc.py \
        --backend triton --precisions fp8 --fp8-fallback-dtype fp16 \
        --profile --profile-steps 10 --output fcsnn-fp8.json

单卡 Spikformer
~~~~~~~~~~~~~~~~

``spikformer_ti`` 使用 ``T=4``、输入 ``224``、eager、Triton LIF，在 RTX 5090 上的
端到端结果如下。训练行使用 1024 类仅用于满足当前 TE FP8 backward 的 16 对齐要求；
真实 1000 类 head 的 FP8 训练会触发 ``lda % 16 == 0``，当前不支持。

.. list-table::
    :header-rows: 1

    * - workload
      - precision path
      - step latency
      - throughput
      - 相对 FP16 / BF16 吞吐
      - peak allocated
    * - 推理：batch 64，1000 类
      - FP16
      - 16.220 ms
      - 3945.7 images/s
      - --
      - 1974.8 MiB
    * - 推理：batch 64，1000 类
      - BF16
      - 16.372 ms
      - 3909.1 images/s
      - --
      - 1974.8 MiB
    * - 推理：batch 64，1000 类
      - FP8 + FP32 fallback
      - 37.173 ms
      - 1721.7 images/s
      - 0.436x / 0.441x
      - 3377.3 MiB
    * - 推理：batch 64，1000 类
      - FP8 + FP16 fallback
      - 22.280 ms
      - 2872.6 images/s
      - 0.728x / 0.735x
      - 2004.8 MiB
    * - 推理：batch 64，1000 类
      - FP8 + BF16 fallback（默认）
      - 22.402 ms
      - 2856.9 images/s
      - 0.724x / 0.731x
      - 2004.8 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP16
      - 37.869 ms
      - 845.0 samples/s
      - --
      - 4508.2 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - BF16
      - 44.671 ms
      - 716.4 samples/s
      - --
      - 4511.5 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP8 + FP32 fallback
      - 59.538 ms
      - 537.5 samples/s
      - 0.636x / 0.750x
      - 7724.4 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP8 + FP16 fallback
      - 41.499 ms
      - 771.1 samples/s
      - 0.913x / 1.076x
      - 4398.4 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP8 + BF16 fallback（默认）
      - 48.654 ms
      - 657.7 samples/s
      - 0.778x / 0.918x
      - 4398.4 MiB

nsys 显示 FP8 autocast 只覆盖 TE 的 pointwise Conv1d/Linear；patch-stem Conv2d、
BatchNorm 和 LIF 输出都回到 FP32，产生额外的 elementwise、copy 和 layout kernel。
所以当前 Spikformer 应选择 FP16/BF16；不要用 FC-SNN 的 Triton 结果推断
Spikformer 也会加速。可以用下面的命令重新采集单卡 profile（``--profile``
通过 ``cudaProfilerApi`` 限定采集区间）：

.. code-block:: bash

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o spikformer-fp8 \
      uv run python benchmark/benchmark_snn_single_gpu.py case \
        --model spikformer_ti --phase inference --execution eager \
        --batch-size 64 --warmup 50 --steps 10 --profile --precision fp8 \
        --neuron-backend triton --fp8-fallback-dtype bf16 \
        --tensor-metadata spikformer-fp8.tensors.jsonl \
        --output spikformer-fp8.json

默认 BF16 fallback 将 FP8 训练/推理 latency 分别降低 18.3%/39.7%，但仍未跨过
BF16。显式 FP16 fallback 更快，但动态范围较小。该模型 requested FP8 tracked
dense-MAC coverage 为 41.98%。当前 RTX 5090 软件栈没有 FP8 Conv2d，不能用模拟
路径声称已验证更高覆盖率。

历史的 2 卡 Spikformer DDP 使用 5 个计时 step：

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

默认先用 BF16。FC-SNN 只有在使用 Triton LIF 且矩阵达到表中规模后才建议测试 FP8；
Spikformer 等 CNN/神经元占比较高的模型，在完成 FP32 边界优化前继续使用 FP16/BF16。
profile 必须以端到端 step 为准；FP8 的显存也可能高于 BF16，不应把它当作显存优化开关。

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
