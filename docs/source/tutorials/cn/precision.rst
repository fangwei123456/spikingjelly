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

Triton 神经元混合精度另外需要：

.. code-block:: bash

    uv pip install --editable ".[triton]"

在已经安装 PyTorch 的环境中，添加 ``--no-build-isolation``，使 Transformer
Engine 根据现有 ``torch.version.cuda`` 选择 CUDA 安装包。否则隔离构建环境可能临时
安装另一版 PyTorch，导致 CUDA 安装包与实际环境不一致：

.. code-block:: bash

    uv pip install --no-build-isolation \
        "transformer-engine[pytorch]>=2.16,<3"

路径一：自定义训练或推理流程
----------------------------

调用顺序不能颠倒：先把模型移到目标设备，再调用
``prepare_model_for_precision``，最后创建优化器。FP8 准备过程会替换部分模块；
如果优化器已经持有旧参数，后续训练不会更新新模块。

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
        PrecisionConfig(mode="bf16"),
    )
    model = precision.model
    optimizer = torch.optim.AdamW(model.parameters())

    optimizer.zero_grad(set_to_none=True)
    with precision.autocast_context():
        output = model(torch.randn(256, 4096, device=device))
        loss = output.square().mean()
    precision.backward(loss, optimizer)

``mode`` 可取 ``fp32``、``fp16``、``bf16`` 或 ``fp8``。返回的
``PrecisionArtifacts`` 会在 CUDA FP16 下创建 GradScaler。其 ``backward`` 方法
默认也会更新优化器。上例输出形状为 ``[256, 1024]``。推理时仍在 ``autocast_context`` 中调用模型，但不调用 ``backward``：

.. code-block:: python

    model.eval()
    with torch.inference_mode(), precision.autocast_context():
        output = model(torch.randn(256, 4096, device=device))

检查 FP8 转换结果
~~~~~~~~~~~~~~~~~

要在支持 FP8 的 CUDA GPU 上尝试 FP8，请重新准备模型，再创建新的优化器。
FP8 转换对齐的 ``torch.nn.Linear``、SpikingJelly ``layer.Linear``、逐点 Conv1d，以及支持的 LayerNorm 和相邻融合模式。Linear 输入维必须是 16 的倍数，输出维
必须是 8 的倍数。不满足对齐的层保留高精度并出现在诊断报告中：

.. code-block:: python

    precision = prepare_model_for_precision(
        model, device, PrecisionConfig(mode="fp8", fp8_recipe="auto")
    )
    model = precision.model
    report = precision.describe()
    print(report["conversion_report"])

Transformer Engine、硬件或数值策略不可用时，准备过程直接报错，不会改用 BF16。
模型中至少要有一个可转换模块。``fp8_recipe="auto"`` 使用当前 Transformer Engine
的默认数值策略；需要固定数值策略时，再显式选择 ``delayed``、``current``、``block``
或 ``mxfp8``。

控制未转换算子的 dtype
~~~~~~~~~~~~~~~~~~~~~~~

Transformer Engine 只改变已转换模块。FP8 默认对其余普通 CUDA 算子启用 BF16
自动混合精度上下文，控制 Conv2d、BatchNorm 和神经元之间的数据类型边界：

.. code-block:: python

    config = PrecisionConfig(
        mode="fp8",
        fp8_recipe="auto",
    )

你可以用 ``fp8_fallback_dtype="fp16"`` 或 ``"fp32"`` 显式覆盖。
该字段为未转换算子指定精度，控制普通 CUDA 自动混合精度上下文和
Transformer Engine 模块的输出边界，不会将这些算子转换成 FP8 内核。

部分 Transformer Engine 数值策略会把 FP8 元数据序列化为 pickle。恢复可信来源的
检查点时，才设置 ``NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1``；不要为未知
检查点开启该选项。

输入、状态和计算精度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

普通层的自动混合精度上下文控制层内运算；神经元还区分输入 dtype、跨时间步的状态存储
数据类型，以及前后向计算数据类型。下表说明多步 IFNode、LIFNode 和 ParametricLIFNode
的配置，其他神经元需查对应 API。

.. list-table:: 神经元精度配置
    :header-rows: 1
    :widths: 40 60

    * - 配置
      - 当前行为
    * - 普通 FP32 输入与状态
      - 根据设备和执行路径自动执行
    * - 普通 FP16/BF16 输入
      - 未配置状态精度时，膜电位跟随输入，已有 FP32 状态也会转换
    * - FP16/BF16 状态
      - 普通调用使用 Torch 参考递推，也支持 fullgraph 编译
    * - 显式 FP32 状态＋低精度输入
      - functional 可直接传入 FP32 初态；模块需要显式配置存储精度。数值策略不同于低精度状态
    * - 显式存储／前向／反向精度
      - CUDA Triton 功能，要求受支持的内置替代梯度与精度组合
    * - FP8 状态和计算
      - 实验性功能，受设备、格式和组合限制；不是所有神经元通用能力

普通层使用 BF16、神经元状态保留 FP32 时，先将模型移到目标设备，再准备精度，
然后创建优化器。下面的 CUDA 训练示例使用随机输入：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import neuron, functional, surrogate
    from spikingjelly.activation_based.precision import (
        PrecisionConfig,
        prepare_model_for_precision,
    )

    device = torch.device("cuda:0")
    model = nn.Sequential(
        nn.Linear(16, 16),
        neuron.LIFNode(step_mode="m", surrogate_function=surrogate.ATan()),
        nn.Linear(16, 4),
    ).to(device)
    precision = prepare_model_for_precision(
        model, device, PrecisionConfig(mode="bf16", neuron_storage="fp32")
    )
    optimizer = torch.optim.SGD(precision.model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast_context():
        output = precision.model(torch.rand(4, 2, 16, device=device))
        loss = output.float().square().mean()
    precision.backward(loss, optimizer)
    functional.reset_net(precision.model)

要分别设置存储、前向和反向精度，可将上例的 config 换成：

.. code-block:: python

    config = PrecisionConfig(
        mode="bf16", neuron_storage="bf16", neuron_fwd="bf16", neuron_bwd="fp32"
    )

显式神经元精度使用 Triton。``neuron_storage`` 可取 ``fp32``、``fp16``、``bf16``、
``float8_e4m3fn`` 或 ``float8_e5m2``；``neuron_fwd`` 或 ``neuron_bwd`` 取默认值
``fp32`` 之外的值时必须设置它；二者可取 ``fp8``、``fp16``、``bf16`` 或 ``fp32``，
但并非任意组合都有效。
FP8 算术要求状态存储为 ``float8_e4m3fn`` 或 ``float8_e5m2``。指数和敏感替代梯度
计算在内核中使用 FP32。七种受支持的替代梯度见 :doc:`./surrogate`。
不支持的显式配置报错，不以另一种数值策略替代。

执行查询只报告普通路径的选择，具体精度配置仍需检查其支持范围。先准备精度，再
编译模型；直接调用 functional 显式精度接口时，先预热相同组合，让设备检查在
图捕获前完成。

改变状态或递推精度可能影响脉冲、最终状态和梯度，采用新配置前应验证模型精度。
Inductor 还可能改变低精度运算的融合与舍入，因此结果可能与 eager 有逐位差异。
测速应保持相同状态策略，例如 ``--precision bf16 --neuron-storage fp32``。
自动执行和编译流程见 :doc:`./triton_backend`。

路径二：``distributed.vision``
--------------------------------

视觉高层接口在并行包装和优化器创建前读取配置中的 ``precision`` 字段。训练使用
``TrainingConfig``；评测和预测分别使用 ``EvaluationConfig`` 和
``PredictionConfig``，三者都接收同一个 ``PrecisionConfig``：

.. code-block:: python

    from spikingjelly.activation_based.precision import PrecisionConfig

    precision_config = PrecisionConfig(
        mode="bf16",
        neuron_storage="bf16",
        neuron_fwd="bf16",
        neuron_bwd="fp32",
    )

将 ``precision_config`` 传给视觉配置的 ``precision`` 字段。使用随机数据完成一次训练，可运行仓库中的命令行示例。
其中，``--precision`` 映射到 ``mode``，其余字段由
``--fp8-recipe``、``--neuron-storage``、``--neuron-fwd`` 和
``--neuron-bwd`` 设置：

.. code-block:: bash

    uv run torchrun --standalone --nproc-per-node=2 \
        benchmark/vision_distributed.py \
        --model spikformer --dataset synthetic \
        --data-parallel ddp --precision fp8 \
        --image-size 128 --classes 1024 \
        --batch-size 32 --max-steps 10

``distributed.vision`` 会在 DDP 包装和优化器创建前准备精度，并用 DDP 进程组
同步 Transformer Engine 的缩放元数据。模型 FP8 和显式神经元精度在训练时要求
DDP 且 TP=PP=1；评测和预测时要求副本模式、即时执行且 TP=PP=1。视觉流水线并行支持
FP32 和 BF16。

路径三：``distributed.llm``
-----------------------------

``distributed.llm`` 不使用 ``PrecisionConfig``。模型与优化器的精度分别由
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
        calculate_per_token_loss=True,
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
``bf16=False`` 和 ``params_dtype=torch.float16``。启用 MCore FP8 时，在 BF16
基线上给 ``TransformerConfig`` 设置 ``fp8="hybrid"`` 和相同的
``fp8_recipe``，同时让 ``OptimizerConfig.fp8_recipe`` 与之匹配。PP 开启时，
``pipeline_dtype`` 必须与 ``params_dtype`` 一致。

独立评测和使用缓存的生成复用模型中的 ``TransformerConfig``。SGLang 模型产物
导出目前要求 MCore BF16；完整的 LLM 模型、数据和训练配置见
:doc:`./distributed_training`。

如何选择
--------

先用 BF16，再在你的工作负载上比较 FP16 或 FP8。对于 FC-SNN，可测试参考交叉点
附近的矩阵规模，并显式选择神经元精度。比较完整训练或推理步骤，FP8 的显存占用
可能超过 BF16。

使用仓库基准脚本在目标 GPU 上重测交叉点：

.. code-block:: bash

    uv run python benchmark/benchmark_fp8_training_inference.py \
        --batch-size 4096 --width 3200 --depth 8 --num-classes 3200 \
        --warmup 8 --training-steps 20 --inference-steps 40 --trials 3 \
        --precisions fp16 bf16 fp8 --baseline-precision bf16 \
        --output benchmark/output/fp8-vs-bf16.json

将 ``--baseline-precision`` 改为 ``fp16``，可分别检查两种 16 位基线。正式
训练还要在真实数据上比较收敛曲线；这个基准脚本只检查损失、输出和参数更新是否
有限，以及稳态性能。

FP8 何时更快
------------

在目标 GPU 上比较端到端吞吐和峰值显存。FP8 的收益来自大矩阵运算，
而卷积和神经元运算可能占据 SNN 的大部分耗时。

Linear/GEMM 交叉点
~~~~~~~~~~~~~~~~~~

参考结果（2026-08-29，Transformer Engine 2.18.0）。两个比值依次为
``FP8 / FP16`` 和 ``FP8 / BF16``，以吞吐提高 5% 为交叉点门槛：

.. list-table::
    :header-rows: 1

    * - 工作负载（批大小、宽度、深度）
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

在这组稠密线性层测试中，固定 batch=4096、depth=8 后，推理交叉点位于
width 2304--2560，训练交叉点位于 width 3072--3200。固定 width=4096 时，训练
交叉点位于 batch 2048--3072。

显存占用也取决于工作负载。在 4096×3200×8 的交叉点，FP8 训练和推理已分配显存
比 FP16 低约 9%/14%，但比 BF16 高约 24%/34%。

端到端 SNN 与 DDP
~~~~~~~~~~~~~~~~~

参考结果（2026-08-30，RTX 5090 32 GiB，Transformer Engine 2.17.1）：

.. list-table::
    :header-rows: 1

    * - 工作负载
      - 训练 FP8 / FP16、BF16
      - 推理 FP8 / FP16、BF16
    * - FC-SNN：T16, batch 256, width 4096, depth 20（Triton LIF，FP16 回退精度）
      - 1.533x / 1.677x
      - 1.578x / 1.786x
    * - FC-SNN：T16, batch 256, width 8192, depth 10（Triton LIF，FP32 回退精度）
      - 1.544x / 1.468x
      - 1.648x / 1.471x

比值为端到端吞吐比值。测量自己的 FC-SNN 时，显式配置 FP32 神经元状态，
使用 Triton 精度路径：

.. code-block:: bash

    SJ_LIF_CUDA_IMPLEMENTATION=triton nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o fcsnn-fp8 \
      uv run python benchmark/benchmark_train_precision_snn_fc.py \
        --precisions fp8 --fp8-fallback-dtype fp16 --neuron-storage fp32 \
        --profile --profile-steps 10 --output fcsnn-fp8.json

单卡 Spikformer
~~~~~~~~~~~~~~~~

参考结果（2026-08-30，RTX 5090 32 GiB，Transformer Engine 2.17.1；
``spikformer_ti``，``T=4``，输入尺寸 224，即时执行）：

.. list-table::
    :header-rows: 1

    * - 工作负载
      - 精度配置
      - 每步耗时
      - 吞吐量
      - 相对 FP16 / BF16 吞吐
      - 峰值已分配显存
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
      - FP8 + FP32 回退精度
      - 37.173 ms
      - 1721.7 images/s
      - 0.436x / 0.441x
      - 3377.3 MiB
    * - 推理：batch 64，1000 类
      - FP8 + FP16 回退精度
      - 22.280 ms
      - 2872.6 images/s
      - 0.728x / 0.735x
      - 2004.8 MiB
    * - 推理：batch 64，1000 类
      - FP8 + BF16 回退精度（默认）
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
      - FP8 + FP32 回退精度
      - 59.538 ms
      - 537.5 samples/s
      - 0.636x / 0.750x
      - 7724.4 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP8 + FP16 回退精度
      - 41.499 ms
      - 771.1 samples/s
      - 0.913x / 1.076x
      - 4398.4 MiB
    * - 训练：batch 32，1024 类（对齐诊断）
      - FP8 + BF16 回退精度（默认）
      - 48.654 ms
      - 657.7 samples/s
      - 0.778x / 0.918x
      - 4398.4 MiB

训练配置使用 1024 类，以满足 Transformer Engine 2.17.1 的 FP8 反向对齐要求；
1000 类输出层在该版本触发了 ``lda % 16 == 0``。推理时 FP16 和 BF16 都快于所有 FP8 配置。训练时 FP16 最快；FP8 + FP16 回退精度快于 BF16，但仍慢于 FP16。
你可以采集自己的模型剖析记录：

.. code-block:: bash

    SJ_LIF_CUDA_IMPLEMENTATION=triton nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,cublas,osrt --sample=none --cpuctxsw=none \
      -o spikformer-fp8 \
      uv run python benchmark/benchmark_snn_single_gpu.py case \
        --model spikformer_ti --phase inference --execution eager \
        --batch-size 64 --warmup 50 --steps 10 --profile --precision fp8 \
        --fp8-fallback-dtype bf16 --neuron-storage fp32 \
        --tensor-metadata spikformer-fp8.tensors.jsonl \
        --output spikformer-fp8.json

参考结果（2026-08-29，双卡 Spikformer DDP，Transformer Engine 2.18.0，
5 个计时步骤）：

.. list-table::
    :header-rows: 1

    * - 每卡批大小
      - FP8 images/s
      - FP16 / BF16 images/s
      - FP8 / 16 位每卡已分配显存
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
