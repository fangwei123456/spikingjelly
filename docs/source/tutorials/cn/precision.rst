训练与推理精度
==============

English version: :doc:`../en/precision`

精度设置统一写在 ``PrecisionConfig`` 中，并在创建 optimizer 之前生效。配置分为
两部分：

* ``mode`` 控制普通模型算子；``fp8`` 使用 NVIDIA Transformer Engine；
* ``triton_storage``、``triton_fwd`` 和 ``triton_bwd`` 控制已有 multi-step
  Triton IF/LIF/PLIF 神经元的状态存储和前、反向算术。

两部分可以单独使用。例如，普通层可以使用 BF16，而 Triton 神经元继续使用 FP32。

安装
----

BF16 和 FP16 只需要 PyTorch。模型级 FP8 需要 Transformer Engine：

.. code-block:: bash

    uv pip install --editable ".[fp8]"

Triton 神经元 mixed precision 另外需要：

.. code-block:: bash

    uv pip install --editable ".[triton]"

配置普通模型
------------

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
``PrecisionArtifacts`` 会在 FP16 下创建 GradScaler，所以四种 mode 可以共用上面的
训练循环。

检查转换结果
~~~~~~~~~~~~

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

Triton 神经元精度
-----------------

Triton 精度也在 ``prepare_model_for_precision`` 中设置，不需要给每个神经元增加构造
参数。下面的配置让普通层使用 BF16，Triton 神经元使用 BF16 存储和前向、FP32
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

分布式 FP8
----------

``distributed.vision`` 会在 DDP 包装和 optimizer 创建前准备精度，并用 DDP 进程组
同步 Transformer Engine 的 scaling metadata。模型 FP8 和 Triton 神经元 mixed
precision 目前只支持 DDP，且 TP=PP=1：

MCore LLM 不使用 ``PrecisionConfig``，仍由 MCore 自己配置 transformer 和
optimizer 的精度。

.. code-block:: bash

    uv run torchrun --standalone --nproc-per-node=2 \
        benchmark/vision_distributed.py \
        --model spikformer --dataset synthetic \
        --data-parallel ddp --precision fp8 \
        --image-size 128 --classes 1024 \
        --batch-size 32 --max-steps 10

这条命令用于检查 FP8 和 DDP 能否正常联合运行。是否更快仍取决于模型；下文的
Spikformer 测试中，FP16 和 BF16 都更快。

FP8 何时更快
------------

先看结论：FP8 的收益来自大矩阵，不来自 dtype 名称本身。普通 FC-SNN 训练和当前
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

默认先用 BF16。profile 显示对齐的 Linear/MLP 占主要耗时，并且矩阵维度接近上面的
交叉区间时，再测试 FP8。CNN、神经元或通信占比较高时，FP8 通常无法弥补转换和
metadata 开销。FP8 的显存也可能高于 BF16，不应把它当作显存优化开关。

可以用仓库 benchmark 在目标 GPU 上重测交叉点：

.. code-block:: bash

    uv run python benchmark/benchmark_fp8_training_inference.py \
        --batch-size 4096 --width 3200 --depth 8 --num-classes 3200 \
        --warmup 8 --training-steps 20 --inference-steps 40 --trials 3 \
        --precisions fp16 bf16 fp8 --baseline-precision bf16 \
        --output benchmark/output/fp8-vs-bf16.json

再把 ``--baseline-precision`` 改为 ``fp16``，分别检查两种 16-bit baseline。正式
训练还要在真实数据上比较收敛曲线；这个 benchmark 只检查 loss、输出和参数更新是否
有限，以及 steady-state 性能。
