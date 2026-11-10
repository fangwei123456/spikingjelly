Transformer ANN2SNN 转换
========================

作者：`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/ann2snn_transformer`

要用 ``spikingjelly.activation_based.ann2snn`` 转换 Transformer，先根据你的模型选择转换方案： ``TransformerTDEquivalentRecipe`` 基线、基于 Spatio-Temporal Approximation (STA) [#sta]_ 的 ``STATransformerRecipe``、用于兼容 SpikeZIP 的 QANN 的 ``SpikeZIPTFQANNRecipe`` [#spikezip]_，以及基于校准的 ``Qwen2SNNRecipe``。如果要做经典 CNN 上的 ReLU 到 IF 神经元率编码转换，请阅读 :doc:`ann2snn`。

.. list-table::
    :header-rows: 1

    * - 转换方案
      - 内部表示
      - 是否二值
      - 执行方式
    * - ``TransformerTDEquivalentRecipe``
      - 浮点差分
      - 否
      - 单步／多步
    * - ``STATransformerRecipe``
      - 有符号整数计数 × 阈值
      - 否
      - 单步／多步
    * - ``SpikeZIPTFQANNRecipe``
      - 有符号三值
      - 否
      - 单步／多步
    * - ``Qwen2SNNRecipe``
      - 有符号 IF 脉冲与浮点差分
      - 否
      - 离线多步

以下示例用于推理。

路径 1： 差分等价基线
-------------------------------------------

先用 ``TransformerTDEquivalentRecipe`` 检查时间差分（TD）等价性。它将受支持的 Transformer 算子替换为输出相邻累计结果之差的模块。时间序列由你提供，无需校准数据加载器。

``STATransformerRecipe`` 在累计差分的基础上加入校准和脉冲编码器。两种转换方案支持的计算图操作不同，选择前需检查模型中的算子。

在线差分
^^^^^^^^^^^^^^^^^^^^^^

Transformer 模型包含仿射投影、LayerNorm 或 RMSNorm、GELU 或 SiLU、注意力、残差加法、掩码以及张量常量等组件。ReLU 到 IF 神经元率编码规则无法覆盖这些组件。你可以用时间差分来处理这些运算。

以普通 ANN 输入 :math:`x` 为例，可把它嵌入为如下差分输入序列：

.. math::

    x^{(0)} = x,\qquad x^{(t)} = 0,\quad t=1,\ldots,T-1.

时间步 :math:`t` 的累计输入为：

.. math::

    X^{(t)} = \sum_{\tau=0}^{t} x^{(\tau)}.

因此 :math:`X^{(0)} = X^{(1)} = \cdots = x`。令 :math:`f` 表示原 ANN 中的一个函数或模块，例如仿射投影、LayerNorm、GELU 或注意力块。TD 等价模块输出 :math:`f` 在相邻累计输入上的差分。记该差分模块为 :math:`F_t`：

.. math::

    F_t\left(X^{(t)}\right)
    =
    f\left(X^{(t)}\right) - f\left(X^{(t-1)}\right),
    \qquad f\left(X^{(-1)}\right) = 0.

在实现中，:math:`F_t` 由包裹原操作的有状态包装模块实现：它计算或复用 :math:`f` 的累计输出，缓存上一时间步的累计输出，并只返回当前差分 :math:`\Delta y^{(t)} = F_t(X^{(t)})`。

累计输出满足：

.. math::

    \sum_{t=0}^{T-1} F_t\left(X^{(t)}\right)
    = f\left(X^{(T-1)}\right) - f\left(X^{(-1)}\right)
    = f(x).

这个恒等式解释了 TD 等价路径为什么成立：如果每个转换模块都只输出累计结果的差分，常量与偏置只计算一次，那么把所有时间步的输出相加就恢复了 ANN 模块的输出。STA 的在线等价部分也复用这一思想。

按 SpikingJelly 的步进模式约定准备输入：

* 原 ANN 的 ``model(x)`` 是一次完整的 ANN 推理，不包含时间展开；
* 转换后模型在 ``step_mode="s"`` 下的一次调用只执行一个差分时间步，外层时间循环需由用户编写；转换后的模块每次调用接收当前时间步的差分输入 :math:`x^{(t)}`，并在内部维护累计输入 :math:`X^{(t-1)}` 和累计输出 :math:`f(X^{(t-1)})`，以此实现差分计算；
* ``step_mode="m"`` 时，转换后模型接收首个维度为时间维的完整差分序列张量，并返回完整输出序列；TD 算子可沿时间维批量计算累计输入。

对普通 ANN 样本 :math:`x` 做推理时，常用的差分输入序列是 :math:`x^{(0)}=x`、后续时间步皆为 0，正如上面的例子所示。而如果输入本身已经是时间序列，则直接传入对应的差分序列。

BERT SST-2 示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransformerTDEquivalentRecipe`` 可以把 Transformer 中常见的核心算子转换为 TD 等价有状态模块，使转换后的模型按照 SpikingJelly 的 ``step_mode`` 约定执行单步或多步推理。除下文的 BERT 算子外，转换方案也会把 ``nn.RMSNorm`` 转换为 ``TDRMSNorm``，把 ``nn.SiLU`` 和 ``torch.nn.functional.silu`` 转换为 ``TDSiLU``；这些算子的输出是浮点时间差分，而非二值脉冲。这里以 BERT 风格的 SST-2 分类为例，转换边界设在嵌入层输出之后：原 Hugging Face BERT 嵌入层仍然接收整数 ``input_ids``；而 ANN2SNN 转换图从浮点 ``embedding_output`` 和 ``extended_attention_mask`` 起步，覆盖编码器、池化器、随机失活层和分类器包装模型。完整可运行示例位于 ``spikingjelly.activation_based.ann2snn.examples.bert_sst2_transformer_td_equivalent``。

Hugging Face 依赖是可选依赖，只在运行该示例时安装：

.. code-block:: shell

    uv pip install transformers datasets

先用少量验证样本检查流程：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.bert_sst2_transformer_td_equivalent \
      --model-name-or-path textattack/bert-base-uncased-SST-2 \
      --dataset-name nyu-mll/glue \
      --dataset-config sst2 \
      --split validation \
      --device cuda:0 \
      --batch-size 32 \
      --eval-samples 256 \
      --time-steps 8 \
      --output benchmark/output/bert_sst2_transformer_td_equivalent_t8_small.json

完整 SST-2 验证时去掉 ``--eval-samples`` 即可：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.bert_sst2_transformer_td_equivalent \
      --model-name-or-path textattack/bert-base-uncased-SST-2 \
      --dataset-name nyu-mll/glue \
      --dataset-config sst2 \
      --split validation \
      --device cuda:0 \
      --batch-size 32 \
      --time-steps 8 \
      --output benchmark/output/bert_sst2_transformer_td_equivalent_t8_full.json

示例脚本通过 ``TransformerTDEquivalentRecipe(time_steps=8)`` 构建 ``converted``。取得嵌入输出和静态掩码后，推理步骤如下：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)
    embedding_seq = torch.zeros(
        converted.time_steps,
        *embedding_output.shape,
        device=embedding_output.device,
        dtype=embedding_output.dtype,
    )
    embedding_seq[0] = embedding_output
    with torch.no_grad():
        logits = converted(embedding_seq, extended_attention_mask).sum(dim=0)

``extended_attention_mask`` 是静态控制张量，传入时保持原样，不添加时间维。

参考结果（提交 `0e24f7d3 <https://github.com/fangwei123456/spikingjelly/commit/0e24f7d3>`_，NVIDIA A100-SXM4-80GB）：

.. list-table:: BERT SST-2 Transformer TD-equivalent 转换结果
    :header-rows: 1
    :widths: 36 18 18 18

    * - 方法
      - 验证样本
      - 时间步
      - 准确率 (%)
    * - ANN
      - 872
      - -
      - 92.431
    * - ``TransformerTDEquivalentRecipe``
      - 872
      - 8
      - 92.431

准确率下降 0.000 个百分点。评测前，示例会先在前两个批次上比对适合 FX 追踪的包装模型与原 Hugging Face 分类器的分类得分。本次 ``batch_size=32`` 的完整运行中，ANN 包装模型推理耗时约 1.60 秒，转换模型约 9.20 秒。对应输出如下：

.. code-block:: shell

    HF_WRAPPER_PARITY {"checked_batches": 2, "max_abs_diff": 5.245208740234375e-06, "atol": 1e-05}
    BASELINE {"accuracy": 0.9243119266055045, "total": 872, "seconds": 1.5974977016448975}
    TRANSFORMER_TD_EQUIVALENT {"accuracy": 0.9243119266055045, "total": 872, "seconds": 9.204399347305298}
    DROP 0.0

路径 2： STA 转换
-----------------------------------------------

STA 沿用路径 1 的累计差分、显式步进模式和时间维求和读出，并在选定的非线性与注意力输出处加入校准后的脉冲编码器。此转换方案使用累计浮点算子和校准后的输出编码器，未复现论文完整的 UGO/TCSA 流程；下文结果针对本实现。

STA 脉冲编码器
^^^^^^^^^^^^^^^^^

``STATransformerRecipe(mode="spiking_encoder")`` 会在选中的增量后加入脉冲编码器。对模拟增量
:math:`a^{(t)}` 和阈值 :math:`V`，编码器维护残差膜电位 :math:`r^{(t)}`。
初始残差为 0：

.. math::

    r^{(-1)} = 0.

每个时间步，编码器先累加模拟增量：

.. math::

    u^{(t)} = r^{(t-1)} + a^{(t)}.

然后计算可以发放多少个阈值单位：

.. math::

    n^{(t)} = \operatorname{trunc}\left(\frac{u^{(t)}}{V}\right),
    \qquad
    s^{(t)} = n^{(t)} V.

其中 :math:`s^{(t)}` 是当前时间步的量化输出。下一时间步的残差为：

.. math::

    r^{(t)} = u^{(t)} - s^{(t)}.

从 SNN 角度看，:math:`r^{(t)}` 是发放后保留下来的膜电位，:math:`n^{(t)}` 是整数脉冲数，膜电位为负时它也可以为负。:math:`s^{(t)}` 是按阈值加权后的脉冲输出。更新式 :math:`r^{(t)} = u^{(t)} - s^{(t)}` 相当于一种广义的软重置：从膜电位里减去本步输出的阈值加权值。当 :math:`n^{(t)} = 1` 时退化为普通软重置；更大的正整数或负整数表示一个时间步内跨过多个阈值单位。

经过 :math:`T` 个时间步后：

.. math::

    \sum_{t=0}^{T-1} s^{(t)}
    =
    \sum_{t=0}^{T-1} a^{(t)}
    - r^{(T-1)}

也就是说，编码器输出的脉冲之和等于模拟增量之和减去最终残差。若 :math:`a^{(t)}` 就是 STA 差分 :math:`F_t(X^{(t)})`，那么模拟增量之和就是 ANN 模块的输出 :math:`f(x)`，脉冲编码的结果与它只差最终残差。STA 校准阈值时使用的是 ``time_steps``，因此在激活范围固定时，更大的 :math:`T` 对应更细的时间量化。

借助在线累计差分，转换后的模型可以在累计输入上调用这些算子，再对输出增量做编码，从而在保留算子语义的同时在选定的输出位置引入脉冲式的时序通信。仿射模块、``LayerNorm``、``GELU``、``MultiheadAttention`` 和浮点 FX 张量常量各自维护一份在线累计差分状态。偏置与图中的常量只注入一次。静态注意力掩码等控制张量在各时间步保留，不会被置零。

使用 ``STATransformerRecipe(mode="spiking_encoder")``。该模式会在 ``LayerNorm``、``GELU`` 和 ``MultiheadAttention`` 的输出侧接入校准后的有状态脉冲编码器，同时保持主干仿射投影处于在线等价模式。阈值由数据加载器校准，并依赖 ``time_steps``。

使用转换方案
^^^^^^^^^^^^^^^^^^^^^^^^^

先在一个小型 LayerNorm/GELU 网络上试用 STA。用一批浮点输入校准，再用 ``FXConverter`` 执行转换：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import ann2snn

    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.GELU()).eval()
    x = torch.randn(2, 3, 4)
    calibration_loader = [x]
    recipe = ann2snn.STATransformerRecipe(
        dataloader=calibration_loader,
        time_steps=8,
        mode="spiking_encoder",
        threshold_mode="mse",
        threshold_scale=0.5,
    )
    converted = ann2snn.FXConverter(recipe=recipe, device="cpu").convert(model)
    converted.eval()

``time_steps`` 属于转换方案参数，因为它参与阈值校准，也用于在转换图中展开那些无法从运行时输入推断序列长度的常量。

选择受支持的 STA 模式：

* ``equivalent``：无需校准的累计差分基线；
* ``spiking_encoder``：在非线性和注意力输出上使用校准脉冲编码器。

步进后端会拒绝 ``spiking_affine``。下面的校准模型使用 ``spiking_encoder``。

步进模式执行
^^^^^^^^^^^^^^

``STATransformerRecipe`` 是 FX 计算图转换方案，转换产物是 ``fx.GraphModule``，也就是 ``nn.Module`` 的子类。用户通过 ``functional.set_step_mode`` 递归设置内部模块的步进模式。下面的单步示例展示普通 ANN 输入 ``x`` 的 ``[x, 0, 0, ...]`` 差分序列；每次 ``converted(x_t)`` 调用只是一个 STA 时间步，最后的 ``y`` 才对应原 ANN 级别的一次读出。如果已有时间序列输入，则按同样方式逐时间步传入对应差分。处理独立序列前需要重置状态：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "s")
    functional.reset_net(converted)

    with torch.no_grad():
        y = None
        for t in range(converted.time_steps):
            x_t = x if t == 0 else torch.zeros_like(x)
            y_t = converted(x_t)
            y = y_t if y is None else y + y_t

要逐层执行多步计算，将转换产物切换为 ``step_mode="m"``，并输入第 0 维为时间维的序列张量：

.. code-block:: python

    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)

    x_zeros = torch.zeros_like(x).expand(converted.time_steps - 1, *x.shape)
    x_seq = torch.cat((x.unsqueeze(0), x_zeros), dim=0)
    with torch.no_grad():
        y_seq = converted(x_seq)
        y = y_seq.sum(dim=0)

多输入模型同样需要用户显式构造浮点输入序列。``attn_mask`` 等具名静态控制张量保持原样，不要额外添加时间维。最终累计读出由用户对输出时间维求和，或按输出结构递归求和。

多步后端的限制比任意 PyTorch 图更严格。当前会拒绝 ``mode="spiking_affine"``、``spike_linear=True``、``spike_conv2d=True``、请求或使用注意力权重的 ``MultiheadAttention`` 调用、``key_padding_mask``、functional ``scaled_dot_product_attention``，以及不支持的 FX 张量操作。遇到这些报错时，请将模型改写为受支持的保持序列结构的模块和操作。

ViT-B/16 ImageNet 示例
^^^^^^^^^^^^^^^^^^^^^^

完整可运行示例位于 ``spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta``。脚本会读取 ``torchvision.models.vit_b_16``（``ViT_B_16_Weights.DEFAULT``）和一个可以用 ``torchvision.datasets.ImageFolder`` 直接加载的 ImageNet 验证集目录。

下面的命令假设 ``/path/to/imagenet/val`` 目录中按类别分子目录，且需要 CUDA：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta \
      --data-root /path/to/imagenet/val \
      --device cuda:0 \
      --batch-size 16 \
      --num-workers 8 \
      --calib-samples 2048 \
      --time-steps 8 \
      --threshold-scale 0.5

只看环境能否跑通时，可以用小批量样本快速验证：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta \
      --data-root /path/to/imagenet/val \
      --device cuda:0 \
      --batch-size 8 \
      --num-workers 2 \
      --calib-samples 32 \
      --eval-samples 32 \
      --time-steps 8 \
      --threshold-scale 0.5

参考结果（提交 `d7ed92d8 <https://github.com/fangwei123456/spikingjelly/commit/d7ed92d8>`_，NVIDIA A100-SXM4-80GB）：

.. list-table:: ViT-B/16 ImageNet STA 转换结果
    :header-rows: 1
    :widths: 30 16 16 14 18 18

    * - 方法
      - 校准样本
      - 验证样本
      - 时间步
      - Top-1 (%)
      - Top-5 (%)
    * - ANN
      - -
      - 50000
      - -
      - 81.068
      - 95.318
    * - STA ``spiking_encoder``
      - 2048
      - 50000
      - 8
      - 80.700
      - 95.202

Top-1 下降 0.368 个百分点。本次 ``batch_size=16`` 的完整运行中，ANN 基线推理耗时约 181.6 秒，STA 转换模型耗时约 1834.0 秒。如需对比单步和多步耗时，请使用专门的步进模式评测。

对应输出如下：

.. code-block:: shell

    BASELINE {"top1": 0.81068, "top5": 0.95318, "total": 50000, "seconds": 181.58131194114685}
    STA_SPIKING_ENCODER_T8_S0p5 {"top1": 0.807, "top5": 0.95202, "total": 50000, "seconds": 1833.9626359939575}
    DROP 0.0036799999999999056

路径 3： SpikeZIP 转换
------------------------------------------

转换限制与 API
^^^^^^^^^^^^^^

``SpikeZIPTFQANNRecipe`` 将量化 ANN（QANN） 转化为 SNN。它要求输入模型必须已经是兼容 SpikeZIP 的 QANN。当前支持两类注意力模块约定：

* RoBERTa 风格自注意力需要暴露 ``query``、``key``、``value`` 线性层，``num_attention_heads``、``attention_head_size``、``all_head_size``、``dropout``，以及带 ``s``、``sym``、``pos_max``、``neg_min``、``level`` 属性的 ``query_quan``、``key_quan``、``value_quan``、``attn_quan``、``after_attn_quan`` 量化器；
* ViT 风格自注意力需要暴露 ``qkv``、``proj`` 线性层，带上述量化属性的 ``quan_q``、``quan_k``、``quan_v``、``attn_quan``、``after_attn_quan``、``quan_proj`` 量化器，以及 ``num_heads``、``head_dim``、``scale``、``attn_drop`` 和 ``proj_drop``。

转换方案将 QANN 侧的量化器和 Transformer 算子替换为 SNN 侧的 ST-BIF、SESA 注意力乘法、Spike-Softmax、Spike-LayerNorm、嵌入层与线性模块。该算法复制模型，再替换 ``nn.Module`` 树中的模块，不执行 FX 追踪，故应使用 ``ModuleConverter``：

.. code-block:: python

    from spikingjelly.activation_based import ann2snn
    from spikingjelly.activation_based.ann2snn.examples.roberta_spikezip_qann_synthetic import (
        TinyQRobertaClassifier,
    )

    qann = TinyQRobertaClassifier().eval()
    recipe = ann2snn.SpikeZIPTFQANNRecipe(
        time_steps=32,
        model_family="roberta",
    )
    converted = ann2snn.ModuleConverter(recipe=recipe, device="cpu").convert(qann)
    converted.eval()

用户可以通过 ``functional.set_step_mode`` 控制单步或多步执行，并显式传入单个时间步或首维为时间维的序列张量。

SpikeZIP 使用的 ST-BIF 神经元只用于推理。构造参数会被复制并与计算图分离；前向输入、状态或缓冲区需要梯度时会抛出异常，Torch、Triton 和原生 CUDA 路径都没有实现反向传播。它适合运行已经转换好的 QANN 到 SNN 模型，不应用于端到端梯度训练或微调。

合成 RoBERTa 一致性示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行合成 RoBERTa 示例，将累计的 SNN 分类得分与原 QANN 的得分进行比较：

.. code-block:: shell

    python -m spikingjelly.activation_based.ann2snn.examples.roberta_spikezip_qann_synthetic \
      --device cpu \
      --time-steps 32 \
      --batch-size 3 \
      --seq-len 5 \
      --output benchmark/output/roberta_spikezip_qann_synthetic_cpu.json

该示例也接受 ``--qann-checkpoint``，用于加载同一微型 QANN 架构保存出来的 ``state_dict``。真实 SpikeZIP 权重文件需要先确保 QANN 模型的 RoBERTa 风格注意力模块暴露 ``query_quan`` 等带 ``s``、``sym``、``pos_max``、``neg_min``、``level`` 属性的量化器，再传给 ``ModuleConverter(recipe=SpikeZIPTFQANNRecipe(...))``。

预期标准输出会包含接近 0 的一致性误差和 ST-BIF 状态摘要：

.. code-block:: shell

    {"accumulated_sequence_shape": [32, 3, 2], "max_abs_diff": 3.2782554626464844e-07, "mean_abs_diff": 8.630255621255856e-08, "recipe": "SpikeZIPTFQANNRecipe", "stbif_state": {"max_accumulated": 0.75, "min_accumulated": -1.0}}

用最大绝对误差检查转换后的分类得分是否在你设定的容差内。

参考结果（提交 `39ba6e93 <https://github.com/fangwei123456/spikingjelly/commit/39ba6e93>`_，CPU）：

.. list-table:: SpikeZIP synthetic RoBERTa QANN-to-SNN parity
    :header-rows: 1
    :widths: 24 14 14 18 18

    * - 模型
      - 时间步
      - 批量 / 序列长度
      - 最大绝对误差
      - 平均绝对误差
    * - 合成 RoBERTa QANN
      - 32
      - 3 / 5
      - 3.278e-07
      - 8.630e-08

SpikeZIP ViT-Small ImageNet 评测
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于真实兼容 SpikeZIP 的 ViT-Small QANN 权重文件，可以使用 ``benchmark/benchmark_ann2snn_spikezip_vit_qann.py`` 运行 ImageNet 验证集。该脚本会记录 QANN 基线正确率、转换后 SNN 正确率、预测一致率、分类得分差异、耗时、CUDA 设备信息和峰值显存。

参考结果使用的 ``vit-small-imagenet-relu-q32-81.59.pth`` 来自 SpikeZIP-TF 官方仓库：`ViT-Small-ReLU-Q32 pretrained QANN <https://github.com/Intelligent-Computing-Research-Group/SpikeZIP-TF>`_，对应 Hugging Face 权重文件为 `XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32 <https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32>`_。官方 README 标注的 md5 前缀为 ``8207d3e``。下载后，把下面命令中的 ``--checkpoint`` 替换为该文件的本地路径，并把 ``--imagenet-root`` 替换为本地 ImageNet 根目录：

.. code-block:: shell

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python benchmark/benchmark_ann2snn_spikezip_vit_qann.py \
      --checkpoint /path/to/vit-small-imagenet-relu-q32-81.59.pth \
      --imagenet-root /path/to/ImageNet \
      --device cuda:0 \
      --time-steps 64 \
      --step-mode m \
      --batch-size 16 \
      --snn-batch-size 4 \
      --samples 50000 \
      --output benchmark/output/spikezip_vit_small_imagenet_t64.json

参考结果（提交 `39ba6e93 <https://github.com/fangwei123456/spikingjelly/commit/39ba6e93>`_，NVIDIA A100-SXM4-80GB）：

.. list-table:: SpikeZIP ViT-Small ImageNet QANN-to-SNN 结果
    :header-rows: 1
    :widths: 24 14 14 16 16 18

    * - 模型
      - 时间步
      - 样本数
      - Top-1 (%)
      - Top-5 (%)
      - 说明
    * - QANN 基线
      - -
      - 50000
      - 81.476
      - 95.922
      - 权重基线
    * - SpikeZIP SNN
      - 64
      - 50000
      - 81.566
      - 96.034
      - ``step_mode="m"``，CUDA 实现自动选择

Top-1 差值为 ``+0.090`` 个百分点，Top-1 预测一致率为 ``97.34%``。QANN 推理耗时 ``79.20`` 秒；SNN 推理耗时 ``3719.15`` 秒（``61.99`` 分钟），使用 ``snn_batch_size=4``，峰值 CUDA 已分配显存为 ``25.63`` GiB。评测中 ``parity_pass=false`` 是因为该字段使用 ``torch.allclose(snn_logits, qann_logits, atol=parity_atol, rtol=1e-5)`` 检查分类得分；本次记录的 ``parity_atol`` 为 ``1e-4``。

路径 4： Qwen2 离线多步转换
---------------------------

``Qwen2SNNRecipe`` 是面向评估模式下的 Hugging Face Qwen2 因果语言模型的
``ModuleConverter`` 转换方案。可用 ``uv pip install -e '.[qwen]'`` 安装固定版本的
可选依赖。校准阶段生成可复用的 ``Qwen2SNNCalibration``；转换阶段使用当前
SpikingJelly 的 TD RMSNorm、TD 线性层、TD SiLU/逐元素乘积、TD SDPA 和基于
``ActivationAwareIFNode`` 的 ``SignedQCFSSequenceEncoder`` 替换完整解码器。

转换后的 SNN 显式使用 ``[T,B,S,H]`` 时间布局，并按照
``layerwise_offline_multistep`` 调度。对于 :math:`L` 层解码器，逐层执行 T 步的总调度约为 :math:`L T`。
独立提示词前应调用
``functional.reset_net(model)``，键值缓存续接通过 ``past_key_values``
显式传递。

使用 ``signed_if`` 执行显式脉冲序列推理，使用 ``exact_td`` 检查 TD 等价性，
使用 ``qcfs_sg`` 进行可微计数重建。下文参考质量结果使用 ``signed_if``。

转换后得到什么模型
^^^^^^^^^^^^^^^^^^^^

转换不会改变 Qwen2 的词元嵌入、解码器层数、隐藏维度、分组查询注意力头
布局、语言模型输出头的权重与维度或原有的权重共享关系，也不会训练新的权重。
在这些 TD 算子之间，Qwen2 转换方案新增有符号 QCFS 编码边界：

语言模型输出头的模块表示也会复制为 ``TDLinear``，但最终前向传播在沿 T 求和后使用其
保留的权重计算分类得分；权重数值、维度以及与嵌入层的共享关系不变。

.. code-block:: text

    token ids
        |
        v
    embedding [B,S,H]
        |
        v
    input SignedQCFSSequenceEncoder
        |
        v
    [T,B,S,H]
        |
        +------------------------------------------------------+
        | L x converted decoder block                          |
        |                                                      |
        | TD RMSNorm -> TD Q/K/V projections                   |
        |             -> RoPE on Q/K; V unchanged              |
        |             -> Q/K/V signed QCFS re-encoding         |
        |             -> TD SDPA -> TD output projection       |
        |             -> residual                              |
        |             -> TD RMSNorm                            |
        |             -> TD gate/up -> TD SiLU/product         |
        |             -> MLP signed QCFS re-encoding           |
        |             -> TD down projection -> residual        |
        +------------------------------------------------------+
        |
        v
    final TD RMSNorm [T,B,S,H]
        |
        v
    sum over T -> LM head -> logits [B,S,V]

一个有 :math:`L` 个解码器块的转换模型包含 :math:`4L+1` 个有符号
编码器：嵌入层后 1 个，每个块内 Q、K、V 和 MLP 中间激活各 1 个。
每个有符号编码器又包含一对正、负
``ActivationAwareIFNode``，所以共有 :math:`8L+2` 个 IFNode 模块实例。每个
实例在整个输入张量上逐元素运行，表示一组共享逐通道阈值的神经元。

二值发放发生在有符号编码器内部。TD RMSNorm、线性层、SiLU/逐元素乘积和注意力
传播可为负的浮点时间差分。模型在选定激活边界使用 IF 动力学，并结合浮点
Transformer 运算。

运行下文的推理示例时，必须先调用 ``eval()``，并在 ``torch.no_grad()`` 或
``torch.inference_mode()`` 中执行。转换保留可训练参数；公共模型还支持用于后训练的
可微 QCFS 计数重建模式 ``qcfs_sg``。

有符号 QCFS 编码器与 IF 神经元
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SignedQCFSSequenceEncoder`` 是编码规则及其执行容器，内部的一对
``ActivationAwareIFNode`` 才是实际运行膜电位动力学的模块。给定静态有符号
激活 :math:`x`，编码器先拆成

.. math::

    x^+ = \operatorname{ReLU}(x), \qquad
    x^- = \operatorname{ReLU}(-x).

正、负 IF 分别接收长度为 :math:`T` 的恒定电流序列：

.. math::

    I_t^+ = \frac{x^+}{T}, \qquad
    I_t^- = \frac{x^-}{T}, \qquad 0 \leq t < T.

也就是说，调用者只向 ``encode(x)`` 传入一次静态激活，但 IF 神经元看到的是
T 个 :math:`x^+/T` 或 :math:`x^-/T`。每个 IF 使用
逐通道阈值 :math:`s`、偏移 :math:`s/2`、无泄漏积分和减法重置：

.. math::

    H_t = V_{t-1} + I_t,

.. math::

    S_t = \Theta(H_t + s/2 - s),

.. math::

    V_t = H_t - S_t s.

偏移 :math:`s/2` 使总脉冲计数对应舍入。减法重置从膜电位减去阈值，保留余量。两个 IF 的二值输出按 :math:`s` 缩放并相减：

.. math::

    Y_t = (S_t^+ - S_t^-)s.

因此编码器的接口输出是显式序列 ``[T,...]``，每个元素属于
:math:`\{-s,0,s\}`，其时间和满足

.. math::

    \sum_{t=0}^{T-1}Y_t
    =
    \operatorname{clamp}\left(
        \operatorname{round}\left(\frac{x}{s}\right), -T, T
    \right)s.

实现还会校正 BF16 累积和舍入到最近偶数边界处偶发的计数不一致，使实际时间和
遵守上式。这个修正不改变常规输入使用 :math:`x/T` 恒定电流的事实。

T 维由谁消费
^^^^^^^^^^^^^^

QCFS 编码器只承诺产生 ``[T,...]``；是否以及何时沿 T 求和，由下游模块决定。
当前 Qwen2 产物模型有三种消费方式：

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 下游位置
     - 如何消费输入
     - 输出
   * - TD 算子和残差
     - 消费完整 T 步；TD 算子可在内部使用累计状态
     - 仍为 ``[T,...]``
   * - 下一个有符号 QCFS 边界
     - 先用 ``sequence.sum(0)`` 恢复聚合激活，再重新编码
     - 一条新的 ``[T,...]`` 脉冲序列
   * - 键值缓存或最终语言模型输出头
     - 沿 T 求和后缓存或计算分类得分
     - 不再含 T 维的浮点张量

TD 算子内部的 ``cumsum`` 产生 T 个累计输入。算子应用 ANN 函数后，再对结果
求差分，得到 T 个输出。QCFS 边界处的 ``sum(dim=0)`` 则移除时间维，随后启动
一次新的有符号脉冲计数编码。

解码器块之间也不会自动折叠 T。一个块的 TD 降维投影与
残差输出完整 ``[T,B,S,H]``，并原样进入下一个块；到下一块的
Q/K/V 编码边界时才分别求和并重新编码。类似地，一个块内注意力输出
保持 T，直到 MLP 中间激活的有符号编码器才再次求和并重新编码。执行时，T 步序列依次通过一组 TD 算子，再聚合和重新编码，随后继续传播。

选择编码模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

三种编码模式共用解码器模块，在各编码边界生成不同的序列：

.. list-table::
   :header-rows: 1
   :widths: 20 34 18 28

   * - mode
     - 边界产生的序列
     - 是否执行 IF
     - 数值含义
   * - ``exact_td``
     - ``[x, 0, ..., 0]``
     - 否
     - 保留浮点 :math:`x`，用于 TD 等价性检查
   * - ``signed_if``
     - 正、负 IF 分别接收 ``ReLU(x)/T`` 与 ``ReLU(-x)/T`` 后产生的有符号脉冲
     - 是
     - 重建为量化并裁剪后的 :math:`x`
   * - ``qcfs_sg``
     - ``[reconstruct(x), 0, ..., 0]``，首步为 QCFS 计数重建值
     - 否，跳过神经元时间循环
     - 以多级脉冲计数替代梯度可微地重建同样的量化并裁剪后的 :math:`x`

``exact_td`` 分支不会调用 ``SignedQCFSSequenceEncoder`` 或其中的 IF 神经元；
模型对象虽然仍持有这些模块，但该次前向传播会绕过它们。因此 ``exact_td`` 与
稠密 Qwen 之间只应剩下浮点和算子实现误差，而 ``signed_if`` 还包含每个编码
边界的舍入和饱和误差。``qcfs_sg`` 跳过神经元时间循环并以可微方式重建同一个计数，
其零尾序列用于可微训练；下文质量结果使用 ``signed_if``。

T、校准量化等级与量化误差
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen2 校准会为每个通道估计绝对激活边界 :math:`R`。当
``calibration_quantile < 1`` 时，:math:`R` 是指定分位数；否则是最大值。
逐通道 QCFS 步长由

.. math::

    s = \frac{R}{\text{calibration\_levels}}

决定。为简化公式，这里假设 :math:`R \geq 10^{-6}`；实现会先用
``clamp_min(1e-6)`` 防止零尺度。T 决定一个二值 IF 在一次编码窗口中最多
发放多少个脉冲。于是

.. math::

    \text{rounding error} \leq \frac{s}{2}
    \quad \text{when } |x| \leq Ts, \qquad
    |x|_{\max} = Ts.

配置要求 ``0 < calibration_levels <= time_steps``。这样 :math:`Ts` 至少覆盖
校准边界 :math:`R`；若 T 小于量化等级，连 :math:`R` 本身都会被截断。

因此在 ``calibration_levels`` 固定时，增大 T 不会让相邻可表示值之间的间隔
:math:`s` 变小；它扩大的是动态范围，并降低激活被裁剪到 :math:`\pm Ts` 的
概率。发生裁剪时，总误差可以远大于 :math:`s/2`。例如教程示例使用
``T=160``、``levels=16``、``quantile=0.999``，此时

.. math::

    s = R_{0.999}/16, \qquad Ts = 10R_{0.999}.

与 ``T=16`` 相比，``T=160`` 的标称舍入精度相同，但可表示范围从
:math:`\pm R_{0.999}` 扩大到约 :math:`\pm 10R_{0.999}`。这个设计为校准样本
之外的尾部激活，以及 Q/K/V、SwiGLU 和多次重新编码中的分布偏移预留裕量，意在
用更高的多步计算和显存成本减少饱和误差。若只增大
``calibration_levels``，:math:`s` 与 :math:`Ts` 会同时缩小；若要保持当前
动态范围并细化量化间隔，还必须按比例增大 T，或采用其他尺度与范围设计。

这一点与前文某些把阈值直接绑定到 ``time_steps`` 的转换路径不同：Qwen2 转换方案
显式分离 ``time_steps`` 和 ``calibration_levels``。阅读后文结果表时，应把 T
理解为计数容量，把量化等级理解为校准范围内的分辨率；选择时需结合这两个值，权衡量化误差与计算开销。

最小校准与推理示例
^^^^^^^^^^^^^^^^^^^^

加载本地 Qwen2.5 权重，校准逐通道尺度，再转换解码器。两个短提示词可用于试用
接口。评估质量时，请选用有代表性的文本，并遵循
``benchmark/snn_llm/qwen_conversion/README.md`` 中的校准流程。

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spikingjelly.activation_based import ann2snn, functional

    device = torch.device("cuda:0")
    model_root = "/path/to/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    ann = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    config = ann2snn.Qwen2SNNConfig(
        time_steps=160,
        calibration_levels=16,
        calibration_quantile=0.999,
    )
    calibration_text = [
        "Spiking neural networks communicate with discrete events.",
        "Qwen2 conversion uses an explicit temporal sequence.",
    ]
    encoded = tokenizer(
        calibration_text, padding=True, truncation=True, return_tensors="pt"
    )
    calibration_batches = [
        {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
    ]
    calibration = ann2snn.calibrate_qwen2_snn(
        ann, calibration_batches, config
    )
    torch.save(calibration.state_dict(), "qwen2_snn_calibration.pt")

    recipe = ann2snn.Qwen2SNNRecipe(calibration, config)
    snn = ann2snn.ModuleConverter(recipe, device=device).convert(ann).eval()
    del ann

    batch = tokenizer("SpikingJelly is", return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    functional.reset_net(snn)
    with torch.inference_mode():
        output = snn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoding_mode="signed_if",
            use_cache=False,
        )
    next_token = output.logits[:, -1].argmax(dim=-1)
    print(tokenizer.batch_decode(next_token.unsqueeze(-1)))

校准状态只包含张量和基础 Python 值，可用
``Qwen2SNNCalibration.from_state_dict(torch.load(..., weights_only=True))``
恢复；校准元数据必须与转换配置完全一致。CPU 使用 Torch 参考实现；CUDA 自动选择兼容的 ActivationAwareIF 推理实现。

参考准确率
^^^^^^^^^^^^^^

固定的 BF16 评测使用完整 WikiText-2 测试集（584 个窗口、298937 个目标词元）和六个零样本任务：LAMBADA、PIQA、HellaSwag、WinoGrande、
ARC-Easy、ARC-Challenge。``平均下降`` 和 ``最大下降`` 表示六任务中
稠密模型准确率减去 SNN 准确率的百分点。

参考结果（提交 `c98d8ac4 <https://github.com/fangwei123456/spikingjelly/commit/c98d8ac4>`_，NVIDIA A100）：

.. list-table:: Qwen2.5 Base 离线多步质量
   :header-rows: 1

   * - 模型
     - T / 量化等级 / 分位数
     - Dense PPL
     - SNN PPL
     - PPL 退化
     - 平均 / 最大下降 (pp)
   * - 0.5B
     - 160 / 16 / 0.999
     - 11.509
     - 12.453
     - 8.20%
     - 2.19 / 5.78
   * - 1.5B
     - 512 / 128 / 1.0
     - 8.160
     - 8.232
     - 0.87%
     - 0.99 / 3.16
   * - 3B
     - 512 / 16 / 0.999
     - 7.095
     - 7.703
     - 8.57%
     - 0.41 / 1.96

在 A100 实测中，高 T 的离线逐层执行显著慢于稠密 Qwen。选择配置时，应同时
检查推理时间和质量。3B 权重文件受 Qwen Research License 约束，
SpikingJelly 不重新分发该权重。评估运行方式见
``benchmark/snn_llm/qwen_conversion/README.md``。

.. [#sta] Jiang Y., et al., "Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers", ICLR 2024.

.. [#spikezip] You K., et al., "SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN", ICML 2024.
