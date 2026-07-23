Transformer ANN2SNN 转换
========================

本页作者：`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/ann2snn_transformer`

本页介绍 ``spikingjelly.activation_based.ann2snn`` 中面向 Transformer 的 ANN2SNN 转换路径，包括 ``TransformerTDEquivalentRecipe`` 基线、基于 Spatio-Temporal Approximation (STA) [#sta]_ 的 ``STATransformerRecipe``、用于 SpikeZIP-compatible QANN 的 ``SpikeZIPTFQANNRecipe`` [#spikezip]_，以及基于校准的 ``Qwen2SNNRecipe``。如果要做经典 CNN 上的 ReLU-to-IFNode rate coding 转换，请阅读 :doc:`ann2snn`。

本教程按从简单到复杂的顺序组织四条 Transformer 转换路径：

* **Path 1: TransformerTDEquivalentRecipe 基线**。这是最直接的 TD-equivalent operator replacement 路径，用 BERT SST-2 说明语言分类模型在 embedding 输出之后的转换边界。
* **Path 2: Spatio-Temporal Approximation (STA) Transformer 转换**。这是 ``STATransformerRecipe`` 的模型级增强路径，沿用累计差分和显式 step-mode 读出思想，并进一步加入 dataloader 校准和 spike encoder；本页在 ViT-B/16 ImageNet 上展示完整结果。
* **Path 3: SpikeZIP QANN-to-SNN 转换**。这是 ``SpikeZIPTFQANNRecipe`` 的 module-tree 路径，输入必须已经是 SpikeZIP-compatible QANN；本页给出 synthetic RoBERTa parity 和 ViT-Small ImageNet 复现实验入口。
* **Path 4: Qwen2 离线多步转换**。``Qwen2SNNRecipe`` 校准 Hugging Face Qwen2 causal LM，并使用当前 SpikingJelly TD 算子和 signed activation-aware IF 神经元转换全部 decoder block。

.. warning::

    STA 转换不是严格意义上的 fully spike-driven SNN 转换。``mode="spiking_encoder"`` 的输出是 "整数脉冲数 × 校准阈值" 的量化值；膜电位为负时整数脉冲数也可以为负，因此输出不是二值脉冲 tensor。

    ``STATransformerRecipe`` 因此应视为 training-free 的 Transformer ANN2SNN 近似流程，不承诺 fully spike-driven LLM 转换。以整数 token 作为输入的语言模型需要额外定义输入和 embedding 的转换契约。转换后的 STA 模型是有状态的，遵循 SpikingJelly ``step_mode`` 的单步与多步执行语义。

.. warning::

    SpikeZIP 路径使用的 ST-BIF 神经元支持带符号三值输出，也不是严格的二值脉冲神经元。用严格 SNN 定义对比不同转换方法时，这一点需要单独说明。

Path 1: 差分等价基线
-------------------------------------------

``TransformerTDEquivalentRecipe`` 是一个不需要 dataloader 的替换路径：把当前支持的 Transformer 算子替换为 TD-equivalent 模块。它是本页最小的 Transformer 转换基线：不做 STA 校准，也不维护内部时间循环，只依赖已有 TD 算子表达累计差分。它不会产生脉冲，也不插入脉冲神经元，转换后模块输出的是浮点差分值。因此，它的产物并不是 SNN。

后面的 ``STATransformerRecipe`` 可以理解为同一累计差分和显式 step-mode 思路的模型级增强：它加入 dataloader 校准和 spike encoder，并面向完整 Transformer FX graph。二者不是同一个 recipe，也不是严格的 API 超集；它们的支持边界和实验含义不同。

在线差分
^^^^^^^^^^^^^^^^^^^^^^

Transformer 模型包含仿射投影、LayerNorm 或 RMSNorm、GELU 或 SiLU、attention、残差加法、mask 以及 tensor 常量等组件。ReLU-to-IFNode rate-coding 规则无法覆盖这些组件，需另辟蹊径。常用的方法是进行时间差分。

以普通 ANN 输入 :math:`x` 为例，可把它嵌入为如下差分输入序列：

.. math::

    x^{(0)} = x,\qquad x^{(t)} = 0,\quad t=1,\ldots,T-1.

时间步 :math:`t` 的累计输入为：

.. math::

    X^{(t)} = \sum_{\tau=0}^{t} x^{(\tau)}.

因此 :math:`X^{(0)} = X^{(1)} = \cdots = x`。令 :math:`f` 表示原 ANN 中的一个函数或模块，例如仿射投影、LayerNorm、GELU 或 attention block。转换后的 TD-equivalent 模块不是 :math:`f` 本身，而是 :math:`f` 在相邻累计输入上的差分。记该差分模块为 :math:`F_t`：

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

这个恒等式解释了 TD-equivalent 路径为什么成立：如果每个转换模块都只输出累计结果的差分，常量与 bias 只计算一次，那么把所有时间步的输出相加就恢复了 ANN 模块的输出。STA 的在线等价部分也复用这一思想。

在线等价路径现在遵循显式的 SpikingJelly step-mode 约定。具体来说，

* 原 ANN 的 ``model(x)`` 是一次完整的 ANN 推理，不包含时间展开；
* 转换后模型在 ``step_mode="s"`` 下的一次调用只执行一个差分时间步，外层时间循环需由用户编写；转换后的模块每次调用接收当前时间步的差分输入 :math:`x^{(t)}`，并在内部维护累计输入 :math:`X^{(t-1)}` 和累计输出 :math:`f(X^{(t-1)})`，以此实现差分计算；
* ``step_mode="m"`` 时，转换后模型接收首个维度为时间维的完整差分序列张量，并返回完整输出序列；此时通常能够沿时间维度并行（或向量化）计算，速度显著快于单步模式；

对普通 ANN 样本 :math:`x` 做推理时，常用的差分输入序列是 :math:`x^{(0)}=x`、后续时间步皆为 0，正如上面的例子所示。而如果输入本身已经是时间序列，则直接传入对应的差分序列。

BERT SST-2 示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransformerTDEquivalentRecipe`` 可以把 Transformer 中常见的核心算子转换为 TD-equivalent 有状态模块，使转换后的模型按照 SpikingJelly 的 ``step_mode`` 约定执行单步或多步推理。除下文的 BERT 算子外，recipe 也会把 ``nn.RMSNorm`` 转换为 ``TDRMSNorm``，把 ``nn.SiLU`` 和 ``torch.nn.functional.silu`` 转换为 ``TDSiLU``；这些算子的输出是浮点时间差分，而非二值脉冲。这里以 BERT 风格的 SST-2 分类为例，转换边界设在 embedding 输出之后：原 Hugging Face BERT embedding 层仍然接收整数 ``input_ids``；而 ANN2SNN 转换图从浮点 ``embedding_output`` 和 ``extended_attention_mask`` 起步，覆盖 encoder、pooler、dropout 和 classifier wrapper。完整可运行示例位于 ``spikingjelly.activation_based.ann2snn.examples.bert_sst2_transformer_td_equivalent``。

Hugging Face 依赖是可选依赖，只在运行该示例时安装：

.. code-block:: shell

    uv pip install transformers datasets

建议先运行一个小的验证切片：

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

转换产物遵循相同的显式 step-mode 约定：

.. code-block:: python

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)
    embedding_seq = torch.zeros(
        converted.time_steps,
        *embedding_output.shape,
        device=embedding_output.device,
        dtype=embedding_output.dtype,
    )
    embedding_seq[0] = embedding_output
    logits = converted(embedding_seq, extended_attention_mask).sum(dim=0)

``extended_attention_mask`` 是静态控制 tensor，传入时保持原样，不添加时间维。

下表数据在 NVIDIA A100-SXM4-80GB 上，使用 SST-2 validation split 的 872 条样本测得：

.. list-table:: BERT SST-2 Transformer TD-equivalent 转换结果
    :header-rows: 1
    :widths: 36 18 18 18

    * - 方法
      - 验证样本
      - 时间步
      - Accuracy (%)
    * - ANN
      - 872
      - -
      - 92.431
    * - ``TransformerTDEquivalentRecipe``
      - 872
      - 8
      - 92.431

Accuracy 下降 0.000 个百分点。评测前，示例会先在前两个 batch 上比对 FX-friendly wrapper 与原 Hugging Face classifier 的 logits。本次 ``batch_size=32`` 的完整运行中，ANN wrapper 推理耗时约 1.60 秒，转换模型约 9.20 秒。关键 stdout 行如下：

.. code-block:: shell

    HF_WRAPPER_PARITY {"checked_batches": 2, "max_abs_diff": 5.245208740234375e-06, "atol": 1e-05}
    BASELINE {"accuracy": 0.9243119266055045, "total": 872, "seconds": 1.5974977016448975}
    TRANSFORMER_TD_EQUIVALENT {"accuracy": 0.9243119266055045, "total": 872, "seconds": 9.204399347305298}
    DROP 0.0

Path 2: STA 转换
-----------------------------------------------

从 Path 1 的差分等价基线往前一步，Spatio-Temporal Approximation (STA) 扩展了差分思想。它仍然依赖显式 step-mode 和最终时间维求和读出，但在选定的非线性与 attention 输出处加入校准后的 spike encoder。STA 的在线差分等价部分就是 Path 1 中不带脉冲的 TD-equivalent 累计差分思路；本节只补充 STA 相比差分等价基线新增的 spike encoder、校准和模型级实验。

STA spike encoder
^^^^^^^^^^^^^^^^^

``STATransformerRecipe(mode="spiking_encoder")`` 会在选中的增量后加入 spike encoder。对模拟增量
:math:`a^{(t)}` 和阈值 :math:`V`，encoder 维护残差膜电位 :math:`r^{(t)}`。
初始残差为 0：

.. math::

    r^{(-1)} = 0.

每个时间步，encoder 先累加模拟增量：

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

也就是说，encoder 输出的脉冲之和等于模拟增量之和减去最终残差。若 :math:`a^{(t)}` 就是 STA 差分 :math:`F_t(X^{(t)})`，那么模拟增量之和就是 ANN 模块的输出 :math:`f(x)`，脉冲编码的结果与它只差最终残差。STA 校准阈值时使用的是 ``time_steps``，因此在激活范围固定时，更大的 :math:`T` 对应更细的时间量化。

借助在线累计差分，转换后的模型可以在累计输入上调用这些算子，再对输出增量做编码，从而在保留算子语义的同时在选定的输出位置引入脉冲式的时序通信。仿射模块、``LayerNorm``、``GELU``、``MultiheadAttention`` 和浮点 FX tensor 常量各自维护一份在线累计差分状态。bias 与图中的常量只注入一次。静态 attention mask 等控制 tensor 在各时间步保留，不会被置零。

本教程推荐的配置是 ``STATransformerRecipe(mode="spiking_encoder")``。该模式会在 ``LayerNorm``、``GELU`` 和 ``MultiheadAttention`` 的输出侧接入校准后的有状态 spike encoder，同时保持主干 affine projection 处于在线等价模式。阈值由 dataloader 校准，并依赖 ``time_steps``。

使用 Recipe
^^^^^^^^^^^^^^^^^^^^^^^^^

``STATransformerRecipe`` 是一个 FX graph recipe。最简 Python API 使用 ``Converter``（``FXConverter`` 的兼容名）执行转换：

.. code-block:: python

    from spikingjelly.activation_based import ann2snn

    recipe = ann2snn.STATransformerRecipe(
        dataloader=calibration_loader,
        time_steps=8,
        mode="spiking_encoder",
        threshold_mode="mse",
        threshold_scale=0.5,
    )
    converted = ann2snn.Converter(recipe=recipe, device="cuda:0").convert(model)
    converted.eval()

``time_steps`` 属于 recipe 参数，因为它参与阈值校准，也用于在转换图中展开那些无法从运行时输入推断序列长度的常量。

STA 当前有三种模式：

* ``equivalent``：无需校准的累计差分基线；
* ``spiking_encoder``：在非线性和 attention 输出上使用校准 spike encoder；
* ``spiking_affine``：当前 step-mode 对齐 STA 后端会明确拒绝。

本教程的模型级结果使用 ``spiking_encoder``。

Step-mode 执行
^^^^^^^^^^^^^^

``STATransformerRecipe`` 是 FX graph recipe，转换产物是 ``fx.GraphModule``，也就是 ``nn.Module`` 的子类。用户通过 ``functional.set_step_mode`` 递归设置内部模块的 step-mode。下面的单步示例展示普通 ANN 输入 ``x`` 的 ``[x, 0, 0, ...]`` 差分序列；每次 ``converted(x_t)`` 调用只是一个 STA 时间步，最后的 ``y`` 才对应原 ANN 级别的一次读出。如果已有时间序列输入，则按同样方式逐时间步传入对应差分。处理独立序列前需要重置状态：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "s")
    functional.reset_net(converted)

    y = None
    for t in range(converted.time_steps):
        x_t = x if t == 0 else torch.zeros_like(x)
        y_t = converted(x_t)
        y = y_t if y is None else y + y_t

如果要触发逐层多步加速，将转换产物切换为 ``step_mode="m"``，并输入第 0 维为时间维的序列 tensor：

.. code-block:: python

    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)

    x_zeros = torch.zeros_like(x).expand(converted.time_steps - 1, *x.shape)
    x_seq = torch.cat((x.unsqueeze(0), x_zeros), dim=0)
    y_seq = converted(x_seq)
    y = y_seq.sum(dim=0)

多输入模型同样需要用户显式构造浮点输入序列。``attn_mask`` 等具名静态控制 tensor 保持原样，不要额外添加时间维。最终累计读出由用户对输出时间维求和，或按输出结构递归求和。

多步后端的限制比任意 PyTorch 图更严格。当前会拒绝 ``mode="spiking_affine"``、``spike_linear=True``、``spike_conv2d=True``、请求或使用 attention weights 的 ``MultiheadAttention`` 调用、``key_padding_mask``、functional ``scaled_dot_product_attention``，以及不支持的 FX tensor 操作。遇到这些报错时，请将模型改写为受支持的 sequence-preserving 模块和操作。

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

下表数据在 NVIDIA A100-SXM4-80GB 上使用完整 50000 张 ImageNet 验证集测得：

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

Top-1 下降 0.368 个百分点。本次 ``batch_size=16`` 的完整运行中，ANN baseline 推理耗时约 181.6 秒，STA 转换模型耗时约 1834.0 秒。wall-clock time 对运行环境比较敏感；如需对比 single-step 和 multi-step 耗时，请使用专门的 step-mode benchmark。

关键 stdout 行如下：

.. code-block:: shell

    BASELINE {"top1": 0.81068, "top5": 0.95318, "total": 50000, "seconds": 181.58131194114685}
    STA_SPIKING_ENCODER_T8_S0p5 {"top1": 0.807, "top5": 0.95202, "total": 50000, "seconds": 1833.9626359939575}
    DROP 0.0036799999999999056

Path 3: SpikeZIP 转换
------------------------------------------

转换限制与 API
^^^^^^^^^^^^^^

``SpikeZIPTFQANNRecipe`` 将量化 ANN (QANN) 转化为 SNN。它要求输入模型必须已经是 SpikeZIP-compatible QANN。当前支持两类 attention module contract：

* RoBERTa 风格 self-attention 需要暴露 ``query``、``key``、``value`` linear layers，``num_attention_heads``、``attention_head_size``、``all_head_size``、``dropout``，以及带 ``s``、``sym``、``pos_max``、``neg_min``、``level`` 属性的 ``query_quan``、``key_quan``、``value_quan``、``attn_quan``、``after_attn_quan`` quantizers；
* ViT 风格 self-attention 需要暴露 ``qkv``、``proj`` linear layers，带上述量化属性的 ``quan_q``、``quan_k``、``quan_v``、``attn_quan``、``after_attn_quan``、``quan_proj`` quantizers，以及 ``num_heads``、``head_dim``、``scale``、``attn_drop`` 和 ``proj_drop``。

recipe 会把 QANN 侧的 quantizers 和 Transformer 算子替换为透明的 SNN 侧 ST-BIF、SESA attention 乘法、Spike-Softmax、Spike-LayerNorm、embedding 与 linear module。该算法直接针对 ``nn.Module`` 进行改动，不执行 FX tracing，故应使用 ``ModuleConverter``：

.. code-block:: python

    from spikingjelly.activation_based import ann2snn

    recipe = ann2snn.SpikeZIPTFQANNRecipe(
        time_steps=32,
        model_family="roberta",
    )
    converted = ann2snn.ModuleConverter(recipe=recipe, device="cuda:0").convert(qann)
    converted.eval()

用户可以通过 ``functional.set_step_mode`` 控制 single-step 或 multi-step 执行，并显式传入单个时间步或首维为时间维的序列 tensor。

SpikeZIP 使用的 ST-BIF 神经元只用于推理。它适合运行已经转换好的 QANN-to-SNN 模型，不应用于端到端梯度训练或微调。

Synthetic RoBERTa parity 示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节以 RoBERTa 为例，简单验证 SpikeZIP-compatible QANN 可被转换为 SNN 模型，并且用户显式累计后的 SNN logits 能与 QANN logits 对齐：

.. code-block:: shell

    python -m spikingjelly.activation_based.ann2snn.examples.roberta_spikezip_qann_synthetic \
      --device cpu \
      --time-steps 32 \
      --batch-size 3 \
      --seq-len 5 \
      --output benchmark/output/roberta_spikezip_qann_synthetic_cpu.json

该示例也接受 ``--qann-checkpoint``，用于加载同一 tiny QANN 架构保存出来的 ``state_dict``。真实 SpikeZIP checkpoint 需要先确保 QANN 模型的 RoBERTa-style attention 模块暴露 ``query_quan`` 等带 ``s``、``sym``、``pos_max``、``neg_min``、``level`` 属性的 quantizer，再传给 ``ModuleConverter(recipe=SpikeZIPTFQANNRecipe(...))``。

预期 stdout 会包含接近 0 的 parity error 和 ST-BIF state 摘要：

.. code-block:: shell

    {"accumulated_sequence_shape": [32, 3, 2], "max_abs_diff": 3.2782554626464844e-07, "mean_abs_diff": 8.630255621255856e-08, "recipe": "SpikeZIPTFQANNRecipe", "stbif_state": {"last_step_spike_values": [0.0], "max_accumulated": 0.75, "min_accumulated": -1.0}}

下表结果在 CPU 上通过上面的命令测得。它说明：只要输入已经是 SpikeZIP-compatible QANN，SpikeZIP 路径就能完成 RoBERTa 风格语言 Transformer 的转换。

.. list-table:: SpikeZIP synthetic RoBERTa QANN-to-SNN parity
    :header-rows: 1
    :widths: 24 14 14 18 18

    * - 模型
      - 时间步
      - Batch / 序列长度
      - Max abs diff
      - Mean abs diff
    * - Synthetic RoBERTa QANN
      - 32
      - 3 / 5
      - 3.278e-07
      - 8.630e-08

SpikeZIP ViT-Small ImageNet Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于真实 SpikeZIP-compatible ViT-Small QANN checkpoint，可以使用 ``benchmark/benchmark_ann2snn_spikezip_vit_qann.py`` 运行 ImageNet validation。该脚本会记录 QANN baseline 正确率、转换后 SNN 正确率、prediction agreement、logits 差异、耗时、CUDA 设备信息和峰值显存。

本教程使用的 ``vit-small-imagenet-relu-q32-81.59.pth`` 来自 SpikeZIP-TF 官方仓库：`ViT-Small-ReLU-Q32 pretrained QANN <https://github.com/Intelligent-Computing-Research-Group/SpikeZIP-TF>`_，对应 Hugging Face checkpoint 为 `XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32 <https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32>`_。官方 README 标注的 md5 前缀为 ``8207d3e``。下载后，把下面命令中的 ``--checkpoint`` 替换为该文件的本地路径，并把 ``--imagenet-root`` 替换为本地 ImageNet root：

.. code-block:: shell

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python benchmark/benchmark_ann2snn_spikezip_vit_qann.py \
      --checkpoint /path/to/vit-small-imagenet-relu-q32-81.59.pth \
      --imagenet-root /path/to/ImageNet \
      --device cuda:0 \
      --time-steps 64 \
      --step-mode m \
      --stbif-backend triton \
      --batch-size 16 \
      --snn-batch-size 4 \
      --samples 50000 \
      --output benchmark/output/spikezip_vit_small_imagenet_t64.json

下表的完整验证结果在 ``g2`` 的 NVIDIA A100-SXM4-80GB 上测得，ImageNet validation 使用全部 50,000 个样本：

.. list-table:: SpikeZIP ViT-Small ImageNet QANN-to-SNN 结果
    :header-rows: 1
    :widths: 24 14 14 16 16 18

    * - 模型
      - 时间步
      - 样本数
      - Top-1 (%)
      - Top-5 (%)
      - 说明
    * - QANN baseline
      - -
      - 50000
      - 81.476
      - 95.922
      - checkpoint baseline
    * - SpikeZIP SNN
      - 64
      - 50000
      - 81.566
      - 96.034
      - ``step_mode="m"``，``stbif_backend="triton"``

Top-1 差值为 ``+0.090`` 个百分点，Top-1 prediction agreement 为 ``97.34%``。QANN 推理耗时 ``79.20`` 秒；SNN 推理耗时 ``3719.15`` 秒（``61.99`` 分钟），使用 ``snn_batch_size=4``，峰值 CUDA allocated memory 为 ``25.63`` GiB。benchmark 中 ``parity_pass=false`` 是因为该字段使用 ``torch.allclose(snn_logits, qann_logits, atol=parity_atol, rtol=1e-5)`` 检查 logits；本次记录的 ``parity_atol`` 为 ``1e-4``。

Path 4: Qwen2 离线多步转换
--------------------------

``Qwen2SNNRecipe`` 是面向 evaluation-mode Hugging Face Qwen2 causal LM 的
``ModuleConverter`` recipe。可用 ``uv pip install -e '.[qwen]'`` 安装固定版本的
可选依赖。校准阶段生成可复用的 ``Qwen2SNNCalibration``；转换阶段使用当前
SpikingJelly 的 TD RMSNorm、TD linear、TD SiLU/product、TD SDPA 和基于
``ActivationAwareIFNode`` 的 ``SignedQCFSSequenceEncoder`` 替换完整 decoder。

转换后的 SNN 显式使用 ``[T,B,S,H]`` 时间布局，并按照
``layerwise_offline_multistep`` 调度。因此总调度更接近 :math:`L T`，而不是在线
T 步推理；本路径不声称支持 Qwen single-step。独立 prompt 前应调用
``functional.reset_net(model)``，KV cache continuation 通过 ``past_key_values``
显式传递。

公共 Qwen2 路径只提供真实的 ``signed_if`` 时间序列执行（以及用于等价性检查的
``exact_td``），质量评测不会用 count-domain 快路径替代脉冲序列。

转换后得到什么模型
^^^^^^^^^^^^^^^^^^^^

转换不会改变 Qwen2 的 token embedding、decoder 层数、hidden size、GQA head
布局、LM head 的权重与维度或原有的权重共享关系，也不会训练新的权重。它冻结
全部参数，并把每个 decoder block 中的 RMSNorm、linear、SiLU/SwiGLU product
和 SDPA 换成前文介绍的 TD 实现。在这些 TD 算子之间，Qwen2 recipe 新增
signed QCFS 编码边界：

LM head 的模块表示也会复制为 ``TDLinear``，但最终 forward 在沿 T 求和后使用其
保留的权重计算 logits；权重数值、维度以及与 embedding 的共享关系不变。

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

一个有 :math:`L` 个 decoder block 的转换模型包含 :math:`4L+1` 个 signed
encoder：embedding 后 1 个，每个 block 内 Q、K、V 和 MLP 中间激活各 1 个。
每个 signed encoder 又包含一对正、负
``ActivationAwareIFNode``，所以共有 :math:`8L+2` 个 IFNode 模块实例。每个
实例在整个输入张量上逐元素运行，表示一组共享逐通道阈值的神经元，而不是单个
标量神经元。

这是一种混合的显式多步模型。真正的二值发放发生在 signed encoder 内；TD
RMSNorm、linear、SiLU/product 和 attention 传播的是浮点 temporal difference，
其元素可以为负，并不都是 ``0/1`` 脉冲。这里的“Qwen SNN”具体指在选定激活
边界使用真实 IF 动力学和 spike-count 表示，同时用 TD 算子保持 Transformer
运算语义，而不是把 Qwen 的所有运算都变成二值事件。

产物模型只支持只读推理：必须先调用 ``eval()``，并在 ``torch.no_grad()`` 或
``torch.inference_mode()`` 中执行。training mode 或启用 autograd 时，
``forward()`` 会直接报错。

Signed QCFS encoder 与 IF 神经元
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SignedQCFSSequenceEncoder`` 是编码规则及其执行容器，内部的一对
``ActivationAwareIFNode`` 才是实际运行膜电位动力学的模块。给定静态 signed
激活 :math:`x`，encoder 先拆成

.. math::

    x^+ = \operatorname{ReLU}(x), \qquad
    x^- = \operatorname{ReLU}(-x).

正、负 IF 分别接收长度为 :math:`T` 的恒定电流序列：

.. math::

    I_t^+ = \frac{x^+}{T}, \qquad
    I_t^- = \frac{x^-}{T}, \qquad 0 \leq t < T.

也就是说，调用者只向 ``encode(x)`` 传入一次静态激活，但 IF 神经元看到的是
T 个 :math:`x/T`，而不是第 0 步收到 :math:`x`、其余步收到 0。每个 IF 使用
逐通道阈值 :math:`s`、偏移 :math:`s/2`、无泄漏积分和 soft reset：

.. math::

    H_t = V_{t-1} + I_t,

.. math::

    S_t = \Theta(H_t + s/2 - s),

.. math::

    V_t = H_t - S_t s.

偏移 :math:`s/2` 使总脉冲计数对应舍入，而 soft reset 从膜电位减去阈值而不是
清零，从而保留减去阈值后的余量。两个 IF 的二值输出按 :math:`s` 缩放并相减：

.. math::

    Y_t = (S_t^+ - S_t^-)s.

因此 encoder 的接口输出是显式序列 ``[T,...]``，每个元素属于
:math:`\{-s,0,s\}`，其时间和满足

.. math::

    \sum_{t=0}^{T-1}Y_t
    =
    \operatorname{clamp}\left(
        \operatorname{round}\left(\frac{x}{s}\right), -T, T
    \right)s.

实现还会校正 BF16 累积和 round-to-even 边界处偶发的计数不一致，使实际时间和
遵守上式。这个修正不改变常规输入使用 :math:`x/T` 恒定电流的事实。

T 维由谁消费
^^^^^^^^^^^^^^

QCFS encoder 只承诺产生 ``[T,...]``；是否以及何时沿 T 求和，由下游模块决定。
当前 Qwen2 产物模型有三种消费方式：

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 下游位置
     - 如何消费输入
     - 输出
   * - TD 算子和 residual
     - 消费完整 T 步；TD 算子可在内部使用累计状态
     - 仍为 ``[T,...]``
   * - 下一个 signed QCFS 边界
     - 先用 ``sequence.sum(0)`` 恢复聚合激活，再重新编码
     - 一条新的 ``[T,...]`` 脉冲序列
   * - KV cache 或最终 LM head
     - 沿 T 求和后缓存或计算 logits
     - 不再含 T 维的浮点张量

不要把 TD 算子内部为执行算子而计算的 ``cumsum`` 与 QCFS 边界的
``sum(dim=0)`` 混淆：前者仍返回 T 个 temporal difference，后者真正折叠时间
维，然后启动一次新的 signed spike-count 编码。

decoder block 之间也不会自动折叠 T。一个 block 的 TD down projection 与
residual 输出完整 ``[T,B,S,H]``，并原样进入下一个 block；到下一 block 的
Q/K/V 编码边界时才分别求和并重新编码。类似地，一个 block 内 attention 输出
保持 T，直到 MLP 中间激活的 signed encoder 才再次求和并重新编码。因此该模型
不是一条脉冲从 embedding 在线穿过全部层的流水线，而是“带 T 传播若干 TD
算子，再聚合和重新编码”的逐层离线调度。

``exact_td`` 与 ``signed_if``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

两个 execution mode 使用相同的 TD decoder，但在每个编码边界采取不同操作：

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
     - 正、负 IF 分别接收 ``ReLU(x)/T`` 与 ``ReLU(-x)/T`` 后产生的 signed spikes
     - 是
     - 重建为量化并裁剪后的 :math:`x`

``exact_td`` 分支不会调用 ``SignedQCFSSequenceEncoder`` 或其中的 IF 神经元；
模型对象虽然仍持有这些模块，但该次 forward 会绕过它们。因此 ``exact_td`` 与
dense Qwen 之间只应剩下浮点和算子实现误差，而 ``signed_if`` 还包含每个编码
边界的舍入和饱和误差。

当前公共 recipe 没有“把 ``[x,0,...,0]`` 送入 IF 并继续运行 T 步”的第三种
模式。对于当前无泄漏、soft-reset、每步至多发放一次的 IF，这种 front-loaded
输入在理想算术下可以得到与恒定 :math:`x/T` 输入相同的总脉冲计数，但发放时刻
不同，也需要相同的 round-to-even 边界处理。它仍然是量化并受 T 限幅的 IF
路径，不能等同于不经过神经元的 ``exact_td``。

T、校准 levels 与量化误差
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen2 校准会为每个通道估计绝对激活边界 :math:`R`。当
``calibration_quantile < 1`` 时，:math:`R` 是指定分位数；否则是最大值。
逐通道 QCFS 步长由

.. math::

    s = \frac{R}{\text{calibration\_levels}}

决定。为简化公式，这里假设 :math:`R \geq 10^{-6}`；实现会先用
``clamp_min(1e-6)`` 防止零 scale。T 决定一个二值 IF 在一次编码窗口中最多
发放多少个脉冲。于是

.. math::

    \text{rounding error} \leq \frac{s}{2}
    \quad \text{when } |x| \leq Ts, \qquad
    |x|_{\max} = Ts.

配置要求 ``0 < calibration_levels <= time_steps``。这样 :math:`Ts` 至少覆盖
校准边界 :math:`R`；若 T 小于 levels，连 :math:`R` 本身都会被截断。

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
动态范围并细化量化间隔，还必须按比例增大 T，或采用其他 scale/range 设计。

这一点与前文某些把阈值直接绑定到 ``time_steps`` 的转换路径不同：Qwen2 recipe
显式分离 ``time_steps`` 和 ``calibration_levels``。阅读后文结果表时，应把 T
理解为计数容量，把 levels 理解为校准范围内的分辨率；最终配置是质量与成本的
实验工作点，并不证明该 T 是理论最优值。

最小校准与推理示例
^^^^^^^^^^^^^^^^^^^^

下面的示例从本地加载 Qwen2.5 checkpoint，校准逐通道 scale，转换完整 decoder，
并执行真实的多步路径。这里的小型 prompt 集只用于演示 API；若要复现后文质量结果，
需使用 ``benchmark/snn_llm/qwen_conversion`` 中固定的 WikiText-2 校准协议。

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
        neuron_backend="triton",
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
    snn = ann2snn.ModuleConverter(recipe, device=device).convert(ann)
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

校准状态只包含张量和基础 Python 值，可用
``Qwen2SNNCalibration.from_state_dict(torch.load(..., weights_only=True))``
恢复；校准元数据必须与转换配置完全一致。Torch 神经元 backend 用作参考实现，
Triton backend 只支持 CUDA 多步推理。

正式质量证据
^^^^^^^^^^^^^^

固定的 BF16 评测使用完整 WikiText-2 test split（584 个窗口、298937 个 target
tokens）和六个 zero-shot 任务：LAMBADA、PIQA、HellaSwag、WinoGrande、
ARC-Easy、ARC-Challenge。``平均下降`` 和 ``最大下降`` 表示六任务中
dense accuracy 减去 SNN accuracy 的百分点。

.. list-table:: Qwen2.5 Base 离线多步质量
   :header-rows: 1

   * - 模型
     - T / levels / quantile
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

这些结果只证明离线逐层调度的质量，不代表延迟或能耗优势。在 A100 实测中，高 T
路径仍显著慢于 dense Qwen。3B checkpoint 受 Qwen Research License 约束，
SpikingJelly 不重新分发该权重。``benchmark/snn_llm/qwen_conversion/README.md``
记录了本表对应的固定 revision、验收阈值、report digest、正确性证据和性能
provenance。

.. [#sta] Jiang Y., et al., "Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers", ICLR 2024.

.. [#spikezip] You K., et al., "SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN", ICML 2024.
