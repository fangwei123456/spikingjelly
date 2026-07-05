基于 STA 的 Transformer ANN2SNN 转换
=====================================

本页作者：`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/ann2snn_transformer`

本页介绍 ``spikingjelly.activation_based.ann2snn`` 中面向 Transformer 的 ANN2SNN 转换路径，核心对象是 ``STATransformerRecipe``，它是一个基于 Spatio-Temporal Approximation (STA) [#sta]_ 的 training-free 转换 recipe。

如果要做经典 CNN 上的 ReLU-to-IFNode rate coding 转换，请阅读
:doc:`ann2snn`。本页介绍的是独立的 Transformer 转换流程。

.. warning::

    STA 转换不是严格意义上的 fully spike-driven SNN 转换。在``mode="spiking_encoder"`` 中，模块输出的是“整数脉冲数乘以校准阈值”的量化值；当残差为负时，这个整数脉冲数也可以为负。它不是二值脉冲 tensor。用严格 SNN 定义比较不同方法时，这一点很重要，也容易引起争议。

    因此，``STATransformerRecipe`` 应理解为一种 training-free Transformer ANN2SNN 近似转换流程，而不是完整 fully spike-driven LLM conversion 的承诺。以整数 token 为输入的语言模型需要额外定义 input 和 embedding 的转换契约。转换后的 STA 模型是有状态模块，并遵循 SpikingJelly ``step_mode`` 的单步和多步执行语义。

STA 转换思想
------------

Transformer 模型包含仿射投影、LayerNorm、GELU、attention、残差加法、mask 和 tensor 常量等组件。直接套用 ReLU-to-IFNode rate coding 规则不足以覆盖这些组件。 STA 用在线时序近似来处理它们。

不带脉冲的在线差分
^^^^^^^^^^^^^^^^^^

先看不带 spike encoder 的在线等价路径，核心概念是累计激活。令 :math:`x` 为原始 ANN 输入，构造输入序列：

.. math::

    x^{(0)} = x,\qquad x^{(t)} = 0,\quad t=1,\ldots,T-1.

第 :math:`t` 个时间步后的累计输入为：

.. math::

    X^{(t)} = \sum_{\tau=0}^{t} x^{(\tau)}.

因此 :math:`X^{(0)} = X^{(1)} = \cdots = x`。令 :math:`f` 表示转换前 ANN 中的一个函数或模块，例如仿射投影、LayerNorm、GELU 或 attention 模块。转换后的 STA 模块不是 :math:`f` 本身。记转换后在单个时间步上执行的差分模块为
:math:`F_t`，它输出 :math:`f` 在相邻累计输入上的差分：

.. math::

    F_t\left(X^{(t)}\right)
    =
    f\left(X^{(t)}\right) - f\left(X^{(t-1)}\right),
    \qquad f\left(X^{(-1)}\right) = 0.

在实现中，:math:`F_t` 由包裹原操作的有状态包装模块实现：它计算或复用
:math:`f` 的累计输出，保存上一时间步的累计输出，并只返回当前差分
:math:`\Delta y^{(t)} = F_t(X^{(t)})`。

累计输出满足：

.. math::

    \sum_{t=0}^{T-1} F_t\left(X^{(t)}\right)
    = f\left(X^{(T-1)}\right) - f\left(X^{(-1)}\right)
    = f(x).

这个恒等式解释了 STA 中在线等价部分为什么成立：如果每个转换模块都输出累计结果的差分，并且常量与 bias 只计算一次，那么把所有时间步输出相加即可恢复 ANN 模块的输出。

不带脉冲编码的在线等价路径遵循 SpikingJelly 其它模块相同的 step-mode 执行约定：

* 第 0 个时间步输入原始 ANN 输入；
* 后续时间步输入零值浮点 tensor；
* 有状态转换模块输出累计结果的差分；
* ``step_mode="s"`` 只执行一个时间步，外层时间循环由用户自己写；
* ``step_mode="m"`` 接收第 0 维为时间维的序列 tensor，并返回完整输出序列。

单步模式的高层执行流程如下：

.. code-block:: text

    reset converted state
    y = 0
    for t in range(time_steps):
        if t == 0:
            x_t = original_input
        else:
            x_t = zeros_like(original_input)
        y = y + converted_graph_step(x_t, static_control_tensors)
    return y

多步模式直接把完整序列交给转换后的模型：

.. code-block:: text

    reset converted state
    x_seq = [original_input, zeros_like(original_input), ...]
    y_seq = converted_graph(x_seq, static_control_tensors)
    y = sum over the time dimension of y_seq

这里，每一次单步调用或序列中的每个元素都对应 STA 的一个时间步。转换后的 FX 计算图包含有状态模块，会记住上一时间步的累计输出，因此每个时间步只返回当前增量。

带脉冲的 spike encoder
^^^^^^^^^^^^^^^^^^^^^^^

``mode="spiking_encoder"`` 会在选中的增量后加入 spike encoder。对模拟增量
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

从 SNN 角度看，:math:`r^{(t)}` 是发放后保留下来的膜电位，
:math:`n^{(t)}` 是整数脉冲数；当残差为负时，它也可以为负。
:math:`s^{(t)}` 是按阈值加权后的脉冲输出。更新式
:math:`r^{(t)} = u^{(t)} - s^{(t)}` 相当于广义的软重置：它从膜电位中减去
本步输出的阈值加权值。当 :math:`n^{(t)} = 1` 时，这退化为普通软重置；更大的正整数或负整数表示在一个时间步内跨过多个阈值单位。

经过 :math:`T` 个时间步后：

.. math::

    \sum_{t=0}^{T-1} s^{(t)}
    =
    \sum_{t=0}^{T-1} a^{(t)}
    - r^{(T-1)}

也就是说，encoder 的输出和模拟增量之和只差最终残差。如果
:math:`a^{(t)}` 就是 STA 差分 :math:`F_t(X^{(t)})`，那么模拟增量之和就是
ANN 模块输出 :math:`f(x)`，脉冲编码后的结果与它之间的差异就是最终残差。由于 STA 校准阈值时会使用 ``time_steps``，在激活范围固定时，更大的
:math:`T` 对应更细的时间量化。

这一点对 Transformer 很关键：LayerNorm、GELU 和 attention 都不是简单的 ReLU rate coding 层。在线累计差分视角允许转换模型在累计输入上计算这些函数，再对输出增量做编码，从而在保留算子语义的同时在选中的输出处引入脉冲式的时序通信。

仿射模块、``LayerNorm``、``GELU``、``MultiheadAttention`` 和浮点 FX tensor 常量都维护在线累计差分状态。Bias 和图中的常量只注入一次。静态 attention mask 等控制 tensor 会在各时间步保留，而不是被置零。

本教程推荐的公开路径是 ``mode="spiking_encoder"``。该模式会在 ``LayerNorm``、``GELU`` 和 ``MultiheadAttention`` 的输出侧加入校准后的有状态 spike encoder，同时保持主干 affine projection 在线等价。阈值来自 dataloader 校准，并依赖 ``time_steps``。

使用 STATransformerRecipe
--------------------------

最小 Python API 和其它 ANN2SNN recipe 一样，遵循 Recipe + Converter 结构：

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

``time_steps`` 属于 recipe 参数，因为它参与阈值校准，以及图中无法从运行时输入推断长度的常量序列展开。

STA 当前有三种模式：

* ``equivalent``：无需校准的累计差分基线；
* ``spiking_encoder``：在非线性和 attention 输出上使用校准 spike encoder；
* ``spiking_affine``：当前 step-mode 对齐 STA 后端会明确拒绝。

本教程的模型级结果使用 ``spiking_encoder``。

Step-mode 执行
--------------

``STATransformerRecipe`` 的转换产物是普通 ``nn.Module`` / ``fx.GraphModule``。用户通过 ``functional.set_step_mode`` 递归设置其内部 step-mode 模块。单步模式下，用户显式编写时间循环，并在处理独立序列前重置状态：

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

如果要触发 layer-wise 多步加速，将转换产物切换为 ``step_mode="m"``，并输入第 0 维为时间维的序列 tensor：

.. code-block:: python

    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)

    x_zeros = torch.zeros_like(x).expand(converted.time_steps - 1, *x.shape)
    x_seq = torch.cat((x.unsqueeze(0), x_zeros), dim=0)
    y_seq = converted(x_seq)
    y = y_seq.sum(dim=0)

多输入模型同样由用户显式构造浮点输入序列。``attn_mask`` 等具名静态控制 tensor 应保持原样，不额外添加时间维。最终累计读出由用户对输出时间维求和或按输出结构自行递归求和。

多步后端比任意 PyTorch 图更严格。它当前会拒绝 ``mode="spiking_affine"``、``spike_linear=True``、``spike_conv2d=True``、请求或使用 attention weights 的 ``MultiheadAttention`` 调用、``key_padding_mask``、functional ``scaled_dot_product_attention``，以及不支持的 FX tensor 操作。如果转换器报出这些限制，请将模型改写为受支持的 sequence-preserving 模块和操作。

与 TransformerSpikeEquivalentRecipe 的关系
--------------------------------------------

``TransformerSpikeEquivalentRecipe`` 是一个不需要 dataloader 的替换路径，用于将当前支持的 Transformer 算子替换为 TD / spike-equivalent 模块。它适合作为算子级转换基线，但不进行 STA 校准，也不提供 STA 输入编码和累计读出 helper。

``STATransformerRecipe`` 是模型级 STA 流程。启用 spike encoder 时它需要校准，并返回一个 step-mode 模块；该模块既可以一次执行一个时间步，也可以接收完整时间序列。

ViT-B/16 ImageNet 示例
----------------------

完整可运行示例是 ``spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta``。该脚本使用 ``torchvision.models.vit_b_16``、``ViT_B_16_Weights.DEFAULT``，以及可被 ``torchvision.datasets.ImageFolder`` 读取的 ImageNet 验证集目录。

下面命令假设 ``/path/to/imagenet/val`` 直接包含各类别文件夹，需要 CUDA。

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta \
      --data-root /path/to/imagenet/val \
      --device cuda:0 \
      --batch-size 16 \
      --num-workers 8 \
      --calib-samples 2048 \
      --time-steps 8 \
      --threshold-scale 0.5

若只想快速检查环境，可使用少量验证样本：

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

下表数据在 NVIDIA A100-SXM4-80GB 上，使用完整 50000 张 ImageNet 验证集测得：

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

Top-1 下降 0.368 个百分点。本次使用 ``batch_size=16`` 的完整运行中 ANN baseline 推理耗时约 181.6 秒， STA 转换模型约 1834.0 秒；wall-clock time 对运行时环境比较敏感。如需比较 single-step 和 multi-step 执行耗时，请使用专门的 step-mode benchmark。

关键 stdout 行如下：

.. code-block:: shell

    BASELINE {"top1": 0.81068, "top5": 0.95318, "total": 50000, "seconds": 181.58131194114685}
    STA_SPIKING_ENCODER_T8_S0p5 {"top1": 0.807, "top5": 0.95202, "total": 50000, "seconds": 1833.9626359939575}
    DROP 0.0036799999999999056

SpikeZIP-TF-style BERT SST-2 示例
---------------------------------

``SpikeZIPTFRecipe`` 是一个窄版 SpikeZIP-TF-style 累计差分 recipe，面向语言 Transformer 分类模型。第一条公开支持路径是 BERT-style SST-2 分类，并采用 embedding 输出之后的转换边界。原 Hugging Face BERT embedding 层仍然接收整数 ``input_ids``。ANN2SNN 转换图从浮点 ``embedding_output`` 和 ``extended_attention_mask`` 开始，转换 encoder、pooler、dropout 和 classifier wrapper。

这个边界把 tokenization、整数 token id、embedding lookup 和 Hugging Face mask 构造保留在 ANN2SNN 图外，也避免把该示例误读为完整 autoregressive LLM conversion：``SpikeZIPTFRecipe`` 不支持 decoder generation、KV cache、causal language-model perplexity 评测，也不实现完整 SpikeZIP-TF 论文中的 ST-BIF+ 与 activation quantization。它复用 SpikingJelly 现有 TD 算子来覆盖 Linear、LayerNorm、GELU、Softmax 和 attention matrix multiplication。

完整可运行示例是 ``spikingjelly.activation_based.ann2snn.examples.bert_sst2_spikezip_tf``。Hugging Face 依赖是可选依赖，只在运行该示例时安装：

.. code-block:: shell

    uv pip install transformers datasets

建议先运行一个小验证切片：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.bert_sst2_spikezip_tf \
      --model-name-or-path textattack/bert-base-uncased-SST-2 \
      --dataset-name nyu-mll/glue \
      --dataset-config sst2 \
      --split validation \
      --device cuda:0 \
      --batch-size 32 \
      --eval-samples 256 \
      --time-steps 8 \
      --output benchmark/output/bert_sst2_spikezip_tf_t8_small.json

完整 SST-2 验证时去掉 ``--eval-samples``：

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.bert_sst2_spikezip_tf \
      --model-name-or-path textattack/bert-base-uncased-SST-2 \
      --dataset-name nyu-mll/glue \
      --dataset-config sst2 \
      --split validation \
      --device cuda:0 \
      --batch-size 32 \
      --time-steps 8 \
      --output benchmark/output/bert_sst2_spikezip_tf_t8_full.json

转换产物遵循相同的显式 step-mode 契约：

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

``extended_attention_mask`` 是静态控制 tensor，保持原样传入，不添加时间维。

下表数据在 NVIDIA A100-SXM4-80GB 上，使用 SST-2 validation split 的 872 条样本测得：

.. list-table:: BERT SST-2 SpikeZIP-TF-style 转换结果
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
    * - ``SpikeZIPTFRecipe``
      - 872
      - 8
      - 92.431

Accuracy 下降 0.000 个百分点。评测前，示例会先在前两个 batch 上检查 FX-friendly wrapper 与原 Hugging Face classifier 的 logits 是否对齐。本次使用 ``batch_size=32`` 的完整运行中 ANN wrapper 推理耗时约 1.60 秒，转换模型约 9.20 秒。关键 stdout 行如下：

.. code-block:: shell

    HF_WRAPPER_PARITY {"checked_batches": 2, "max_abs_diff": 5.245208740234375e-06, "atol": 1e-05}
    BASELINE {"accuracy": 0.9243119266055045, "total": 872, "seconds": 1.5974977016448975}
    SPIKEZIP_TF {"accuracy": 0.9243119266055045, "total": 872, "seconds": 9.204399347305298}
    DROP 0.0

.. [#sta] Y. Jiang, K. Hu, T. Zhang, H. Gao, Y. Liu, Y. Fang, and F. Chen, "Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers," ICLR 2024. https://openreview.net/forum?id=XrunSYwoLr
