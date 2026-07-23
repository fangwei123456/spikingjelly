Transformer ANN2SNN Conversion
==============================

Author: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版：:doc:`../cn/ann2snn_transformer`

This page introduces Transformer-oriented ANN2SNN paths in ``spikingjelly.activation_based.ann2snn``, including the ``TransformerTDEquivalentRecipe`` baseline, ``STATransformerRecipe`` based on Spatio-Temporal Approximation (STA) [#sta]_, ``SpikeZIPTFQANNRecipe`` for SpikeZIP-compatible QANNs [#spikezip]_, and calibration-driven ``Qwen2SNNRecipe`` conversion. For classical ReLU-to-IFNode rate-coding conversion on CNNs, see :doc:`ann2snn`.

This page orders four Transformer ANN2SNN paths from the simplest to the most specialized:

* **Path 1: TransformerTDEquivalentRecipe baseline**. This is the most direct TD-equivalent operator replacement path. The BERT SST-2 example shows the conversion boundary for language classification models after the embedding output.
* **Path 2: Spatio-Temporal Approximation (STA) Transformer conversion**. This is the model-level enhanced path implemented by ``STATransformerRecipe``. It keeps the cumulative-difference and explicit step-mode readout idea, then adds dataloader calibration and spike encoders. This page reports a full ViT-B/16 ImageNet result.
* **Path 3: SpikeZIP QANN-to-SNN conversion**. This is the ``SpikeZIPTFQANNRecipe`` module-tree path. The input must already be a SpikeZIP-compatible QANN. This page gives a synthetic RoBERTa parity check and a ViT-Small ImageNet reproducibility entry point.
* **Path 4: Qwen2 offline multi-step conversion**. ``Qwen2SNNRecipe`` calibrates a Hugging Face Qwen2 causal LM and converts every decoder block with current SpikingJelly TD operators and signed activation-aware IF neurons.

.. warning::

    STA conversion is not a strict fully spike-driven SNN conversion. ``mode="spiking_encoder"`` emits a quantized value equal to an integer spike count, possibly negative for negative residuals, multiplied by a calibrated threshold, not a binary spike tensor.

    ``STATransformerRecipe`` should therefore be read as a training-free Transformer ANN2SNN approximation, not as a promise of fully spike-driven LLM conversion. Language models that take integer tokens need a separate input and embedding contract. Converted STA models are stateful and follow SpikingJelly ``step_mode`` semantics for single-step and multi-step execution.

.. warning::

    The ST-BIF neuron used by the SpikeZIP path supports signed ternary outputs and is also not a strict binary-spike neuron. This distinction matters when comparing conversion methods under a strict SNN definition.

Path 1: Differential-equivalence Baseline
-----------------------------------------

``TransformerTDEquivalentRecipe`` is a dataloader-free replacement path that swaps supported Transformer operators for TD-equivalent modules. It is the smallest Transformer conversion baseline on this page: it performs no STA calibration, owns no internal timestep loop, and relies on existing TD operators to express cumulative differences. It does not emit spikes or insert spiking neurons, and converted modules return floating-point difference values. Therefore, its output is not an SNN.

The later ``STATransformerRecipe`` can be understood as a model-level enhancement of the same cumulative-difference and explicit step-mode idea. It adds dataloader calibration and spike encoders, and targets a complete Transformer FX graph. The two recipes are not the same API and neither is a strict superset of the other; their support boundaries and result interpretations differ.

TD-equivalent Online Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformer models contain affine projections, LayerNorm or RMSNorm, GELU or SiLU, attention, residual additions, masks, and tensor constants. A ReLU-to-IFNode rate-coding rule cannot cover these components directly, so Transformer conversion needs a different route. A common method is temporal differencing.

For an ordinary ANN input :math:`x`, embed it as the following differential input sequence:

.. math::

    x^{(0)} = x,\qquad x^{(t)} = 0,\quad t=1,\ldots,T-1.

The cumulative input after timestep :math:`t` is:

.. math::

    X^{(t)} = \sum_{\tau=0}^{t} x^{(\tau)}.

So :math:`X^{(0)} = X^{(1)} = \cdots = x`. Let :math:`f` denote one function or block in the original ANN, such as an affine projection, LayerNorm, GELU, or attention block. The converted TD-equivalent block is not :math:`f` itself but the difference of :math:`f` on adjacent cumulative inputs. Denote this difference block by :math:`F_t`:

.. math::

    F_t\left(X^{(t)}\right)
    =
    f\left(X^{(t)}\right) - f\left(X^{(t-1)}\right),
    \qquad f\left(X^{(-1)}\right) = 0.

In the implementation, :math:`F_t` is realized by a stateful wrapper around the original operation. It evaluates or reuses the cumulative output of :math:`f`, caches the previous cumulative output, and returns only the current difference :math:`\Delta y^{(t)} = F_t(X^{(t)})`.

The accumulated output satisfies:

.. math::

    \sum_{t=0}^{T-1} F_t\left(X^{(t)}\right)
    = f\left(X^{(T-1)}\right) - f\left(X^{(-1)}\right)
    = f(x).

This identity explains why the TD-equivalent path works: if every converted block emits only the difference of the cumulative output and constants/bias terms are counted once, summing the timestep outputs recovers the ANN block output. The online-equivalent part of STA reuses the same idea.

The online-equivalent path follows explicit SpikingJelly step-mode semantics:

* ``model(x)`` on the original ANN is one complete ANN inference call, without temporal expansion;
* with ``step_mode="s"``, one converted-model call executes one differential timestep, and users write the outer time loop explicitly. Converted modules receive the current differential input :math:`x^{(t)}` and keep cumulative input :math:`X^{(t-1)}` and cumulative output :math:`f(X^{(t-1)})` internally;
* with ``step_mode="m"``, the converted model receives a complete differential sequence tensor whose first dimension is time, and returns the complete output sequence. This mode can usually parallelize or vectorize work along the time dimension, and is much faster than single-step mode.

For ordinary ANN inference on one sample :math:`x`, the common differential input sequence is :math:`x^{(0)}=x` followed by zero timesteps, as in the example above. If the input is already temporal, pass its corresponding differential sequence directly.

BERT SST-2 Example
^^^^^^^^^^^^^^^^^^

``TransformerTDEquivalentRecipe`` converts common Transformer core operators into TD-equivalent stateful modules, so the converted model can run single-step or multi-step inference with the SpikingJelly ``step_mode`` contract. In addition to the BERT operators below, the recipe converts ``nn.RMSNorm`` to ``TDRMSNorm`` and both ``nn.SiLU`` and ``torch.nn.functional.silu`` to ``TDSiLU``. Their outputs are floating-point temporal differences, not binary spikes. This example uses BERT-style SST-2 classification and places the conversion boundary after the embedding output: the original Hugging Face BERT embedding layer still consumes integer ``input_ids``, while the ANN2SNN graph starts from floating-point ``embedding_output`` and ``extended_attention_mask`` and covers the encoder, pooler, dropout, and classifier wrapper. The runnable example is ``spikingjelly.activation_based.ann2snn.examples.bert_sst2_transformer_td_equivalent``.

Hugging Face packages are optional; install them only when running this example:

.. code-block:: shell

    uv pip install transformers datasets

Run a small validation slice first:

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

For a full SST-2 validation run, omit ``--eval-samples``:

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

The converted model follows the same explicit step-mode contract:

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

``extended_attention_mask`` is a static control tensor and is passed unchanged, without a time dimension.

The full SST-2 validation run below was measured on an NVIDIA A100-SXM4-80GB with the 872-sample validation split:

.. list-table:: BERT SST-2 Transformer TD-equivalent conversion results
    :header-rows: 1
    :widths: 36 18 18 18

    * - Method
      - Validation samples
      - Timesteps
      - Accuracy (%)
    * - ANN
      - 872
      - -
      - 92.431
    * - ``TransformerTDEquivalentRecipe``
      - 872
      - 8
      - 92.431

The accuracy drop is 0.000 percentage points. Before evaluation, the example checks the logit alignment between the FX-friendly wrapper and the original Hugging Face classifier on the first two batches. In this full run with ``batch_size=32``, the ANN wrapper took about 1.60 seconds and the converted model about 9.20 seconds. The key stdout lines are:

.. code-block:: shell

    HF_WRAPPER_PARITY {"checked_batches": 2, "max_abs_diff": 5.245208740234375e-06, "atol": 1e-05}
    BASELINE {"accuracy": 0.9243119266055045, "total": 872, "seconds": 1.5974977016448975}
    TRANSFORMER_TD_EQUIVALENT {"accuracy": 0.9243119266055045, "total": 872, "seconds": 9.204399347305298}
    DROP 0.0

Path 2: STA Conversion
----------------------

Taking one step beyond the Path 1 differential-equivalence baseline, STA extends the differential idea. It still relies on explicit step-mode execution and final summation over the time dimension, but it adds calibrated spike encoders after selected nonlinear and attention outputs. The online differential-equivalence part of STA is the same spike-free TD-equivalent cumulative-difference idea introduced in Path 1; this section only adds the STA-specific spike encoder, calibration, and model-level experiment.

STA Spike Encoder
^^^^^^^^^^^^^^^^^

``mode="spiking_encoder"`` adds a spike encoder after selected increments. For an analog increment :math:`a^{(t)}` and threshold :math:`V`, the encoder keeps a residual membrane :math:`r^{(t)}`. The initial residual is zero:

.. math::

    r^{(-1)} = 0.

At each timestep, the encoder first integrates the analog increment:

.. math::

    u^{(t)} = r^{(t-1)} + a^{(t)}.

It then computes how many threshold-sized units can be emitted:

.. math::

    n^{(t)} = \operatorname{trunc}\left(\frac{u^{(t)}}{V}\right),
    \qquad
    s^{(t)} = n^{(t)} V.

Here :math:`s^{(t)}` is the quantized output of this timestep. The residual for the next timestep is:

.. math::

    r^{(t)} = u^{(t)} - s^{(t)}.

In SNN terms, :math:`r^{(t)}` is the membrane voltage retained after firing, :math:`n^{(t)}` is an integer spike count that may be negative when the residual is negative, and :math:`s^{(t)}` is the threshold-weighted spike output. The update :math:`r^{(t)} = u^{(t)} - s^{(t)}` is a generalized soft reset: it subtracts the emitted threshold-weighted value from the membrane. When :math:`n^{(t)} = 1` it reduces to the usual soft reset; larger positive or negative integers represent multiple threshold-unit crossings in one timestep.

After :math:`T` timesteps:

.. math::

    \sum_{t=0}^{T-1} s^{(t)}
    =
    \sum_{t=0}^{T-1} a^{(t)}
    - r^{(T-1)}

So the encoder output equals the analog increment total minus the final residual. If :math:`a^{(t)}` is the STA difference :math:`F_t(X^{(t)})`, the analog sum is the ANN block output :math:`f(x)`, and the spike-encoded result differs from it only by the final residual. STA calibrates thresholds using ``time_steps``, so when the calibrated activation range is fixed, larger :math:`T` gives finer temporal quantization.

With online cumulative differences, the converted model can evaluate these operators on cumulative inputs and then encode output increments, preserving operator semantics while introducing spike-like temporal communication at selected output locations. Affine modules, ``LayerNorm``, ``GELU``, ``MultiheadAttention``, and floating FX tensor constants each maintain their own online cumulative-difference state. Bias terms and graph constants are injected once. Static attention masks and similar control tensors persist across timesteps rather than being zeroed.

The recommended configuration in this tutorial is ``STATransformerRecipe(mode="spiking_encoder")``. It adds calibrated stateful spike encoders after ``LayerNorm``, ``GELU``, and ``MultiheadAttention`` outputs, and keeps the main affine projections in the online-equivalent regime. Thresholds are calibrated from a dataloader and depend on ``time_steps``.

Using the Recipe
^^^^^^^^^^^^^^^^

``STATransformerRecipe`` is an FX graph recipe. The minimum Python API uses ``Converter`` (the compatibility name for ``FXConverter``) to execute the conversion:

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

``time_steps`` belongs to the recipe because it drives threshold calibration and the conversion-time expansion of any graph constant whose sequence length cannot be inferred from runtime inputs.

There are three STA modes:

* ``equivalent``: cumulative-difference baseline without calibration;
* ``spiking_encoder``: calibrated spike encoders on nonlinear and attention outputs;
* ``spiking_affine``: currently rejected by the step-mode-aligned STA backend.

This tutorial uses ``spiking_encoder`` for the model-level result.

Step-mode Execution
^^^^^^^^^^^^^^^^^^^

``STATransformerRecipe`` is an FX graph recipe. It produces an ``fx.GraphModule``, which is an ``nn.Module`` subclass. Users call ``functional.set_step_mode`` to recursively configure its internal step-mode modules. The single-step example below shows the ``[x, 0, 0, ...]`` differential sequence for an ordinary ANN input ``x``; each ``converted(x_t)`` call is one STA timestep, and the final ``y`` is the ANN-level readout. If the input is already temporal, pass the corresponding differential timestep in the same way. Reset state before an independent sequence:

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

For layer-wise multi-step acceleration, switch the converted model to ``step_mode="m"`` and pass sequence tensors whose first dimension is time:

.. code-block:: python

    from spikingjelly.activation_based import functional

    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)

    x_zeros = torch.zeros_like(x).expand(converted.time_steps - 1, *x.shape)
    x_seq = torch.cat((x.unsqueeze(0), x_zeros), dim=0)
    y_seq = converted(x_seq)
    y = y_seq.sum(dim=0)

For multi-input models, users construct floating-point input sequences explicitly. Named static control tensors such as ``attn_mask`` are passed unchanged, without adding a time dimension. Users perform the final accumulated readout by summing the output time dimension, recursively when outputs are structured.

The multi-step backend is stricter than arbitrary PyTorch graphs. It currently rejects ``mode="spiking_affine"``, ``spike_linear=True``, ``spike_conv2d=True``, ``MultiheadAttention`` calls that request or use attention weights, ``key_padding_mask``, functional ``scaled_dot_product_attention``, and unsupported FX tensor operations. When the converter reports one of these limits, rewrite the model around supported sequence-preserving modules and operations.

ViT-B/16 ImageNet Example
^^^^^^^^^^^^^^^^^^^^^^^^^

The runnable example is ``spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta``. It loads ``torchvision.models.vit_b_16`` with ``ViT_B_16_Weights.DEFAULT`` and an ImageNet validation directory that ``torchvision.datasets.ImageFolder`` can read directly.

The command below assumes that ``/path/to/imagenet/val`` contains the class folders, and that CUDA is available:

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta \
      --data-root /path/to/imagenet/val \
      --device cuda:0 \
      --batch-size 16 \
      --num-workers 8 \
      --calib-samples 2048 \
      --time-steps 8 \
      --threshold-scale 0.5

To verify the environment end to end, run a small validation subset:

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

The full ImageNet validation run below was measured on an NVIDIA A100-SXM4-80GB with the full 50,000-image validation set:

.. list-table:: ViT-B/16 ImageNet STA conversion results
    :header-rows: 1
    :widths: 30 16 16 14 18 18

    * - Method
      - Calibration samples
      - Validation samples
      - Timesteps
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

The Top-1 drop is 0.368 percentage points. In this full run with ``batch_size=16``, the ANN baseline took about 181.6 seconds and the converted STA model about 1834.0 seconds. Wall-clock time is sensitive to runtime conditions; for single-step versus multi-step timing, use the dedicated step-mode benchmarks.

The key stdout lines are:

.. code-block:: shell

    BASELINE {"top1": 0.81068, "top5": 0.95318, "total": 50000, "seconds": 181.58131194114685}
    STA_SPIKING_ENCODER_T8_S0p5 {"top1": 0.807, "top5": 0.95202, "total": 50000, "seconds": 1833.9626359939575}
    DROP 0.0036799999999999056

Path 3: SpikeZIP Conversion
---------------------------

Conversion Contract And API
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SpikeZIPTFQANNRecipe`` converts a quantized ANN (QANN) into an SNN. The input model must already be a SpikeZIP-compatible QANN. The current implementation supports two attention module contracts:

* RoBERTa-style self-attention must expose ``query``, ``key``, and ``value`` linear layers; ``num_attention_heads``, ``attention_head_size``, ``all_head_size``, and ``dropout``; plus ``query_quan``, ``key_quan``, ``value_quan``, ``attn_quan``, and ``after_attn_quan`` quantizers with ``s``, ``sym``, ``pos_max``, ``neg_min``, and ``level`` attributes.
* ViT-style self-attention must expose ``qkv`` and ``proj`` linear layers; ``quan_q``, ``quan_k``, ``quan_v``, ``attn_quan``, ``after_attn_quan``, and ``quan_proj`` quantizers with the same quantization attributes; plus ``num_heads``, ``head_dim``, ``scale``, ``attn_drop``, and ``proj_drop``.

The recipe replaces the QANN-side quantizers and Transformer operators with transparent SNN-side ST-BIF, SESA attention multiplication, Spike-Softmax, Spike-LayerNorm, embedding, and linear modules. This algorithm directly modifies the ``nn.Module`` tree and does not run FX tracing, so use ``ModuleConverter``:

.. code-block:: python

    from spikingjelly.activation_based import ann2snn

    recipe = ann2snn.SpikeZIPTFQANNRecipe(
        time_steps=32,
        model_family="roberta",
    )
    converted = ann2snn.ModuleConverter(recipe=recipe, device="cuda:0").convert(qann)
    converted.eval()

Users control single-step or multi-step execution with ``functional.set_step_mode`` and explicitly pass either a single timestep or a sequence whose first dimension is time.

The ST-BIF neuron used by this SpikeZIP path is inference-only. The torch implementation contains ``detach``, ``round``, and discrete state updates, and the Triton kernel does not implement backward. Use it to run converted QANN-to-SNN models, not for end-to-end gradient training or fine-tuning.

Synthetic RoBERTa Parity Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section uses RoBERTa as a simple example to verify that a SpikeZIP-compatible QANN can be converted to an SNN model whose user-accumulated logits match the QANN logits:

.. code-block:: shell

    python -m spikingjelly.activation_based.ann2snn.examples.roberta_spikezip_qann_synthetic \
      --device cpu \
      --time-steps 32 \
      --batch-size 3 \
      --seq-len 5 \
      --output benchmark/output/roberta_spikezip_qann_synthetic_cpu.json

The example also accepts ``--qann-checkpoint`` to load a plain ``state_dict`` saved from the same tiny QANN architecture. For real SpikeZIP checkpoints, first ensure that the QANN model's RoBERTa-style attention modules expose quantizers such as ``query_quan`` with ``s``, ``sym``, ``pos_max``, ``neg_min``, and ``level`` attributes, then pass the model to ``ModuleConverter(recipe=SpikeZIPTFQANNRecipe(...))``.

The expected stdout contains a near-zero parity error and an ST-BIF state summary:

.. code-block:: shell

    {"accumulated_sequence_shape": [32, 3, 2], "max_abs_diff": 3.2782554626464844e-07, "mean_abs_diff": 8.630255621255856e-08, "recipe": "SpikeZIPTFQANNRecipe", "stbif_state": {"last_step_spike_values": [0.0], "max_accumulated": 0.75, "min_accumulated": -1.0}}

The result below was measured on CPU with the command above. It shows that the SpikeZIP path can convert a RoBERTa-style language Transformer when the input is already a SpikeZIP-compatible QANN.

.. list-table:: SpikeZIP synthetic RoBERTa QANN-to-SNN parity
    :header-rows: 1
    :widths: 24 14 14 18 18

    * - Model
      - Timesteps
      - Batch / seq-len
      - Max abs diff
      - Mean abs diff
    * - Synthetic RoBERTa QANN
      - 32
      - 3 / 5
      - 3.278e-07
      - 8.630e-08

SpikeZIP ViT-Small ImageNet Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a real SpikeZIP-compatible ViT-Small QANN checkpoint, use ``benchmark/benchmark_ann2snn_spikezip_vit_qann.py`` to run ImageNet validation. The script records QANN baseline accuracy, converted SNN accuracy, prediction agreement, logit differences, timing, CUDA device metadata, and peak memory.

The ``vit-small-imagenet-relu-q32-81.59.pth`` checkpoint used in this tutorial comes from the official SpikeZIP-TF repository: `ViT-Small-ReLU-Q32 pretrained QANN <https://github.com/Intelligent-Computing-Research-Group/SpikeZIP-TF>`_. The corresponding Hugging Face checkpoint is `XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32 <https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32>`_. The official README lists the md5 prefix as ``8207d3e``. After downloading it, replace ``--checkpoint`` below with the local checkpoint path, and replace ``--imagenet-root`` with your local ImageNet root:

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

The full validation result below was measured on ``g2`` with an NVIDIA A100-SXM4-80GB, using all 50,000 ImageNet validation samples:

.. list-table:: SpikeZIP ViT-Small ImageNet QANN-to-SNN result
    :header-rows: 1
    :widths: 24 14 14 16 16 18

    * - Model
      - Timesteps
      - Samples
      - Top-1 (%)
      - Top-5 (%)
      - Notes
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
      - ``step_mode="m"``, ``stbif_backend="triton"``

The Top-1 difference is ``+0.090`` percentage points, and the Top-1 prediction agreement is ``97.34%``. QANN inference took ``79.20`` seconds; SNN inference took ``3719.15`` seconds (``61.99`` minutes), using ``snn_batch_size=4`` and peaking at ``25.63`` GiB of allocated CUDA memory. In the benchmark output, ``parity_pass=false`` because that field checks logits with ``torch.allclose(snn_logits, qann_logits, atol=parity_atol, rtol=1e-5)``; this run used ``parity_atol=1e-4``.

Path 4: Qwen2 Offline Multi-step Conversion
-------------------------------------------

``Qwen2SNNRecipe`` is a ``ModuleConverter`` recipe for evaluation-mode Hugging
Face Qwen2 causal LMs. Install the pinned optional dependency with
``uv pip install -e '.[qwen]'``. Calibration produces a reusable
``Qwen2SNNCalibration`` artifact; conversion replaces the complete decoder
stack with TD RMSNorm, TD linear, TD SiLU/product, TD SDPA, and
``SignedQCFSSequenceEncoder`` modules backed by current SpikingJelly
``ActivationAwareIFNode`` implementations.

The converted SNN uses the explicit temporal layout ``[T,B,S,H]`` and executes
``layerwise_offline_multistep``. Consequently the schedule is closer to
:math:`L T` work than online T-step inference. It does not offer a single-step
Qwen execution claim. Call ``functional.reset_net(model)`` before independent
prompts; KV-cache continuation remains explicit through ``past_key_values``.

The public Qwen2 path deliberately exposes only the real ``signed_if`` temporal
execution (plus ``exact_td`` for equivalence checks). It does not replace the
spike sequence with a count-domain shortcut during quality evaluation.

What conversion produces
^^^^^^^^^^^^^^^^^^^^^^^^

Conversion does not change the Qwen2 token embedding, decoder depth, hidden
size, GQA head layout, LM-head weights and dimensions, or existing weight
tying, and it does not train new weights. It freezes all parameters and replaces
RMSNorm, linear, SiLU/SwiGLU product, and SDPA operations in every decoder block
with the TD implementations introduced earlier. The Qwen2 recipe adds signed
QCFS encoding boundaries between those TD operators:

The LM-head module is also represented by a ``TDLinear`` copy, but the final
forward uses its preserved weight after summing over T to compute logits. The
weight values, dimensions, and embedding-tying relationship remain unchanged.

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

A converted model with :math:`L` decoder blocks contains :math:`4L+1` signed
encoders: one after the embedding and four per block for Q, K, V, and the MLP
intermediate activation. Each signed encoder owns a positive and a negative
``ActivationAwareIFNode``, giving :math:`8L+2` IFNode module instances in
total. Each instance operates elementwise over its input tensor and represents
a population that shares per-channel thresholds, not one scalar neuron.

This is a hybrid explicit multi-step model. Binary firing occurs inside the
signed encoders. TD RMSNorm, linear, SiLU/product, and attention propagate
floating-point temporal differences, whose elements may be negative and are not
all ``0/1`` spikes. Here, “Qwen SNN” specifically means that selected activation
boundaries use real IF dynamics and spike-count representations while TD
operators preserve Transformer semantics; it does not mean that every Qwen
operation has become a binary event.

The converted model is read-only and inference-only: call ``eval()`` and run it
inside ``torch.no_grad()`` or ``torch.inference_mode()``. Its ``forward()``
raises an error in training mode or while autograd is enabled.

Signed QCFS encoders and IF neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SignedQCFSSequenceEncoder`` is the encoding rule and its execution container;
its pair of ``ActivationAwareIFNode`` modules actually runs the membrane
dynamics. Given a static signed activation :math:`x`, the encoder first splits
it into

.. math::

    x^+ = \operatorname{ReLU}(x), \qquad
    x^- = \operatorname{ReLU}(-x).

The positive and negative IF neurons receive length-:math:`T` constant-current
sequences:

.. math::

    I_t^+ = \frac{x^+}{T}, \qquad
    I_t^- = \frac{x^-}{T}, \qquad 0 \leq t < T.

Thus the caller passes a static activation to ``encode(x)`` once, but each IF
neuron sees T copies of :math:`x/T`, not :math:`x` at step 0 followed by zeros.
Each IF uses a per-channel threshold :math:`s`, offset :math:`s/2`, non-leaky
integration, and soft reset:

.. math::

    H_t = V_{t-1} + I_t,

.. math::

    S_t = \Theta(H_t + s/2 - s),

.. math::

    V_t = H_t - S_t s.

The :math:`s/2` offset makes the total spike count implement rounding. Soft
reset subtracts the threshold instead of zeroing the membrane, thereby
preserving the residual after that subtraction. The two binary IF outputs are
scaled by :math:`s` and subtracted:

.. math::

    Y_t = (S_t^+ - S_t^-)s.

The encoder therefore returns an explicit ``[T,...]`` sequence whose elements
belong to :math:`\{-s,0,s\}` and whose temporal sum satisfies

.. math::

    \sum_{t=0}^{T-1}Y_t
    =
    \operatorname{clamp}\left(
        \operatorname{round}\left(\frac{x}{s}\right), -T, T
    \right)s.

The implementation also corrects rare count mismatches caused by BF16
accumulation and round-to-even boundaries, so the observed temporal sum follows
this expression. That correction does not change the normal use of
:math:`x/T` constant current.

Who consumes the T dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A QCFS encoder only promises an output shaped ``[T,...]``. The downstream
module decides whether and when to sum over T. The converted Qwen2 model has
three consumption patterns:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Downstream position
     - How it consumes the input
     - Output
   * - TD operators and residuals
     - Consume all T steps; a TD operator may use cumulative states internally
     - Still ``[T,...]``
   * - Next signed QCFS boundary
     - Recover the aggregate with ``sequence.sum(0)``, then re-encode it
     - A new ``[T,...]`` spike sequence
   * - KV cache or final LM head
     - Sum over T before caching or computing logits
     - A floating-point tensor without T

Do not confuse the ``cumsum`` used internally by a TD operator with
``sum(dim=0)`` at a QCFS boundary. The former still returns T temporal
differences. The latter actually collapses the temporal dimension and starts a
new signed spike-count encoding.

Decoder-block boundaries do not automatically collapse T either. The TD down
projection and residual at the end of one block produce a full
``[T,B,S,H]`` sequence, which enters the next block unchanged. Q, K, and V are
summed and re-encoded at their encoding boundaries inside that next block.
Similarly, attention output retains T until the MLP-intermediate signed encoder
sums and re-encodes it. The model is therefore not an online pipeline in which
one spike travels from the embedding through every layer. It repeatedly
propagates T steps through a group of TD operators, aggregates them, and
re-encodes the result.

``exact_td`` versus ``signed_if``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both execution modes use the same TD decoder, but they take different actions
at each encoding boundary:

.. list-table::
   :header-rows: 1
   :widths: 20 34 18 28

   * - Mode
     - Sequence produced at the boundary
     - Runs IF neurons
     - Numerical meaning
   * - ``exact_td``
     - ``[x, 0, ..., 0]``
     - No
     - Preserves floating-point :math:`x` for TD equivalence checks
   * - ``signed_if``
     - Signed spikes after the positive and negative IF neurons receive ``ReLU(x)/T`` and ``ReLU(-x)/T``
     - Yes
     - Reconstructs a quantized and clipped :math:`x`

The ``exact_td`` branch does not call ``SignedQCFSSequenceEncoder`` or its IF
neurons. Those modules still exist on the model object, but that forward pass
bypasses them. Consequently only floating-point and operator-implementation
differences should remain between ``exact_td`` and dense Qwen, while
``signed_if`` additionally incurs rounding and saturation error at every
encoding boundary.

The public recipe currently has no third mode that sends ``[x,0,...,0]`` into
an IF neuron and continues running it for T steps. For the current non-leaky,
soft-reset IF with at most one spike per step, that front-loaded input can
produce the same total spike count as constant :math:`x/T` input in ideal
arithmetic. Its spike times differ, and it needs the same round-to-even boundary
handling. It remains a quantized, T-limited IF path and is not equivalent to
``exact_td``, which bypasses neurons.

T, calibration levels, and quantization error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen2 calibration estimates an absolute activation bound :math:`R` for each
channel. When ``calibration_quantile < 1``, :math:`R` is that quantile;
otherwise it is the maximum. The per-channel QCFS step is

.. math::

    s = \frac{R}{\text{calibration\_levels}},

where this simplified expression assumes :math:`R \geq 10^{-6}`; the
implementation applies ``clamp_min(1e-6)`` to prevent a zero scale. T controls
how many spikes a binary IF can emit in one encoding window. Therefore

.. math::

    \text{rounding error} \leq \frac{s}{2}
    \quad \text{when } |x| \leq Ts, \qquad
    |x|_{\max} = Ts.

The configuration requires
``0 < calibration_levels <= time_steps``. This makes :math:`Ts` cover at least
the calibrated bound :math:`R`; if T were smaller than levels, even :math:`R`
itself would be clipped.

With fixed ``calibration_levels``, increasing T does not reduce the spacing
:math:`s` between adjacent representable values. It expands the dynamic range
and reduces the chance of clipping an activation to :math:`\pm Ts`. Once
clipping occurs, total error can be much larger than :math:`s/2`. For example,
this tutorial uses ``T=160``, ``levels=16``, and ``quantile=0.999``:

.. math::

    s = R_{0.999}/16, \qquad Ts = 10R_{0.999}.

Compared with ``T=16``, ``T=160`` has the same nominal rounding resolution but
expands the representable range from :math:`\pm R_{0.999}` to approximately
:math:`\pm 10R_{0.999}`. This design provides headroom for tail activations
outside the calibration sample and distribution shifts in Q/K/V, SwiGLU, and
repeated re-encoding. It is intended to trade more multi-step computation and
memory for less saturation error. Increasing ``calibration_levels`` alone
reduces both :math:`s` and :math:`Ts`; retaining the current dynamic range while
refining the spacing also requires increasing T proportionally, or using
another scale/range design.

This differs from conversion paths earlier in the tutorial that tie their
threshold calibration directly to ``time_steps``. The Qwen2 recipe separates
``time_steps`` from ``calibration_levels``. Read T as count capacity and levels
as resolution within the calibrated range. The final configurations are
empirical quality/cost operating points, not evidence that their T values are
theoretically optimal.

Minimal calibration and inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example loads a local Qwen2.5 checkpoint, calibrates per-channel
scales, converts the complete decoder, and runs the real multi-step path. The
small prompt bank demonstrates the API; reproducing the quality results below
requires the pinned WikiText-2 calibration protocol in
``benchmark/snn_llm/qwen_conversion``.

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

The calibration state contains tensors and basic Python values and can be
restored with ``Qwen2SNNCalibration.from_state_dict(torch.load(...,
weights_only=True))``. Calibration metadata must exactly match the conversion
configuration. Use the Torch neuron backend for a reference implementation;
the Triton backend is CUDA, multi-step, inference-only.

Reference quality evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^

The pinned BF16 evaluations use the full WikiText-2 test split (584 windows,
298,937 target tokens) and six zero-shot tasks: LAMBADA, PIQA, HellaSwag,
WinoGrande, ARC-Easy, and ARC-Challenge. ``Mean drop`` and ``max drop`` are
dense-minus-SNN accuracy percentage points across those tasks.

.. list-table:: Qwen2.5 Base offline multi-step quality
   :header-rows: 1

   * - Model
     - T / levels / quantile
     - Dense PPL
     - SNN PPL
     - PPL degradation
     - Mean / max drop (pp)
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

These are quality results for an offline layerwise schedule, not latency or
energy claims. High-T execution remains substantially slower than dense Qwen
on the evaluated A100 GPU. The 3B checkpoint is governed by the Qwen Research
License; SpikingJelly does not redistribute it. The benchmark README at
``benchmark/snn_llm/qwen_conversion/README.md`` records the pinned revisions,
acceptance thresholds, report digests, correctness evidence, and performance
provenance behind this table.

.. [#sta] Jiang Y., et al., "Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers", ICLR 2024.

.. [#spikezip] You K., et al., "SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN", ICML 2024.
