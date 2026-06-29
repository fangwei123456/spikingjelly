STA-based Transformer ANN2SNN Conversion
========================================

Author: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版：:doc:`../cn/ann2snn_transformer`

This page introduces the Transformer-oriented ANN2SNN path in
``spikingjelly.activation_based.ann2snn``. It focuses on
``STATransformerRecipe``, a training-free conversion recipe based on
Spatio-Temporal Approximation (STA) [#sta]_.

For classical ReLU-to-IFNode rate-coding conversion on CNNs, see
:doc:`ann2snn`. This page covers a separate Transformer conversion workflow.

.. warning::

    STA conversion is not a strict fully spike-driven SNN conversion. In
    ``mode="spiking_encoder"``, the emitted value is a quantized value equal to
    an integer spike count, possibly negative for negative residuals, multiplied
    by a calibrated threshold, not a binary spike tensor. This distinction is
    important and can be controversial when comparing methods under a strict SNN
    definition.

    ``STATransformerRecipe`` should therefore be read as a training-free
    Transformer ANN2SNN approximation workflow, not as a promise of fully
    spike-driven LLM conversion. Integer token-input language models need a
    separate input and embedding contract. The current STA implementation is
    intentionally online and stateful; layer-wise sequence TD execution and
    faster multi-step inference are later execution-backend work, not the
    default path shown here.

STA conversion idea
-------------------

Transformer models contain operators such as affine projections, LayerNorm,
GELU, attention, residual additions, masks, and tensor constants. A direct
ReLU-to-IFNode rate-coding rule is not enough to describe these components.
STA uses an online temporal approximation instead.

Online Differences Without Spikes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the online-equivalent path without spike encoders first. The central
object is a cumulative activation. Let :math:`x` be the original ANN input and
define the input sequence:

.. math::

    x^{(0)} = x,\qquad x^{(t)} = 0,\quad t=1,\ldots,T-1.

The cumulative input after timestep :math:`t` is:

.. math::

    X^{(t)} = \sum_{\tau=0}^{t} x^{(\tau)}.

So :math:`X^{(0)} = X^{(1)} = \cdots = x`. Let :math:`f` denote one function
or block in the original ANN, such as an affine projection, LayerNorm, GELU, or
attention block. The converted STA block is not :math:`f` itself. Denote the
converted single-timestep block by :math:`F_t`. The mathematical relationship
is that :math:`F_t` outputs a temporal difference of the original ANN function
:math:`f`:

.. math::

    F_t\left(X^{(t)}\right)
    =
    f\left(X^{(t)}\right) - f\left(X^{(t-1)}\right),
    \qquad f\left(X^{(-1)}\right) = 0.

In the implementation, :math:`F_t` is realized by a stateful wrapper around the
original operation. It evaluates or reuses the cumulative output of :math:`f`,
stores the previous cumulative output, and returns only the current difference
:math:`\Delta y^{(t)} = F_t(X^{(t)})`.

The accumulated output satisfies:

.. math::

    \sum_{t=0}^{T-1} F_t\left(X^{(t)}\right)
    = f\left(X^{(T-1)}\right) - f\left(X^{(-1)}\right)
    = f(x).

This identity explains the online-equivalent part of STA: if every converted
block emits cumulative-output differences and constants/bias terms are counted
once, summing the timestep outputs recovers the ANN block output.

The online-equivalent path uses the following execution pattern:

* timestep 0 receives the original ANN input;
* later timesteps receive zero-valued floating inputs;
* the converted model runs an internal ``time_steps`` loop;
* stateful converted modules emit cumulative-output differences;
* the wrapper accumulates the outputs and returns the accumulated result.

At a high level:

.. code-block:: text

    y = 0
    for t in range(time_steps):
        if t == 0:
            x_t = original_input
        else:
            x_t = zeros_like(original_input)
        y = y + converted_graph_step(x_t, static_control_tensors)
    return y

Here, ``converted_graph_step`` means one execution of the converted FX graph for
a single internal timestep. The graph contains stateful modules that remember
their previous cumulative outputs, so each call returns only the current
increment.

Spike Encoding Path
^^^^^^^^^^^^^^^^^^^

``mode="spiking_encoder"`` adds a spike encoder after selected increments. For
an analog increment :math:`a^{(t)}` and threshold :math:`V`, the encoder keeps a
residual membrane :math:`r^{(t)}`. The initial residual is zero:

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

Here :math:`s^{(t)}` is the quantized output of this timestep. The residual for
the next timestep is:

.. math::

    r^{(t)} = u^{(t)} - s^{(t)}.

In SNN terms, :math:`r^{(t)}` is the membrane voltage retained after firing,
:math:`n^{(t)}` is an integer spike count that may be negative when the
residual is negative, and :math:`s^{(t)}` is the threshold-weighted spike
output. The update :math:`r^{(t)} = u^{(t)} - s^{(t)}` is a generalized soft
reset: it subtracts the emitted threshold-weighted value from the membrane. When
:math:`n^{(t)} = 1`, this reduces to the usual soft reset; larger positive or
negative integers represent multiple threshold-unit crossings in one timestep.

After :math:`T` timesteps:

.. math::

    \sum_{t=0}^{T-1} s^{(t)}
    =
    \sum_{t=0}^{T-1} a^{(t)}
    - r^{(T-1)}

The encoder output equals the total analog increment minus the final residual. If
:math:`a^{(t)}` is the STA difference :math:`F_t(X^{(t)})`, then the analog sum
is the ANN block output :math:`f(x)`, and the spike-encoded result differs from
it by the final residual. Because STA calibrates thresholds using
``time_steps``, larger :math:`T` gives finer temporal quantization when the
calibrated activation range is fixed.

This matters for Transformers because LayerNorm, GELU, and attention are not
simple ReLU rate-coding layers. The online cumulative-difference view lets the
converted model evaluate their ANN functions on cumulative inputs and then encode
the increments. The method therefore preserves Transformer operator semantics while
introducing spike-like temporal communication at selected module outputs.

Affine modules, ``LayerNorm``, ``GELU``, ``MultiheadAttention``, and floating
FX tensor constants keep online cumulative-difference state. Bias and graph
constants are injected once. Static attention masks and similar control tensors
are preserved across timesteps rather than zeroed.

``mode="spiking_encoder"`` is the recommended public path in this tutorial. It
adds calibrated stateful spike encoders after ``LayerNorm``, ``GELU``, and
``MultiheadAttention`` outputs, while keeping the main affine projections
online-equivalent. The thresholds are calibrated from a dataloader and depend
on ``time_steps``.

Using STATransformerRecipe
--------------------------

The minimum Python API follows the same Recipe + Converter template as other
ANN2SNN recipes:

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

``time_steps`` is part of the recipe because it is used both by threshold
calibration and by the converted model's internal inference loop. Unlike
rate-coded CNN conversion, users do not call the converted Transformer
once per timestep from Python; the wrapper owns that loop.

There are three STA modes:

* ``equivalent``: online cumulative-difference baseline without calibration;
* ``spiking_encoder``: calibrated spike encoders on nonlinear and attention
  outputs;
* ``spiking_affine``: an advanced path that also replaces selected affine
  modules with spiking affine modules.

This tutorial uses ``spiking_encoder`` for the model-level result.

Relation to TransformerSpikeEquivalentRecipe
--------------------------------------------

``TransformerSpikeEquivalentRecipe`` is a dataloader-free replacement path for
supported Transformer operators using TD / spike-equivalent modules. It is a
useful operator-level conversion baseline, but it does not perform STA
calibration and does not own an internal timestep loop.

``STATransformerRecipe`` is a model-level STA workflow. It uses calibration
when spike encoders are enabled and returns a wrapper module that runs
``time_steps`` internally.

ViT-B/16 ImageNet example
-------------------------

The runnable example is
``spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta``. It uses
``torchvision.models.vit_b_16`` with
``ViT_B_16_Weights.DEFAULT`` and an ImageNet validation directory readable by
``torchvision.datasets.ImageFolder``.

The command below assumes that ``/path/to/imagenet/val`` directly contains the
class folders. CUDA is required for this example.

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python -m spikingjelly.activation_based.ann2snn.examples.imagenet_vit_sta \
      --data-root /path/to/imagenet/val \
      --device cuda:0 \
      --batch-size 64 \
      --num-workers 8 \
      --calib-samples 2048 \
      --time-steps 8 \
      --threshold-scale 0.5

To check the environment, run a small validation subset:

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

The full ImageNet validation run below was measured on an NVIDIA A100-SXM4-80GB
with the full 50000-image validation set:

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
      - 80.590
      - 95.074

The Top-1 drop is 0.478 percentage points. The measured inference time was
about 115.4 seconds for the ANN baseline and 1197.1 seconds for the online STA
converted model in the original run. A rerun with the same accuracy result
measured about 250.8 seconds for the ANN baseline and 2613.1 seconds for STA,
reflecting sensitivity to runtime conditions.

The key stdout lines are:

.. code-block:: shell

    BASELINE {"top1": 0.81068, "top5": 0.95318, "total": 50000, "seconds": 115.39487862586975}
    STA_SPIKING_ENCODER_T8_S05 {"top1": 0.8059, "top5": 0.95074, "total": 50000, "seconds": 1197.0657494068146}
    DROP 0.0047800000000000065

.. [#sta] Y. Jiang, K. Hu, T. Zhang, H. Gao, Y. Liu, Y. Fang, and F. Chen,
   "Spatio-Temporal Approximation: A Training-Free SNN Conversion for
   Transformers," ICLR 2024. https://openreview.net/forum?id=XrunSYwoLr
