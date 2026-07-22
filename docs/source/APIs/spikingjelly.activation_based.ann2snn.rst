spikingjelly.activation\_based.ann2snn package
==============================================

Overview
++++++++

``spikingjelly.activation_based.ann2snn`` is organized around two conversion
paths:

* FX graph conversion for recipes that trace and rewrite a
  ``torch.fx.GraphModule``.
* Direct ``nn.Module`` tree conversion for recipes that replace modules without
  FX tracing.

Most users start by choosing a converter, then a conversion recipe. Lower-level
operators, rules, factories, threshold utilities, and helper functions are
documented after the public conversion APIs.

Converters
++++++++++

ANN2SNN exposes two explicit conversion executors:

* ``FXConverter`` converts through a ``torch.fx.GraphModule``. The historical
  public names ``Converter`` and ``ConversionRecipe`` are compatibility aliases
  for ``FXConverter`` and ``FXConversionRecipe``.
* ``ModuleConverter`` converts a plain ``nn.Module`` tree without FX tracing.
  It accepts only ``ModuleConversionRecipe`` instances. This is the path used
  by module-tree conversions such as SpikeZIP QANN-to-SNN.

There is no automatic cross-path dispatch. Passing a module-tree recipe to
``Converter`` / ``FXConverter`` or an FX recipe to ``ModuleConverter`` raises a
``TypeError``.

.. automodule:: spikingjelly.activation_based.ann2snn.converter
   :members:
   :undoc-members:
   :show-inheritance:

Conversion Recipes
++++++++++++++++++

Recipes describe the conversion algorithm. The built-in recipes fall into three
groups:

* **CNN and rate-coding conversion**: ``RateCodingRecipe`` and
  ``LocalThresholdBalancingRecipe``.
* **Transformer FX conversion**: ``TransformerTDEquivalentRecipe`` and
  ``STATransformerRecipe``.
* **Module-tree QANN/LLM-to-SNN conversion**: ``SpikeZIPTFQANNRecipe`` and
  ``Qwen2SNNRecipe``.

FX recipes subclass ``FXConversionRecipe`` and are executed by
``FXConverter`` / ``Converter``. Module-tree recipes subclass
``ModuleConversionRecipe`` and are executed by ``ModuleConverter``. The
historical name ``ConversionRecipe`` is a compatibility alias for
``FXConversionRecipe``.

.. automodule:: spikingjelly.activation_based.ann2snn.recipes
   :members:
   :undoc-members:
   :show-inheritance:

Rate-Coding Rules, Factories, and Thresholds
++++++++++++++++++++++++++++++++++++++++++++

These modules support the ReLU-to-spiking-neuron path used by rate-coding
recipes, primarily ``RateCodingRecipe`` and ``LocalThresholdBalancingRecipe``.
``ActivationRule`` / ``ReLURule`` match FX activation nodes, insert calibration
hooks, and replace calibrated activations with spiking-neuron subgraphs.
``HookFactory``, ``NeuronFactory``, and ``ThresholdOptimizer`` are the matching
calibration and construction utilities.

Transformer TD-equivalent, STA Transformer, and SpikeZIP conversions do not use
this graph-rule interface. They implement their own recipe-specific operator or
module replacement logic.

.. automodule:: spikingjelly.activation_based.ann2snn.rules
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.ann2snn.factories
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.ann2snn.threshold
   :members:
   :undoc-members:
   :show-inheritance:

Stateful Operators and Runtime Modules
++++++++++++++++++++++++++++++++++++++

Temporal-difference (TD) operators in
``spikingjelly.activation_based.ann2snn.operators`` follow stateful
SpikingJelly step-mode semantics:

* ``ann_forward(...)`` runs the ordinary stateless ANN/PyTorch operation and
  does not read or write module memory.
* ``step_mode="s"`` / ``single_step_forward(...)`` consumes one differential
  timestep, updates cumulative memory, and returns one differential output.
* ``step_mode="m"`` / ``multi_step_forward(...)`` consumes a complete sequence
  whose first dimension is time, uses vectorized cumulative-sum /
  temporal-difference execution where implemented, and leaves the final memory.
* Call ``reset()`` before starting an independent sequence.

This means ``single_step_forward`` is not the ordinary ANN forward path for TD
operators. Use ``ann_forward`` when comparing a TD module with the source
PyTorch module at one non-temporal input.

.. automodule:: spikingjelly.activation_based.ann2snn.operators
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.ann2snn.qcfs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.ann2snn.modules
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

Utilities
+++++++++

.. automodule:: spikingjelly.activation_based.ann2snn.utils
   :members:
   :undoc-members:
   :show-inheritance:

Examples
++++++++

.. toctree::
   :maxdepth: 2

   examples <spikingjelly.activation_based.ann2snn.examples>
