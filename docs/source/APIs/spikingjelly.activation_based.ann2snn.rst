spikingjelly.activation\_based.ann2snn package
==============================================

Converter
+++++++++++

.. automodule:: spikingjelly.activation_based.ann2snn.converter
   :members:
   :undoc-members:
   :show-inheritance:

Extension Points
++++++++++++++++

The ``recipes`` module is the public extension surface for ANN2SNN conversion
algorithms. A recipe defines the algorithm-specific steps, while
``Converter.convert(model)`` owns execution.

.. automodule:: spikingjelly.activation_based.ann2snn.recipes
   :members:
   :undoc-members:
   :show-inheritance:

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

Helper Modules and Functions
++++++++++++++++++++++++++++++

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

.. automodule:: spikingjelly.activation_based.ann2snn.modules
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: extra_repr

.. automodule:: spikingjelly.activation_based.ann2snn.utils
   :members:
   :undoc-members:
   :show-inheritance:

Examples
++++++++++++

.. toctree::
   :maxdepth: 2

   examples <spikingjelly.activation_based.ann2snn.examples>
