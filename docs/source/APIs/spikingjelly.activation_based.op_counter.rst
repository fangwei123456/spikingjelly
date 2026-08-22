spikingjelly.activation_based.op_counter package
=====================================================

Quick Start
+++++++++++

Count one real execution with one or more counters:

.. code-block:: python

   from spikingjelly.activation_based import op_counter

   counter = op_counter.FlopCounter()
   with op_counter.DispatchCounterMode([counter]):
       model(x)
   print(counter.get_total())

Use the basic counters for runtime counts. Use an energy estimator when you
need a specific cost model; the estimators are not interchangeable.

Built-in counters register ATen rules and should use ``DispatchCounterMode``.
Use ``FunctionCounterMode`` only with custom counters whose rules target the
intercepted ``torch.*`` functions.

Base Classes and Context Managers
++++++++++++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.base
   :members:
   :undoc-members:
   :show-inheritance:

FLOP Counter
++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.flop
   :members:
   :undoc-members:
   :show-inheritance:

Memory Access Counter
++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.memory_access
   :members:
   :undoc-members:
   :show-inheritance:

MAC / AC / SynOp Counters
++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.mac
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.op_counter.ac
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.op_counter.synop
   :members:
   :undoc-members:
   :show-inheritance:

Neuromorphic Memory Counter
++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.neuromorphic_memory_access
   :members:
   :undoc-members:
   :show-inheritance:

Simple Runtime Energy Estimator
+++++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.simple_energy
   :members:
   :undoc-members:
   :show-inheritance:

Analytical and Runtime Energy Modules
++++++++++++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.lemaire_addressing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.op_counter.memory_residency
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.op_counter.neuron_state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.op_counter.analytical_energy
   :members:
   :undoc-members:
   :show-inheritance:

NeuroMC Energy Profiler
+++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.neuromc.core
   :members:
   :undoc-members:
   :show-inheritance:

SpikeSim Event-Driven Energy Profiler
++++++++++++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.op_counter.spikesim
   :members:
   :undoc-members:
   :show-inheritance:
