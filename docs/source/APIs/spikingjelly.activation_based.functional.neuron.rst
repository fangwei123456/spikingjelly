Neuron State Updates
++++++++++++++++++++

``functional.neuron`` contains explicit-state transitions and sequence
functions used by neuron modules. A ``*_step`` function advances one time step;
``*_multi_step`` consumes a time-major sequence and returns the updated state.
The neuron module owns persistent state and reset behavior.

The input tensor device selects execution. CPU uses the Torch reference. CUDA uses
the registered operator and automatically selects a compatible implementation
for that device. Users do not pass or store a neuron backend. Call
:func:`neuron_implementation` to inspect the selected CUDA implementation and
the reasons other candidates were unavailable.

The optional ``SJ_<NEURON>_CUDA_IMPLEMENTATION`` environment variables are for
diagnostics and require a fresh process. Their default is ``auto``; an explicit
value is strict and raises an error when that implementation cannot serve the
requested execution.

----

``functional.neuron`` provides explicit-state transitions and sequence
functions used by neuron modules. A ``*_step`` function advances one time step;
``*_multi_step`` consumes a time-major sequence and returns the updated state.
The neuron module owns persistent state and reset behavior.

Execution follows the input tensor device. CPU uses the Torch reference. CUDA
uses the registered operator and automatically selects a compatible
implementation for the device. Users do not pass or store a neuron backend. Use
:func:`neuron_implementation` to inspect the selected CUDA implementation and
why other candidates were unavailable.

The optional ``SJ_<NEURON>_CUDA_IMPLEMENTATION`` environment variables are
diagnostic controls and require a fresh process. They default to ``auto``; an
explicit value is strict and raises an error when the implementation cannot
serve the requested execution.

.. automodule:: spikingjelly.activation_based.functional.neuron
   :members:
   :undoc-members:
