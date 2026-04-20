spikingjelly.activation\_based.triton\_kernel.neuron\_kernel package
=======================================================================

This package contains multi-step neuron kernels implemented with Triton.

torch.compile Compatibility
---------------------------------------------------------------------------------------------------------------

Triton neuron backend is now compatible with ``torch.compile`` for IF/LIF/PLIF multi-step kernels.

Compatibility conditions:

- ``torch>=2.6.0`` with Triton installed.
- CUDA device is required (Triton backend does not run on CPU).
- Use neuron modules with ``step_mode='m'`` and ``backend='triton'``.
- Supported surrogate types in Triton backend are ``Sigmoid`` and ``ATan``.

Current limits and notes:

- Unsupported surrogate functions will raise ``NotImplementedError`` in Triton path.
- ``torch.library.triton_op`` is used when available; runtime fallback to ``custom_op`` is supported.
- Known problematic compile configurations from current validation:

  - ``torch.compile(..., mode="reduce-overhead")`` may trigger CUDAGraph output-overwrite runtime errors.
  - ``fullgraph=True`` can trigger backend compiler exceptions (observed on PLIF).
  - ``mode`` and ``options`` cannot be used together on some PyTorch versions.

- It is recommended to use ``backend="inductor"`` with explicit ``options`` and tune CUDAGraph-related options if needed.


IF
---------------------------------------------------------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.triton_kernel.neuron_kernel.integrate_and_fire
   :members:
   :undoc-members:
   :show-inheritance:

LIF
---------------------------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.triton_kernel.neuron_kernel.lif
   :members:
   :undoc-members:
   :show-inheritance:

PLIF
---------------------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.triton_kernel.neuron_kernel.plif
   :members:
   :undoc-members:
   :show-inheritance:
