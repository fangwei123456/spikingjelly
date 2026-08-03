spikingjelly.activation_based.cuda_kernel.neuron_kernel package
===============================================================

``neuron_kernel`` owns concrete CuPy neuron kernels and the public multi-step
custom-kernel extension interface. Generic CUDA code-generation primitives remain
in :mod:`spikingjelly.activation_based.cuda_kernel.auto_cuda`.

Implementation overview
------------------------

The built-in kernels use two implementation styles:

* **AutoCUDA-generated:** single-step IF/LIF and multi-step IF/LIF/PLIF kernels.
* **Hand-written CUDA:** multi-step EIF/QIF/Izhikevich kernels, which build fixed
  CUDA sources directly with :class:`cupy.RawKernel`.

``auto_cuda`` provides code-generation primitives only; neuron-specific kernels
remain under ``neuron_kernel``.

.. automodule:: spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step
   :members:
   :show-inheritance:
