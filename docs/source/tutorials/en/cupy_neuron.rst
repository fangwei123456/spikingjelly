Neuron Kernel Organization
===========================

Neuron users choose a device, not a kernel library. CPU execution uses Torch;
CUDA execution is registered with PyTorch and selects a compatible implementation
automatically. The CUDA choices currently include native CUDA, Triton, CuPy, and
a Torch fallback.

The implementation files live in the repository-root ``ops/`` directory, with
one subpackage per neuron family. They are installed as ``spikingjelly._ops`` in
the same Python distribution. Users should call neuron classes or the public
``functional.*_step`` and ``functional.*_multi_step`` functions; provider modules
are internal implementation details.

See :doc:`triton_backend` for automatic selection, diagnostics, and optional
provider controls.

Fixed Kernels and Custom Neurons
--------------------------------

Native CUDA and CuPy implementations of fixed neurons use explicit
``kernels.cuh`` sources in each neuron subpackage. CuPy JIT-compiles and launches
these sources without Auto CUDA neuron generation. Fused IF/LIF-Linear also uses
explicit CUDA sources; backward rematerializes spikes with the same charge and
reset formulas as forward.

The Auto CUDA translator and ``surrogate.cuda_codes()`` have been removed.
Custom surrogates provide their PyTorch forward and gradient behavior. Use
:doc:`flexsn` for custom multi-step neurons. FlexSN generates Triton kernels;
it does not provide the old translator's CUDA source output or standalone
single-step kernel generation.

The old generator's global thread-count, compiler-options, compiler-selection,
and neuron bool-spike storage settings are also removed. Explicit operators own
their compilation and launch parameters. Binary Linear/convolution still uses
``configure.save_bool_spike_level`` for backward-input compression.
