Automatic Neuron Execution
==========================

Neuron modules select execution from the input tensor's device. CPU uses the
Torch reference implementation. CUDA calls a registered PyTorch operator, which
selects and caches compatible implementations per device and execution path.
Eager uses offline priorities per neuron family and GPU compute capability;
unknown devices retain native CUDA, Triton, CuPy, then Torch. Inductor expansion
prefers Triton, native CUDA, CuPy, then Torch in a separate cache. Eager calls do
not check compilation mode. CUDA Graphs retain the warmed choice of the function
being captured. Selection never runs online profiling.

The neuron API has no backend constructor argument or mutable backend property.
This applies to IF, LIF, PLIF, QIF, EIF, Izhikevich, I-LIF, ActivationAwareIF,
and STBIF. Existing state, reset, surrogate, and step-mode settings remain on the
neuron module.

Inspect the selected implementation when diagnosing a CUDA run:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional, neuron

    device = torch.device("cuda:0")
    lif = neuron.LIFNode(step_mode="m").to(device)
    output = lif(torch.rand(4, 128, device=device))
    print(functional.neuron_implementation("lif", device))
    print(functional.neuron_implementation("lif", device, execution="compile"))

The query reports the selected provider and why earlier candidates were
unavailable. Importing SpikingJelly does not initialize CUDA; selection happens
on first use of each path (or an explicit diagnostic query). SpikingJelly logs
each path's selection once through its logger. Compiled selection targets the
Inductor expansion; arbitrary compiler backends need separate evaluation.

Advanced diagnostics may set ``SJ_<NEURON>_CUDA_IMPLEMENTATION`` before Python
starts, for example ``SJ_LIF_CUDA_IMPLEMENTATION=triton``. The default is
``auto``. A forced provider is strict: if it is unavailable for the requested
profile, the operator raises an error. Unknown kernel and execution failures are
also raised rather than hidden by fallback. Restart the process after changing
these variables.

Operator sources are kept in the repository-root ``ops/`` tree and installed
inside the SpikingJelly package as ``spikingjelly._ops``. Triton and CuPy retain
their JIT compilation and caches. The optional native CUDA implementation is
built locally when the matching PyTorch/CUDA toolchain is available; regular
PyPI wheels remain pure Python.
