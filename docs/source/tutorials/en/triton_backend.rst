Automatic neuron execution
==========================

中文版： :doc:`../cn/triton_backend`

From CPU to CUDA
----------------------------

Create a neuron, then move it and its inputs to the target device. Neuron
constructors and properties no longer expose backend selection. The following
IF, LIF and PLIF training code runs on CPU or NVIDIA CUDA:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional, neuron, surrogate

    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for node_type in (neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode):
        node = node_type(
            step_mode="m", surrogate_function=surrogate.ATan()
        ).to(device)
        x = torch.rand(4, 2, 8, device=device, requires_grad=True)  # [T, N, C]
        parameters = list(node.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.01) if parameters else None
        before = [p.detach().clone() for p in parameters]
        spikes = node(x)
        loss = spikes.sum() + node.v.sum()
        loss.backward()
        assert spikes.shape == x.shape and torch.isfinite(x.grad).all()
        if optimizer is not None:
            assert all(p.grad is not None for p in parameters)
            optimizer.step()
            assert any(not torch.equal(old, p) for old, p in zip(before, parameters))
        functional.reset_net(node)  # Reset after backward/update.

Reset independent batches after backward and parameter updates. See :doc:`./neuron`
for continuous state and detach, and :doc:`./install` for installation/local CUDA builds.

Automatic execution and limits
------------------------------

CPU uses the Torch reference implementation. On CUDA, each execution path checks
compatibility on first use of a device and reuses its selection afterward. This
check performs no online profiling; importing SpikingJelly does not initialize
CUDA either. Eager checks native CUDA, Triton and Torch in that order. Inductor
expansion checks Triton, native CUDA and Torch. These orders are the same across
GPU architectures; compatibility can change which implementation is available.
Measure the workload you intend to run.

Automatic dispatch covers IF, LIF, PLIF, QIF, EIF, Izhikevich, I-LIF,
ActivationAwareIF and STBIF. Training modes, dtypes and surrogate support vary
by family; consult the corresponding API. Other neurons retain their own
execution paths, some without dedicated CUDA kernels.

Ordinary fused paths accept FP32/FP16/BF16 inputs with FP32 state and supported
built-in surrogates. Low-precision state and custom surrogates can use Torch
reference equations. Without explicit precision configuration, modules keep
membrane state in the input dtype, including when an existing state is FP32.
An ordinary FP16/BF16 module call therefore uses the reference recurrence.
Functional calls can supply FP32 state with low-precision input directly;
module storage overrides use :doc:`./precision`. See :doc:`./flexsn` for custom cores.

Tensor layouts
----------------------------

The nine built-in point-neuron native CUDA and Triton implementations directly
read compact, nonoverlapping layouts (including channels-last and transposes)
and their ``expand`` broadcast views. Inputs, initial states, saved tensors and
upstream gradients may have different strides. Time remains logical dimension
zero, including when its physical stride is zero or is not the largest stride.
These layouts do not require contiguous copies at the kernel boundary.

Outputs and returned gradients use independent, nonoverlapping storage. PyTorch
reduces gradients back to broadcast sources. Other valid strided views retain
numerical support but may be converted; necessary dtype conversions are separate
from layout copies. The existing dtype, state and surrogate restrictions still
apply, and this contract does not cover FlexSN or fused projection kernels.

Compiler-selected layouts and convolutions may introduce additional copies;
inspect the generated code or a profile before claiming a copy-free model.
Rebuild native extensions after this layout upgrade and regenerate compiled or
exported graphs that assumed contiguous outputs. Layouts specialize Triton code;
warm up the layouts to be measured before timing.

Compilation and CUDA Graphs
----------------------------

Eager and Inductor expansion select separately. This example uses default FP32
state, runs an eager forward/backward to initialize the device and warm up,
then resets state before compiling the model:

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import functional, neuron, surrogate

    device = torch.device("cuda:0")
    model = (
        nn.Sequential(
            nn.Linear(16, 16),
            neuron.LIFNode(
                step_mode="m", surrogate_function=surrogate.ATan()
            ),
            nn.Linear(16, 4),
        )
        .to(device)
        .train()
    )
    x = torch.rand(4, 2, 16, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Initialize device/state and warm up an eager forward/backward.
    model(x).square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    functional.reset_net(model)
    compiled = torch.compile(model, fullgraph=True)
    output = compiled(x)
    output.square().mean().backward()
    optimizer.step()
    functional.reset_net(model)
    assert output.shape == (4, 2, 4)

The example targets PyTorch Inductor; other compiler backends require separate
validation. Initialize/warm explicit precision combinations as described in
:doc:`./precision` before graph capture.

First calls include loading, compilation or JIT; measure after warmup. CUDA
Graphs retain the captured function's selection, so eager capture uses the eager
implementation and compiled capture uses the compiled implementation. Prepare
memory and warm forward/backward calls as required by PyTorch CUDA Graphs.

Compilation may alter fusion and rounding; bitwise equality with eager is not
promised. Keep model, inputs, state precision, reset and synchronization identical
when comparing performance. Standard workflows are in ``benchmark/README.md``.

Queries and logging
----------------------------

Ordinary training needs no implementation query. To diagnose CUDA execution:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    device = torch.device("cuda:0")
    print(functional.neuron_implementation("lif", device))
    print(functional.neuron_implementation("lif", device, execution="compile"))

A query initializes and caches the selected path without computing neuron
outputs or changing module state. The result contains the bound
``implementation`` and ``unavailable`` reasons for earlier candidates.
Low-precision state, custom surrogates and explicit precision policies can
follow other paths; profile the individual call when that distinction matters.

Package logging is disabled by default; selections are logged once. Applications
can enable INFO at their entry point; earlier records are not replayed.
``logger.remove()`` affects global Loguru sinks and is an application decision:

.. code-block:: python

    import sys
    from spikingjelly.logger import logger

    # Configure sinks at the application entry point, before feature imports.
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

    import torch
    from spikingjelly.activation_based import neuron

    node = neuron.LIFNode(step_mode="m").cuda()
    node(torch.rand(4, 2, 8, device="cuda"))
    node.reset()

See :doc:`/APIs/spikingjelly.logger` for logging configuration.

Final state and full voltage traces
-----------------------------------

The default ``store_v_seq=False`` retains only final voltage. Set it to ``True``
to monitor the temporal trace; the extra trace memory grows with time steps.
Both policies support input and initial-state gradients. Final-state execution
passed eager and fullgraph forward/backward checks on RTX 5090, Torch
2.11.0+cu128 and Triton 3.6.0 with FP32/FP16/BF16 inputs. Compilation can use
the default configuration.

Troubleshooting
----------------------------

* Missing extensions/dependencies or known incompatible devices allow auto mode
  to check the next candidate. No available candidate produces an error.
* Kernel, JIT, OOM or gradient failures raise errors; automatic selection does
  not hide execution failures.
* For native loading incompatibility, check build/runtime Torch/CUDA versions and
  target GPU support, rebuilding if needed.
* Advanced diagnostics may set ``SJ_LIF_CUDA_IMPLEMENTATION=triton`` or the matching
  ``SJ_<NEURON>_CUDA_IMPLEMENTATION`` before Python starts. The default is ``auto``;
  strict alternatives are ``cuda``, ``triton`` and ``torch``. Unsupported profiles fail.
* Restart after changing these variables, dependencies or installed extensions.
  Ordinary training needs no diagnostic overrides.
* For state/input shape, device or precision mismatches, check for retained state
  from an independent previous batch.

The current package has no CuPy dependency. See :doc:`./migrate_from_legacy`
for retired installation options and interfaces.
