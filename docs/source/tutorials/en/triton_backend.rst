Automatic Neuron Execution
==========================

中文版： :doc:`../cn/triton_backend`

From CPU to CUDA
----------------------------

Ordinary neurons have no backend argument or mutable backend property. Create a
neuron and move the module and inputs to the target device. The same training
workflow works on CPU and NVIDIA CUDA:

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
for continuous state and detach, and :doc:`/index` for installation/local CUDA builds.

Automatic execution and limits
------------------------------

CPU uses Torch reference execution. CUDA checks compatible implementations on
first use of a device/execution path and reuses the selection, without online
profiling. Importing SpikingJelly does not initialize CUDA. Eager currently prefers
compatible native CUDA, then Triton and Torch; Inductor expansion prefers Triton,
then native CUDA and Torch. Verified GPU priorities come from offline tests, not a
promise that one implementation wins every workload.

The unified execution interface covers IF, LIF, PLIF, QIF, EIF, Izhikevich, I-LIF,
ActivationAwareIF and STBIF. This does not imply that every family supports training,
all dtypes or arbitrary surrogates; consult each API. Other neurons retain their
public contracts and do not necessarily have dedicated CUDA kernels.

Ordinary fused paths support FP32/FP16/BF16 inputs with FP32 state and supported
built-in surrogates. Low-precision state and custom surrogates can use Torch
reference equations. Explicit neuron precision is a separate policy; see
:doc:`./precision`. For custom cores, see :doc:`./flexsn`.

Compilation and CUDA Graphs
----------------------------

Eager and Inductor expansion select separately. This example uses default FP32
state and initializes the device through an eager forward/backward before
compilation, resetting the state produced by warmup:

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

First calls include loading, compilation or JIT and are not steady-state latency.
CUDA Graphs retain the warmed choice of the captured function: capturing eager
execution does not automatically switch to Triton; compiled capture retains its
compiled path. Follow PyTorch CUDA Graph memory and forward/backward warmup rules.

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

Queries initialize/cache the selected path without neuron computation or module
state changes. The result contains ``implementation`` and ``unavailable`` reasons
for earlier candidates. It is not a per-call profiler: low-precision state,
custom surrogates and explicit precision policies can follow separate paths.

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

The default ``store_v_seq=False`` retains only final voltage. Enable ``True``
when monitoring the full temporal trace; it adds voltage-trace memory proportional
to time steps. Both policies support input and initial-state gradients. Final-state
execution was verified on RTX 5090, Torch 2.11.0+cu128 and Triton 3.6.0 with
FP32/FP16/BF16 inputs in eager and fullgraph forward/backward. A full trace is
not required to avoid a compiler error.

Troubleshooting
----------------------------

* Missing extensions/dependencies or known incompatible devices allow auto mode
  to check the next candidate. No available candidate produces an error.
* Unknown kernel, JIT, OOM or gradient failures are reported without silent fallback.
* For native loading incompatibility, check build/runtime Torch/CUDA versions and
  target GPU support, rebuilding if needed.
* Advanced diagnostics may set ``SJ_LIF_CUDA_IMPLEMENTATION=triton`` or the matching
  ``SJ_<NEURON>_CUDA_IMPLEMENTATION`` before Python starts. The default is ``auto``;
  strict alternatives are ``cuda``, ``triton`` and ``torch``. Unsupported profiles fail.
* Restart after changing these variables, dependencies or installed extensions.
  They are not model constructor parameters or required training steps.
* For state/input shape, device or precision mismatches, check for retained state
  from an independent previous batch.

The current package has no CuPy dependency. See :doc:`./migrate_from_legacy`
for retired installation options and interfaces.
