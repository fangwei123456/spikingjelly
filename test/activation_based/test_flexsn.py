import copy

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import FlexSN


def lif_core(x, v):
    h = v + (x - v) / 2.0
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike)


def if_core(x, v):
    h = v + x
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike)


def plif_core(x, v, w):
    h = v + w.sigmoid() * (x - v)
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike)


def eif_core(x, v):
    h = v + (x - v + torch.exp(v - 0.8)) / 2.0
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike)


def qif_core(x, v):
    h = v + (x + (v - 0.0) * (v - 0.8)) / 2.0
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike)


def izhikevich_core(x, v, w):
    h = v + (x + (v + 0.1) * (v - 0.8) - w) / 2.0
    w = w + (0.1 * (v + 0.1) - w) / 2.0
    spike = torch.sigmoid(h - 1.0)
    return spike, h * (1.0 - spike), w + 0.1 * spike


def manual_scan(core, inputs, states, static_inputs=()):
    output_steps = None
    for step_inputs in zip(*inputs, strict=True):
        returns = tuple(core(*step_inputs, *states, *static_inputs))
        outputs = returns[: len(returns) - len(states)]
        states = returns[len(outputs) :]
        if output_steps is None:
            output_steps = [[] for _ in outputs]
        for values, value in zip(output_steps, outputs, strict=True):
            values.append(value)
    return tuple(torch.stack(values) for values in output_steps), states


def test_public_surface_only_exports_flexsn():
    from spikingjelly.activation_based.neuron import flexsn

    assert flexsn.__all__ == ["FlexSN"]
    assert not hasattr(flexsn, "FlexSNKernel")


def test_torch_managed_and_functional_state_are_equivalent():
    x = torch.randn(4, 3)
    module = FlexSN(
        lif_core,
        num_states=1,
        backend="torch",
        store_state_seqs=True,
    )
    initial_state = (torch.zeros_like(x[0]),)

    outputs, next_states = module.functional_forward(
        (x,), initial_state, static_inputs=()
    )

    assert module.states == (None,)
    assert module.state_seqs is None
    torch.testing.assert_close(module(x), outputs[0])
    torch.testing.assert_close(module.states[0], next_states[0])
    torch.testing.assert_close(module.state_seqs[0][-1], next_states[0])


def test_static_parameter_and_buffer_are_registered_and_differentiable():
    def core(x, v, w, bias):
        v = v + w.sigmoid() * (x + bias - v)
        return v, v

    w = torch.nn.Parameter(torch.tensor(0.0))
    bias = torch.ones(3)
    module = FlexSN(core, 1, (w, bias), backend="torch")
    x = torch.randn(4, 3, requires_grad=True)

    module(x).sum().backward()

    assert tuple(module.parameters()) == (w,)
    assert tuple(module.buffers()) == (bias,)
    assert set(module.state_dict()) == {"_static_input_0", "_static_input_1"}
    assert w.grad is not None
    assert x.grad is not None


def test_functional_forward_accepts_explicit_static_inputs():
    def core(x, v, gain):
        v = v + gain * x
        return v, v

    module = FlexSN(core, 1, (torch.tensor(1.0),), backend="torch")
    x = torch.ones(3, 2)
    initial = (torch.zeros(2),)

    outputs, states = module.functional_forward(
        (x,), initial, static_inputs=(torch.tensor(2.0),)
    )

    torch.testing.assert_close(
        outputs[0], torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    )
    torch.testing.assert_close(states[0], torch.tensor([6.0, 6.0]))
    assert module.states == (None,)


def test_multiple_inputs_outputs_and_states_use_tuples():
    def core(x, y, a, b):
        a = a + x
        b = b + y
        return a, b, a, b

    x = torch.randn(4, 3)
    y = torch.randn(4, 3)
    module = FlexSN(core, 2, backend="torch", store_state_seqs=True)

    outputs = module(x, y)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert isinstance(module.states, tuple)
    assert isinstance(module.state_seqs, tuple)
    torch.testing.assert_close(outputs[0], x.cumsum(0))
    torch.testing.assert_close(outputs[1], y.cumsum(0))


def test_single_step_torch_mode():
    module = FlexSN(lif_core, 1, step_mode="s", backend="torch")
    x = torch.randn(3)

    output = module(x)

    expected, state = lif_core(x, torch.zeros_like(x))
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(module.states[0], state)


def test_backend_and_step_mode_switches_preserve_states():
    module = FlexSN(lif_core, 1, backend="torch", store_state_seqs=True)
    module(torch.randn(2, 3))
    state = module.states[0]

    module.backend = "hop"
    assert module.states[0] is state
    assert module.state_seqs is None
    with pytest.raises(RuntimeError, match="does not support"):
        module.step_mode = "s"

    module.backend = "torch"
    module.step_mode = "s"
    assert module.states[0] is state
    with pytest.raises(RuntimeError, match="requires step_mode"):
        module.backend = "triton"


def test_hop_matches_torch_forward_and_backward():
    x_torch = torch.randn(4, 8, requires_grad=True)
    x_hop = x_torch.detach().clone().requires_grad_(True)
    torch_module = FlexSN(lif_core, 1, backend="torch", store_state_seqs=True)
    hop_module = FlexSN(lif_core, 1, backend="hop", store_state_seqs=True)

    y_torch = torch_module(x_torch)
    y_hop = hop_module(x_hop)
    y_torch.sum().backward()
    y_hop.sum().backward()

    torch.testing.assert_close(y_hop, y_torch)
    torch.testing.assert_close(hop_module.states[0], torch_module.states[0])
    torch.testing.assert_close(hop_module.state_seqs[0], torch_module.state_seqs[0])
    torch.testing.assert_close(x_hop.grad, x_torch.grad)


def test_hop_fullgraph_compile_matches_eager():
    x = torch.randn(4, 8)
    eager = FlexSN(lif_core, 1, backend="hop")
    compiled = torch.compile(FlexSN(lif_core, 1, backend="hop"), fullgraph=True)

    torch.testing.assert_close(compiled(x), eager(x))


def test_copy_preserves_configuration_and_state_without_runtime():
    module = FlexSN(lif_core, 1, backend="torch")
    module(torch.randn(2, 3))

    copied = copy.deepcopy(module)

    assert copied.backend == module.backend
    assert copied.step_mode == module.step_mode
    assert copied._triton_handle is None
    torch.testing.assert_close(copied.states[0], module.states[0])


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FlexSN(lif_core, -1), "num_states"),
        (lambda: FlexSN(lif_core, 1, (1.0,)), "static_inputs"),
        (
            lambda: FlexSN(lif_core, 1, step_mode="s", backend="triton"),
            "requires step_mode",
        ),
    ],
)
def test_constructor_rejects_invalid_contract(factory, message):
    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        factory()


def test_rejects_tensor_closure():
    bias = torch.ones(3)

    def core(x, v):
        v = v + x + bias
        return v, v

    with pytest.raises(TypeError, match="static_inputs"):
        FlexSN(core, 1)


def test_rejects_empty_sequence_and_arity_changes():
    module = FlexSN(lif_core, 1, backend="torch")
    with pytest.raises(ValueError, match="empty"):
        module(torch.empty(0, 3))

    module(torch.randn(2, 3))
    with pytest.raises(ValueError, match="expects 1 inputs"):
        module(torch.randn(2, 3), torch.randn(2, 3))


def test_rejects_mismatched_tensor_contract():
    module = FlexSN(lif_core, 1, backend="torch")
    module.states = (torch.zeros(4),)
    with pytest.raises(ValueError, match="numel"):
        module(torch.randn(2, 3))


def test_triton_requires_cuda_without_fallback():
    if torch.cuda.is_available():
        pytest.skip("CPU-only failure contract")
    module = FlexSN(lif_core, 1, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.randn(2, 3))


def test_triton_registered_operator_surface_is_minimal():
    from spikingjelly.activation_based.triton_kernel.flexsn import custom_ops

    assert custom_ops.__all__ == []
    assert str(torch.ops.sj.flexsn_triton_inference.default._schema) == (
        "sj::flexsn_triton_inference(SymInt handle, Tensor[] flat_args, "
        "bool return_state_sequences) -> Tensor[]"
    )
    assert str(torch.ops.sj.flexsn_triton_training.default._schema) == (
        "sj::flexsn_triton_training(SymInt handle, Tensor[] flat_args, "
        "bool return_state_sequences) -> Tensor[]"
    )


def _make_core_case(name, backend, dtype):
    if name == "plif":
        parameter = torch.nn.Parameter(torch.tensor(0.0, device="cuda", dtype=dtype))
        return FlexSN(
            plif_core,
            1,
            static_inputs=(parameter,),
            backend=backend,
        ), parameter
    core, num_states = {
        "if": (if_core, 1),
        "lif": (lif_core, 1),
        "eif": (eif_core, 1),
        "qif": (qif_core, 1),
        "izhikevich": (izhikevich_core, 2),
    }[name]
    return FlexSN(core, num_states, backend=backend), None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", ["if", "lif", "plif", "eif", "qif", "izhikevich"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_representative_triton_cores_match_torch(name, dtype):
    torch_module, torch_parameter = _make_core_case(name, "torch", dtype)
    triton_module, triton_parameter = _make_core_case(name, "triton", dtype)
    x_torch = torch.randn(4, 32, device="cuda", dtype=dtype, requires_grad=True)
    x_triton = x_torch.detach().clone().requires_grad_(True)
    torch_states = tuple(
        torch.randn(32, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(torch_module.num_states)
    )
    triton_states = tuple(
        state.detach().clone().requires_grad_(True) for state in torch_states
    )
    torch_module.states = torch_states
    triton_module.states = triton_states

    torch_output = torch_module(x_torch)
    triton_output = triton_module(x_triton)
    torch_loss = torch_output.sum() + sum(state.sum() for state in torch_module.states)
    triton_loss = triton_output.sum() + sum(
        state.sum() for state in triton_module.states
    )
    torch_loss.backward()
    triton_loss.backward()

    tolerance = {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.float16 else {}
    torch.testing.assert_close(triton_output, torch_output, **tolerance)
    torch.testing.assert_close(x_triton.grad, x_torch.grad, **tolerance)
    for triton_state, torch_state in zip(triton_states, torch_states, strict=True):
        torch.testing.assert_close(triton_state.grad, torch_state.grad, **tolerance)
    if torch_parameter is not None:
        torch.testing.assert_close(
            triton_parameter.grad, torch_parameter.grad, **tolerance
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_matches_torch_and_is_captured_by_compile():
    x_torch = torch.randn(4, 32, device="cuda", requires_grad=True)
    x_triton = x_torch.detach().clone().requires_grad_(True)
    torch_module = FlexSN(lif_core, 1, backend="torch")
    triton_module = FlexSN(lif_core, 1, backend="triton")

    y_torch = torch_module(x_torch)
    y_triton = triton_module(x_triton)
    y_torch.sum().backward()
    y_triton.sum().backward()

    torch.testing.assert_close(y_triton, y_torch)
    torch.testing.assert_close(x_triton.grad, x_torch.grad)

    from torch._dynamo import explain

    fresh = FlexSN(lif_core, 1, backend="triton").cuda()
    with torch.no_grad():
        compiled_output = torch.compile(fresh, fullgraph=True)(x_triton.detach())
    assert compiled_output.shape == x_triton.shape
    explanation = explain(fresh)(x_triton.detach())
    targets = [
        str(node.target) for graph in explanation.graphs for node in graph.graph.nodes
    ]
    assert any("sj.flexsn_triton" in target for target in targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("training", [False, True])
def test_triton_registered_operators_pass_opcheck(training):
    module = FlexSN(lif_core, 1, backend="triton")
    x = torch.randn(4, 32, device="cuda", requires_grad=training)
    state = torch.zeros(32, device="cuda", requires_grad=training)
    operator = (
        torch.ops.sj.flexsn_triton_training.default
        if training
        else torch.ops.sj.flexsn_triton_inference.default
    )

    result = torch.library.opcheck(
        operator,
        (module._triton_handle, [x, state], False),
        raise_exception=False,
    )

    assert set(result.values()) == {"SUCCESS"}
