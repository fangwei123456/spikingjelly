import copy
import functools
import gc
import inspect
import linecache
import sys
import types
import weakref

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


def hard_if_core(x, v):
    h = v + x
    spike = (h >= 1.0).to(h.dtype)
    return spike, h * (1.0 - spike)


def test_public_surface_only_exports_flexsn():
    from spikingjelly.activation_based.neuron import flexsn

    assert flexsn.__all__ == ["FlexSN"]
    assert not hasattr(flexsn, "FlexSNKernel")


def test_codegen_executes_in_memory_and_preserves_namespace(monkeypatch, tmp_path):
    from spikingjelly._ops.torch2triton import graph2triton

    triton = types.ModuleType("triton")

    def jit(function):
        return types.SimpleNamespace(fn=function, src=inspect.getsource(function))

    triton.jit = jit
    monkeypatch.setattr(graph2triton, "triton", triton)
    monkeypatch.setattr(graph2triton, "tl", types.ModuleType("triton.language"))
    monkeypatch.setenv("HOME", str(tmp_path))

    namespace = {"offset": 2}
    modules_before = set(sys.modules)
    sources_before = {
        name for name in linecache.cache if "_ops.torch2triton.generated" in name
    }
    kernel = graph2triton.compile_triton_code_str(
        "@triton.jit\ndef generated(x):\n    return x + offset\n",
        "generated",
        namespace,
    )
    standalone = graph2triton.compile_triton_code_str(
        "@triton.jit\ndef standalone(x):\n    return x + 1\n",
        "standalone",
    )

    assert kernel.fn(1) == 3
    assert standalone.fn(1) == 2
    assert "def generated" in kernel.src
    assert namespace["generated"] is kernel
    assert "def generated" in inspect.getsource(kernel.fn)
    assert "def standalone" in inspect.getsource(standalone.fn)
    assert set(sys.modules) - modules_before == set()
    assert not (tmp_path / ".spikingjelly").exists()

    for index in range(64):
        name = f"unique_{index}"
        generated = graph2triton.compile_triton_code_str(
            f"@triton.jit\ndef {name}(x):\n    return x + {index}\n",
            name,
        )
        assert generated.fn(1) == index + 1
        assert "def unique_" in inspect.getsource(generated.fn)
    del generated
    gc.collect()
    sources_after = {
        name for name in linecache.cache if "_ops.torch2triton.generated" in name
    }
    assert len(sources_after - sources_before) == 2
    assert set(sys.modules) - modules_before == set()
    assert not (tmp_path / ".spikingjelly").exists()

    with pytest.raises(SyntaxError) as error:
        graph2triton.compile_triton_code_str("@triton.jit\ndef broken(\n", "broken")
    assert "broken" in str(error.value)
    assert error.value.text.strip() == "def broken("


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


def test_triton_backend_requires_installed_dependency(monkeypatch):
    from spikingjelly.activation_based import base

    monkeypatch.setattr(base, "triton", None)
    with pytest.raises(ImportError, match="Triton is not installed"):
        FlexSN(lif_core, 1, backend="triton")

    module = FlexSN(lif_core, 1, backend="torch")
    with pytest.raises(ImportError, match="Triton is not installed"):
        module.backend = "triton"
    assert module.backend == "torch"


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


@pytest.mark.parametrize("store_state_seqs", [False, True])
def test_hop_fullgraph_bptt_with_static_parameter(store_state_seqs):
    torch_parameter = torch.nn.Parameter(torch.tensor(0.0))
    hop_parameter = torch.nn.Parameter(torch.tensor(0.0))
    torch_module = FlexSN(
        plif_core,
        1,
        static_inputs=(torch_parameter,),
        backend="torch",
        store_state_seqs=store_state_seqs,
    )
    hop_module = FlexSN(
        plif_core,
        1,
        static_inputs=(hop_parameter,),
        backend="hop",
        store_state_seqs=store_state_seqs,
    )
    x_torch = torch.randn(4, 8, requires_grad=True)
    x_hop = x_torch.detach().clone().requires_grad_(True)

    torch_output = torch_module(x_torch)
    hop_output = torch.compile(hop_module, fullgraph=True)(x_hop)
    torch_output.sum().backward()
    hop_output.sum().backward()

    torch.testing.assert_close(hop_output, torch_output)
    torch.testing.assert_close(x_hop.grad, x_torch.grad)
    torch.testing.assert_close(hop_parameter.grad, torch_parameter.grad)


def test_hop_has_one_private_implementation():
    from spikingjelly._ops.flexsn import hop

    assert hop.__all__ == []
    assert not hasattr(hop, "lowerable_scan")
    assert not hasattr(hop, "lowerable_while_loop_scan")
    assert not hasattr(hop, "eager_scan_final_state")


def test_copy_preserves_configuration_and_state_without_runtime():
    module = FlexSN(lif_core, 1, backend="torch")
    module(torch.randn(2, 3))

    copied = copy.deepcopy(module)

    assert copied.backend == module.backend
    assert copied.step_mode == module.step_mode
    assert copied._triton_handle is None
    torch.testing.assert_close(copied.states[0], module.states[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_deepcopy_rebuilds_runtime():
    module = FlexSN(lif_core, 1, backend="triton")
    copied = copy.deepcopy(module)
    x = torch.randn(4, 32, device="cuda")

    assert copied._triton_handle is None
    with torch.no_grad():
        expected = module(x)
        actual = copied(x)

    assert copied._triton_handle is not None
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("factory", "exception", "message"),
    [
        (lambda: FlexSN(lif_core, -1), ValueError, "num_states"),
        (lambda: FlexSN(lif_core, 1, (1.0,)), TypeError, "static_inputs"),
        (
            lambda: FlexSN(lif_core, 1, step_mode="s", backend="triton"),
            RuntimeError,
            "requires step_mode",
        ),
    ],
)
def test_constructor_rejects_invalid_contract(factory, exception, message):
    with pytest.raises(exception, match=message):
        factory()


def test_rejects_tensor_closure():
    bias = torch.ones(3)

    def core(x, v):
        v = v + x + bias
        return v, v

    with pytest.raises(TypeError, match="static_inputs"):
        FlexSN(core, 1)


def test_rejects_tensor_in_nested_partial():
    bias = torch.ones(3)

    def core(bias, x, v):
        return x + bias, v

    nested = functools.partial(functools.partial(core, bias))
    with pytest.raises(TypeError, match="static_inputs"):
        FlexSN(nested, 1)


def test_partial_core_name_is_informative():
    core = functools.partial(lif_core)
    assert "partial(lif_core)" in repr(FlexSN(core, 1, backend="torch"))


def test_constructor_wraps_arity_inference_failure():
    def core(x, v):
        raise AssertionError("requires real inputs")

    with pytest.raises(RuntimeError, match="construction-time unit tensors"):
        FlexSN(core, 1, backend="torch")


def test_rejects_empty_sequence_and_arity_changes():
    module = FlexSN(lif_core, 1, backend="torch")
    with pytest.raises(ValueError, match="empty"):
        module(torch.empty(0, 3))

    module(torch.randn(2, 3))
    with pytest.raises(ValueError, match="expects 1 inputs"):
        module(torch.randn(2, 3), torch.randn(2, 3))


def test_rejects_scalar_multi_step_input():
    module = FlexSN(lif_core, 1, backend="torch")
    with pytest.raises(ValueError, match="time dimension"):
        module(torch.tensor(1.0))


def test_rejects_mismatched_tensor_contract():
    module = FlexSN(lif_core, 1, backend="torch")
    module.states = (torch.zeros(4),)
    with pytest.raises(ValueError, match="numel"):
        module(torch.randn(2, 3))


def test_triton_requires_cuda_without_fallback():
    if torch.cuda.is_available():
        pytest.skip("CPU-only failure contract")
    pytest.importorskip("triton")
    module = FlexSN(lif_core, 1, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.randn(2, 3))


def test_triton_registered_operator_surface_is_minimal():
    from spikingjelly._ops.flexsn import triton as custom_ops

    assert custom_ops.__all__ == []
    assert str(torch.ops.sj_flexsn.triton_inference.default._schema) == (
        "sj_flexsn::triton_inference(SymInt handle, Tensor[] flat_args, "
        "bool return_state_sequences) -> Tensor[]"
    )
    assert str(torch.ops.sj_flexsn.triton_training.default._schema) == (
        "sj_flexsn::triton_training(SymInt handle, Tensor[] flat_args, "
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
    x_torch = torch.randn(4, 513, device="cuda", dtype=dtype, requires_grad=True)
    x_triton = x_torch.detach().clone().requires_grad_(True)
    torch_states = tuple(
        torch.randn(513, device="cuda", dtype=dtype, requires_grad=True)
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
    assert any("sj_flexsn.triton" in target for target in targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_runtime_rebuilds_for_input_dtype():
    module = FlexSN(lif_core, 1, backend="triton")
    initial_handle = module._triton_handle
    x = torch.randn(4, 32, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        actual = module(x)
        expected = FlexSN(lif_core, 1, backend="torch")(x)

    assert module._triton_runtime_dtype == torch.float16
    assert module._triton_handle != initial_handle
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_hard_threshold_inference_matches_torch():
    x = torch.randn(4, 32, device="cuda")
    torch_module = FlexSN(hard_if_core, 1, backend="torch")
    triton_module = FlexSN(hard_if_core, 1, backend="triton")

    with torch.no_grad():
        expected = torch_module(x)
        actual = triton_module(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("training", [False, True])
def test_triton_registered_operators_pass_opcheck(training):
    module = FlexSN(lif_core, 1, backend="triton")
    x = torch.randn(4, 32, device="cuda", requires_grad=training)
    state = torch.zeros(32, device="cuda", requires_grad=training)
    operator = (
        torch.ops.sj_flexsn.triton_training.default
        if training
        else torch.ops.sj_flexsn.triton_inference.default
    )

    result = torch.library.opcheck(
        operator,
        (module._triton_handle, [x, state], False),
        raise_exception=False,
    )

    assert set(result.values()) == {"SUCCESS"}


def test_frontend_traces_graphs_on_cpu():
    from spikingjelly.activation_based.neuron.flexsn_trace import _trace_core

    examples = (torch.randn(7), torch.randn(7))
    snapshots = tuple(value.clone() for value in examples)
    inference, forward, backward, differentiable = _trace_core(
        hard_if_core, examples, 1, 1
    )
    actual = torch.fx.GraphModule({}, inference)(*examples)
    torch.testing.assert_close(actual, hard_if_core(*examples))
    forward.lint()
    backward.lint()
    assert differentiable == [False, True]
    torch.testing.assert_close(examples, snapshots)


def coupled_core(x, y, v, w, gain):
    h = 0.5 * v + x * gain + y
    z = torch.sigmoid(h)
    return z, h + w, h * (1.0 - z), 0.25 * w + z


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("store_state_seqs", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("time_steps", [1, 4])
def test_migrated_triton_matches_retained_and_torch(
    store_state_seqs, dtype, time_steps
):
    from spikingjelly.activation_based.triton_kernel.flexsn import custom_ops as old
    from spikingjelly.activation_based.triton_kernel.flexsn.kernel import (
        build_inference_kernels,
        build_training_kernels,
    )

    torch.manual_seed(42)
    source = [
        (torch.randn(time_steps, 17, 2, device="cuda", dtype=dtype) * 0.1)[..., 0],
        (torch.randn(time_steps, 17, 2, device="cuda", dtype=dtype) * 0.1)[..., 0],
        torch.randn(17, device="cuda", dtype=dtype) * 0.1,
        torch.randn(17, device="cuda", dtype=dtype) * 0.1,
        torch.tensor(0.25, device="cuda", dtype=dtype),
    ]
    results = []
    for backend in ("torch", "triton", "retained"):
        args = [value.detach().requires_grad_(True) for value in source]
        node = FlexSN(
            coupled_core,
            2,
            (args[-1],),
            backend="torch" if backend == "retained" else backend,
            store_state_seqs=store_state_seqs,
        )
        node.states = tuple(args[2:4])
        if backend == "retained":
            wrapped = node._wrapped_core(2, 2)
            examples = tuple(torch.zeros(1, device="cuda", dtype=dtype) for _ in args)
            ik, fk, ii = build_inference_kernels(wrapped, 2, 3, 2, examples)
            fw, bw, ti = build_training_kernels(wrapped, 2, 3, 2, examples)
            handle = old.register_flexsn_kernel_handle(
                inference_kernel=ik,
                inference_final_state_kernel=fk,
                inference_info=ii,
                forward_kernel=fw,
                backward_kernel=bw,
                training_info=ti,
            )
            finalizer = old.attach_flexsn_handle_finalizer(node, handle)
            values = old.flexsn_triton_training(
                handle, [*args[:4], args[-1].expand_as(args[2])], store_state_seqs
            )
            outputs = tuple(values[:2])
            states = (
                tuple(v[-1] for v in values[2:4])
                if store_state_seqs
                else tuple(values[2:4])
            )
            traces = tuple(values[2:4]) if store_state_seqs else ()
        else:
            outputs = node(*args[:2])
            states = node.states
            traces = node.state_seqs or ()
        loss = sum(t.float().square().sum() for t in (*outputs, *states, *traces))
        grads = torch.autograd.grad(loss, args)
        results.append((outputs, states, traces, grads))
        if backend == "retained":
            finalizer()
    tolerance = {} if dtype == torch.float32 else {"atol": 0.04, "rtol": 0.04}
    torch.testing.assert_close(results[1], results[0], **tolerance)
    torch.testing.assert_close(results[1], results[2], **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("store_state_seqs", [False, True])
def test_triton_fullgraph_backward_and_chunked_state(store_state_seqs):
    torch.manual_seed(42)
    x = torch.randn(4, 33, device="cuda") * 0.1
    expected_node = FlexSN(
        lif_core, 1, backend="torch", store_state_seqs=store_state_seqs
    )
    actual_node = FlexSN(
        lif_core, 1, backend="triton", store_state_seqs=store_state_seqs
    )
    with torch.no_grad():
        actual_node(x)
    actual_node.reset()
    compiled = torch.compile(actual_node, fullgraph=True)
    expected_x = x.detach().requires_grad_(True)
    actual_x = x.detach().requires_grad_(True)
    expected = torch.cat([expected_node(chunk) for chunk in expected_x.chunk(2)])
    actual = torch.cat([compiled(chunk) for chunk in actual_x.chunk(2)])
    expected_loss = expected.sum() + expected_node.states[0].sum()
    actual_loss = actual.sum() + actual_node.states[0].sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_node.states, expected_node.states)
    torch.testing.assert_close(
        torch.autograd.grad(actual_loss, actual_x),
        torch.autograd.grad(expected_loss, expected_x),
    )
    actual_node.reset()
    assert actual_node.states == (None,)
    assert actual_node.state_seqs is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_backward_outlives_node_and_can_repeat():
    from spikingjelly._ops.flexsn.triton import _bundle

    x = torch.randn(3, 17, device="cuda", requires_grad=True)
    reference = FlexSN(lif_core, 1, backend="torch")(x)
    expected = torch.autograd.grad(reference.sum(), x)[0]
    node = FlexSN(lif_core, 1, backend="triton")
    result = node(x)
    kernel_bundle = weakref.ref(_bundle(node._triton_handle))
    del node
    gc.collect()
    first = torch.autograd.grad(result.sum(), x, retain_graph=True)[0]
    second = torch.autograd.grad(result.sum(), x)[0]
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    del result
    gc.collect()
    assert kernel_bundle() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_non_default_stream_and_device():
    target = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device("cuda", target)
    x = torch.randn(4, 19, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.no_grad():
        expected = FlexSN(lif_core, 1, backend="torch")(x)
        node = FlexSN(lif_core, 1, backend="triton")
        actual = node(x)
    stream.synchronize()
    assert actual.device == device
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_rejects_cpu_input_with_cached_cuda_runtime():
    node = FlexSN(lif_core, 1, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        node(torch.randn(2, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("surrogate_name", ["Sigmoid", "ATan"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_triton_hard_spikes_preserve_surrogate_gradients(
    surrogate_name, dtype, detach_reset
):
    from spikingjelly.activation_based import surrogate

    spike_function = getattr(surrogate, surrogate_name).spiking_function

    def core(x, v):
        h = v + (x - v) / 2.0
        spike = spike_function(h - 1.0, 4.0)
        reset_spike = spike.detach() if detach_reset else spike
        return spike, h * (1.0 - reset_spike)

    inputs = torch.tensor([1.5, 2.5, 0.25, 3.0], device="cuda", dtype=dtype)
    inputs = inputs[:, None].expand(4, 17)
    initial = torch.linspace(-0.2, 0.2, 17, device="cuda", dtype=dtype)
    results = []
    for backend in ("torch", "triton"):
        x = inputs.detach().requires_grad_(True)
        v = initial.detach().requires_grad_(True)
        node = FlexSN(core, 1, backend=backend, store_state_seqs=True)
        node.states = (v,)
        spikes = node(x)
        loss = spikes.float().sum() + node.state_seqs[0].float().sum()
        results.append((spikes, node.states, torch.autograd.grad(loss, (x, v))))
    tolerance = {} if dtype == torch.float32 else {"atol": 0.02, "rtol": 0.02}
    torch.testing.assert_close(results[1], results[0], **tolerance)
