import copy

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import FlexSN


def _lif_core(x, v, threshold):
    h = v + (x - v) * 0.5
    spike = torch.sigmoid(h - threshold)
    return spike, h * (1.0 - spike)


def _division_core(x, v, scale):
    h = (x + v) / 2.5
    return torch.sigmoid(h / scale), h


def _reference(x_seq, v, threshold):
    outputs, states = [], []
    for x in x_seq:
        output, v = _lif_core(x, v, threshold)
        outputs.append(output)
        states.append(v)
    return torch.stack(outputs), v, torch.stack(states)


def test_cpu_sequence_matches_reference_and_tracks_state_sequences():
    threshold = torch.nn.Parameter(torch.tensor(0.8))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True)
    x = torch.randn(4, 2, 3, requires_grad=True)
    initial = torch.randn(2, 3, requires_grad=True)
    module.states = (initial,)

    actual = module(x)
    expected, final, state_seq = _reference(x, initial, threshold)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.states[0], final)
    torch.testing.assert_close(module.state_seqs[0], state_seq)
    loss = actual.sum() + module.states[0].sum()
    reference_loss = expected.sum() + final.sum()
    got = torch.autograd.grad(loss, (x, initial, threshold))
    want = torch.autograd.grad(reference_loss, (x, initial, threshold))
    for actual_grad, expected_grad in zip(got, want, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("path", ["hop", "triton"])
def test_stored_sequence_final_state_detaches_without_aliasing(path):
    if path == "triton":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    from spikingjelly.activation_based import functional

    device = "cuda" if path == "triton" else "cpu"
    threshold = torch.nn.Parameter(torch.tensor(0.8, device=device))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True)
    x = torch.randn(4, 2, 3, device=device, requires_grad=True)
    initial = torch.randn(2, 3, device=device, requires_grad=True)
    args = ((x,), (initial,), module.static_inputs, True)
    if path == "hop":
        outputs, states, traces = module._hop_scan(*args)
    else:
        outputs, states, traces = module._triton_scan(*args, x[0])
    expected, final, trace = _reference(x, initial, threshold)
    for got, want in zip((outputs[0], states[0], traces[0]), (expected, final, trace)):
        torch.testing.assert_close(got, want)
    actual_gradients = torch.autograd.grad(
        outputs[0].sum() + states[0].sum() + traces[0].sum(),
        (x, initial, threshold),
    )
    reference_gradients = torch.autograd.grad(
        expected.sum() + final.sum() + trace.sum(), (x, initial, threshold)
    )
    for got, want in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(got, want)
    module.states = states
    functional.detach_net(module)
    assert module.states[0].grad_fn is None
    with torch.no_grad():
        module.states[0].add_(1)
    torch.testing.assert_close(traces[0], trace)


def test_functional_forward_does_not_mutate_managed_state():
    module = FlexSN(_lif_core, 1, (torch.tensor(0.8),), store_state_seqs=True)
    x = torch.randn(3, 2)
    state = torch.zeros_like(x[0])

    outputs, updated = module.functional_forward((x,), (state,))

    assert module.states == (None,)
    assert module.state_seqs is None
    torch.testing.assert_close(outputs[0], module(x))
    torch.testing.assert_close(updated[0], module.states[0])
    assert module.state_seqs is not None


@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_functional_static_defaults_follow_registration_and_preserve_overrides(
    step_mode,
):
    module = FlexSN(_lif_core, 1, (torch.tensor(0.8),), step_mode=step_mode)
    module = module.to(dtype=torch.float64)
    x = torch.randn((2,) if step_mode == "s" else (3, 2), dtype=torch.float64)
    state = torch.randn(2, dtype=torch.float64)
    registered = module.static_inputs[0]
    default = module.functional_forward((x,), (state,))
    explicit = module.functional_forward((x,), (state,), static_inputs=(registered,))
    for actual, expected in zip(
        (*default[0], *default[1]), (*explicit[0], *explicit[1])
    ):
        torch.testing.assert_close(actual, expected)

    override = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    outputs, updated = module.functional_forward(
        (x,), (state,), static_inputs=(override,)
    )
    if step_mode == "s":
        expected, final = _lif_core(x, state, override)
    else:
        expected, final, _ = _reference(x, state, override)
    torch.testing.assert_close(outputs[0], expected)
    torch.testing.assert_close(updated[0], final)
    got = torch.autograd.grad(outputs[0].sum(), override)[0]
    want = torch.autograd.grad(expected.sum(), override)[0]
    torch.testing.assert_close(got, want)
    assert module.static_inputs[0] is registered
    assert module.states == (None,)
    with pytest.raises(ValueError, match="static inputs"):
        module.functional_forward((x,), (state,), static_inputs=())
    low_level = (
        module.single_step_functional_forward
        if step_mode == "s"
        else module.multi_step_functional_forward
    )
    with pytest.raises(TypeError, match="static_inputs"):
        low_level((x,), (state,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_functional_static_defaults_compile_with_gradients(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    from spikingjelly.activation_based import base

    threshold = torch.nn.Parameter(torch.tensor(0.8, device=device))
    module = FlexSN(_lif_core, 1, (threshold,))
    function = base.to_functional_forward(module)
    x = torch.randn(3, 2, device=device, requires_grad=True)
    initial = torch.randn(2, device=device, requires_grad=True)
    expected, final, _ = _reference(x, initial, threshold)
    function((x,), (initial,))
    outputs, states = torch.compile(function, fullgraph=True)((x,), (initial,))
    torch.testing.assert_close(outputs[0], expected)
    torch.testing.assert_close(states[0], final)
    actual_gradients = torch.autograd.grad(
        outputs[0].sum() + states[0].sum(), (x, initial, threshold)
    )
    expected_gradients = torch.autograd.grad(
        expected.sum() + final.sum(), (x, initial, threshold)
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    assert module.states == (None,)


def test_cpu_scan_is_visible_to_fullgraph_compile():
    x = torch.randn(3, 2, 4)
    eager = FlexSN(_lif_core, 1, (torch.tensor(0.8),))(x)
    compiled = torch.compile(
        FlexSN(_lif_core, 1, (torch.tensor(0.8),)), fullgraph=True
    )(x)
    torch.testing.assert_close(compiled, eager)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("detach_first_output", [False, True])
def test_multi_input_output_and_state_counts(device, detach_first_output):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def core(x, y, v, w, scale):
        v_next = v + x
        w_next = w + y * scale
        output = (x + y).detach() if detach_first_output else x + y
        return output, x - y, v_next, w_next

    scale = torch.nn.Parameter(torch.tensor(2.0, device=device))
    module = FlexSN(core, 2, (scale,), store_state_seqs=True)
    x = torch.randn(3, 2, device=device, requires_grad=True)
    y = torch.randn(3, 2, device=device, requires_grad=True)
    v0 = torch.randn(2, device=device, requires_grad=True)
    w0 = torch.randn(2, device=device, requires_grad=True)
    module.states = (v0, w0)
    outputs = module(x, y)
    if device == "cuda":
        assert module._triton_handle is not None
    expected = ((x + y).detach() if detach_first_output else x + y, x - y)
    traces = (v0 + x.cumsum(0), w0 + (y * scale).cumsum(0))
    for got, want in zip(outputs, expected, strict=True):
        torch.testing.assert_close(got, want)
    for got, want in zip(module.state_seqs, traces, strict=True):
        torch.testing.assert_close(got, want)
    for got, want in zip(module.states, (t[-1] for t in traces), strict=True):
        torch.testing.assert_close(got, want)
    loss = sum(t.sum() for t in (*outputs, *module.states, *module.state_seqs))
    reference = sum(t.sum() for t in (*expected, *(t[-1] for t in traces), *traces))
    got = torch.autograd.grad(loss, (x, y, v0, w0, scale))
    want = torch.autograd.grad(reference, (x, y, v0, w0, scale))
    for actual, reference in zip(got, want, strict=True):
        torch.testing.assert_close(actual, reference)


def test_constructor_rejects_invalid_or_captured_core_values():
    with pytest.raises(ValueError, match="num_states"):
        FlexSN(_lif_core, -1)
    with pytest.raises(TypeError, match="static_inputs"):
        FlexSN(_lif_core, 1, (1.0,))

    captured = torch.tensor(1.0)

    def invalid_core(x, v):
        return x + captured, v

    with pytest.raises(TypeError, match="capture tensors"):
        FlexSN(invalid_core, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_automatically_builds_triton_for_supported_core():
    pytest.importorskip("triton")
    threshold = torch.nn.Parameter(torch.tensor(0.8, device="cuda"))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True).cuda()
    x = torch.randn(4, 2, 3, device="cuda", requires_grad=True)
    initial = torch.randn(2, 3, device="cuda", requires_grad=True)
    module.states = (initial,)

    actual = module(x)
    expected, final, state_seq = _reference(x, initial, threshold)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.states[0], final)
    torch.testing.assert_close(module.state_seqs[0], state_seq)
    torch.autograd.grad(actual.sum() + module.states[0].sum(), (x, initial, threshold))
    assert module._triton_handle is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_runtime_lifetime_survives_copy_and_rebuilds():
    pytest.importorskip("triton")
    module = FlexSN(_lif_core, 1, (torch.tensor(0.8, device="cuda"),)).cuda()
    x = torch.randn(3, 2, device="cuda")
    module(x)
    assert module._triton_handle is not None

    copied = copy.deepcopy(module)
    assert copied._triton_handle is None
    torch.testing.assert_close(copied(x), module(x))
    assert copied._triton_handle is not None


def test_triton_kernels_take_runtime_sequence_length_and_width(monkeypatch):
    pytest.importorskip("triton")
    from spikingjelly import configure
    from spikingjelly._ops.flexsn import launch
    from spikingjelly._ops.flexsn.kernel import _build_kernels
    from spikingjelly.activation_based.neuron.flexsn_trace import _trace_core

    def core(x, v):
        return _lif_core(x, v, 0.8)

    module = FlexSN(core, 1)
    wrapped = module._wrapped_core(module._num_inputs, module._num_outputs)
    examples = (torch.zeros(1), torch.zeros(1))
    graphs = _trace_core(wrapped, examples, module._num_outputs, 1)
    try:
        kernels = _build_kernels(
            wrapped.__name__, module._num_inputs, 1, module._num_outputs, *graphs
        )
    except RuntimeError as error:  # older Triton autotuners query the GPU driver
        if "active drivers" not in str(error):
            raise
        pytest.skip("Triton has no active driver on this host")
    for key in (
        "inference_kernel",
        "inference_final_state_kernel",
        "forward_kernel",
        "backward_kernel",
    ):
        fn = kernels[key].fn
        constexprs = {p.name for p in fn.params if p.is_constexpr}
        # T / NCL as constexpr recompiled the kernel for every sequence length
        # and batch size, and fully unrolled long sequences.
        assert "T" not in constexprs and "NCL" not in constexprs
        assert "tl.range(" in fn.src and "tl.static_range(" in fn.src

    monkeypatch.setattr(configure, "triton_neuron_kernel_static_range_max_T", 8)
    assert launch._t_static(8) == 8
    assert launch._t_static(9) == 0
    assert launch._t_static(1000) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_division_codegen_preserves_fx_dtype_on_cpu(dtype):
    from spikingjelly._ops.torch2triton.graph2triton import generate_triton_code_str
    from spikingjelly.activation_based.neuron.flexsn_trace import _trace_core

    examples = (torch.ones(1, dtype=dtype),) * 3
    graphs = _trace_core(_division_core, examples, num_outputs=1, num_states=1)
    for graph in graphs[:3]:
        source, _ = generate_triton_code_str(graph, "division_core")
        divisions = [
            node for node in graph.nodes if node.target == torch.ops.aten.div.Tensor
        ]
        assert divisions
        for node in divisions:
            assert node.meta["val"].dtype == dtype
            lhs = node.args[0].name
            rhs = node.args[1]
            rhs = rhs.name if isinstance(rhs, torch.fx.Node) else str(rhs)
            expression = f"{lhs} / {rhs}"
            if dtype != torch.float32:
                tl_dtype = "float16" if dtype == torch.float16 else "bfloat16"
                expression = f"({expression}).to(tl.{tl_dtype})"
            assert f"    {node.name} = {expression}\n" in source
        assert all(node.target != torch.ops.aten.detach.default for node in graph.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("T", [5, 37])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("store_state_seqs", [False, True])
def test_cuda_low_precision_division_matches_torch(
    monkeypatch, dtype, T, training, store_state_seqs
):
    pytest.importorskip("triton")
    from spikingjelly import configure
    from spikingjelly._ops.flexsn.launch import _t_static

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 required")
    monkeypatch.setattr(configure, "triton_neuron_kernel_static_range_max_T", 16)
    assert _t_static(T) == (T if T <= 16 else 0)
    torch.manual_seed(0)
    scale = torch.nn.Parameter(torch.tensor(1.25, device="cuda", dtype=dtype))
    module = FlexSN(_division_core, 1, (scale,), store_state_seqs=store_state_seqs)
    module.train(training)
    x = torch.randn(T, 2, 65, device="cuda", dtype=dtype, requires_grad=training)
    initial = torch.randn(2, 65, device="cuda", dtype=dtype, requires_grad=training)
    module.states = (initial,)
    tolerance = (
        {"rtol": 1e-2, "atol": 2e-3}
        if dtype == torch.float16
        else {"rtol": 4e-2, "atol": 1e-2}
    )

    with torch.set_grad_enabled(training):
        actual = module(x)
        outputs, states = [], []
        final = initial
        for step in x:
            output, final = _division_core(step, final, scale)
            outputs.append(output)
            states.append(final)
        expected = torch.stack(outputs)
        assert module._triton_handle is not None
        assert module._triton_capability[x.device, dtype] is True
        actual_values = (actual, module.states[0])
        expected_values = (expected, final)
        if store_state_seqs:
            actual_values += (module.state_seqs[0],)
            expected_values += (torch.stack(states),)
        else:
            assert module.state_seqs is None
        for got, want in zip(actual_values, expected_values, strict=True):
            assert got.dtype == want.dtype == dtype
            torch.testing.assert_close(got, want, **tolerance)
        if training:
            inputs = (x, initial, scale)
            got = torch.autograd.grad(sum(t.sum() for t in actual_values), inputs)
            want = torch.autograd.grad(sum(t.sum() for t in expected_values), inputs)
            for actual_grad, expected_grad in zip(got, want, strict=True):
                assert actual_grad.dtype == expected_grad.dtype == dtype
                torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("T", [5, 37])
def test_cuda_runtime_time_loop_matches_reference(monkeypatch, T):
    pytest.importorskip("triton")
    from spikingjelly import configure

    monkeypatch.setattr(configure, "triton_neuron_kernel_static_range_max_T", 16)
    threshold = torch.nn.Parameter(torch.tensor(0.8, device="cuda"))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True).cuda()
    for batch in (2, 7):
        x = torch.randn(T, batch, 3, device="cuda", requires_grad=True)
        initial = torch.randn(batch, 3, device="cuda", requires_grad=True)
        module.states = (initial,)
        actual = module(x)
        expected, final, state_seq = _reference(x, initial, threshold)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(module.states[0], final)
        torch.testing.assert_close(module.state_seqs[0], state_seq)
        inputs = (x, initial, threshold)
        got = torch.autograd.grad(actual.sum() + module.states[0].sum(), inputs)
        want = torch.autograd.grad(expected.sum() + final.sum(), inputs)
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w)
    assert module._triton_handle is not None


def test_nondifferentiable_core_reports_unsupported_training_graph():
    from spikingjelly.activation_based.neuron.flexsn_trace import _trace_core

    def core(x):
        return ((x > 0).float(),)

    with pytest.raises(NotImplementedError, match="No differentiable Tensor"):
        _trace_core(core, (torch.zeros(1),), num_outputs=1, num_states=0)


def test_core_tracing_isolates_outer_saved_tensor_hooks_and_rng():
    from spikingjelly.activation_based.neuron.flexsn_trace import _trace_core

    saved = []

    def pack(tensor):
        saved.append(tensor.detach())
        return tensor.detach()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        rng_state = torch.random.get_rng_state()
        _trace_core(_lif_core, (torch.zeros(1),) * 3, num_outputs=1, num_states=1)
        assert not saved
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        x = torch.randn(2, requires_grad=True)
        loss = x.square().sum()
        assert saved
    loss.backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_nondifferentiable_core_automatically_uses_hop():
    def core(x):
        return (x > 0).float()

    module = FlexSN(core, num_states=0).cuda()
    x = torch.tensor([[-1.0, 1.0], [2.0, -2.0]], device="cuda", requires_grad=True)
    for values in (x, -x):
        actual = module(values)
        torch.testing.assert_close(actual, core(values))
        assert actual.requires_grad is False
        assert module.states == ()
    assert module._triton_capability[x.device, x.dtype] is False
    compiled = torch.compile(module, fullgraph=True)
    torch.testing.assert_close(compiled(x), core(x))
