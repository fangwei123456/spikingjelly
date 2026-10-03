"""Layout changes must not change logical neuron identity or temporal state."""

import copy
import importlib
import itertools
import math
import sys

import pytest
import torch

from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based._neuron_layout import _empty_like, _layout_args


@pytest.mark.parametrize("flag,value", [("--steps", "0"), ("--warmup", "-1")])
def test_layout_benchmark_rejects_invalid_iterations(monkeypatch, capsys, flag, value):
    from benchmark.benchmark_neuron_layout import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "triton",
            "--layout",
            "contiguous",
            "--output",
            "unused.json",
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--steps must be positive" in capsys.readouterr().err


def test_cupy_codegen_reports_unsupported_array_parameter():
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.strides import (
        _strided_code,
    )

    with pytest.raises(ValueError, match=r"probe: expected.* x_seq"):
        _strided_code(
            "void probe(const double* x_seq) {}",
            "probe",
            (3,),
            (("x_seq", (6, 2), False, True, True),),
        )


def _layout(x, kind):
    if kind == "contiguous":
        return x.contiguous()
    if kind == "time_inner":
        order = (*range(1, x.ndim), 0)
        return x.permute(order).contiguous().permute(x.ndim - 1, *range(x.ndim - 1))
    if kind == "channels_last":
        order = (0, 1, *range(3, x.ndim), 2) if x.ndim == 5 else (*range(1, x.ndim), 0)
        inverse = tuple(order.index(d) for d in range(x.ndim))
        return x.permute(order).contiguous().permute(inverse)
    if kind == "sliced":
        base = x.new_empty((*x.shape[:-1], x.shape[-1] * 2 + 1))
        result = base[..., 1::2]
        result.copy_(x)
        return result
    if kind == "offset":
        base = x.new_empty(x.numel() + 1)
        result = base[1:].view(x.shape)
        result.copy_(x)
        return result
    if kind == "broadcast":
        return x[:1].expand_as(x)
    if kind == "overlap":
        return x.contiguous().view(-1).as_strided(x.shape, (1,) * x.ndim)
    raise ValueError(kind)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_layout_offsets_and_dense_allocation(order):
    x = torch.empty(2, 3, 5).permute(order)
    output = _empty_like(x)
    assert output.stride() == x.stride()
    state = torch.empty_like(x[0]).transpose(0, 1).contiguous().transpose(0, 1)
    sizes, layouts = _layout_args(x, x, state)
    for index in range(math.prod(sizes)):
        coordinates = []
        n = index
        for size in sizes:
            coordinates.append(n % size)
            n //= size
        for time in range(x.shape[0]):
            offset = time * layouts[0][0] + sum(
                c * s for c, s in zip(coordinates, layouts[0][1:])
            )
            assert 0 <= offset < x.numel()
    assert layouts[1][0] == 0


@pytest.mark.parametrize("kind", ["sliced", "offset", "broadcast", "overlap"])
def test_nondense_output_does_not_alias(kind):
    x = _layout(torch.arange(30.0).view(2, 3, 5), kind)
    result = _empty_like(x)
    result.copy_(torch.arange(30.0).view_as(x))
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result.flatten(), torch.arange(30.0))


@pytest.mark.parametrize(
    "input_strides,output_strides",
    [
        ((1, 10, 2), (1, 15, 3)),  # Dense, with time physically innermost.
        ((1, 20, 4), (15, 5, 1)),  # Gaps: compact space, then time.
        ((0, 0, 1), (15, 1, 3)),
        ((1, 1, 1), (15, 1, 3)),  # Equal strides retain logical axis order.
    ],
)
def test_resized_neuron_buffer_layout(input_strides, output_strides):
    x = torch.empty_strided((2, 3, 5), input_strides)
    output = _empty_like(x, shape=(3, 3, 5), dtype=torch.float16)
    assert output.shape == (3, 3, 5)
    assert output.stride() == output_strides
    assert output.dtype == torch.float16
    values = torch.arange(output.numel(), dtype=output.dtype).view_as(output)
    output.copy_(values)
    assert torch.equal(output, values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "time_inner", "sliced", "broadcast"])
@pytest.mark.parametrize("width", [7, 8])
def test_cupy_state_sequence_alignment(dtype, layout, width):
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step.base import (
        _aligned_v_v_seq,
    )

    x = _layout(torch.empty(4, 3, width, dtype=dtype), layout)
    output = _aligned_v_v_seq(x)
    assert output.shape == (5, 3, width)
    assert output[1:].data_ptr() % 16 == 0
    values = torch.arange(output.numel(), dtype=dtype).view_as(output)
    output.copy_(values)
    assert torch.equal(output, values)
    from torch.fx.experimental.proxy_tensor import make_fx

    traced = make_fx(_aligned_v_v_seq, tracing_mode="symbolic")(x)
    assert traced(x).stride() == output.stride()


@pytest.mark.parametrize("sequence", [False, True])
def test_layout_helpers_symbolic_trace(sequence):
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(x, state):
        shape = (x.shape[0] + 1, *x.shape[1:]) if sequence else None
        return (
            _empty_like(x, shape=shape, sequence=sequence),
            _layout_args(x, x, state, sequence=sequence),
        )

    for batch in (2, 4):
        shape = (batch, 3, 5) if sequence else (3, 5)
        strides = (1, 20, 4) if sequence else (20, 4)
        x = torch.empty_strided(shape, strides)
        state = torch.empty_strided((3, 5), (1, 6))
        expected, metadata = run(x, state)
        traced = make_fx(run, tracing_mode="symbolic")(x, state)
        actual, actual_metadata = traced(x, state)
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual_metadata == metadata
        assert metadata == ((5, 3), ((1 if sequence else 0, 4, 20), (0, 6, 1)))


_CASES = [
    ("triton", "IFNode"),
    ("triton", "LIFNode"),
    ("triton", "ParametricLIFNode"),
    ("triton", "ILIFNode"),
    ("cupy", "IFNode"),
    ("cupy", "LIFNode"),
    ("cupy", "ParametricLIFNode"),
    ("cupy", "QIFNode"),
    ("cupy", "EIFNode"),
    ("cupy", "IzhikevichNode"),
]


def _broadcast_source(shape, kind, dtype):
    source_shape = {
        "time": (1, *shape[1:]),
        "space": (*shape[:2], 1, *shape[3:]),
        "scalar": (1,) * len(shape),
    }[kind]
    source = _layout(
        torch.rand(source_shape, device="cuda", dtype=dtype) * 0.8,
        "channels_last",
    ).requires_grad_()
    return source, source.expand(shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend,kind", _CASES)
@pytest.mark.parametrize("broadcast", ["time", "space", "scalar"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_broadcast_gradients_reach_sources(backend, kind, broadcast, dtype):
    pytest.importorskip(backend)
    if kind == "IzhikevichNode" and dtype == torch.float16:
        pytest.skip("Izhikevich CuPy supports FP32 only")
    torch.manual_seed(29)
    node = (
        getattr(neuron, kind)(
            backend=backend, step_mode="m", store_v_seq=True, v_threshold=0.6
        )
        .cuda()
        .to(dtype)
    )
    reference = copy.deepcopy(node)
    width = 4 if kind == "ParametricLIFNode" else 5
    shape = (4, 1, 3, 1, width)
    source, x = _broadcast_source(shape, broadcast, dtype)
    source_ref = source.detach().clone().requires_grad_()
    xr = source_ref.expand(shape).clone(memory_format=torch.contiguous_format)
    sources, sources_ref = [source], [source_ref]
    for name in ("v", "w") if kind == "IzhikevichNode" else ("v",):
        state_source = (
            torch.rand(1, 1, 1, width, device="cuda", dtype=dtype) * 0.1
        ).requires_grad_()
        state_ref = state_source.detach().clone().requires_grad_()
        setattr(node, name, state_source.expand(shape[1:]))
        setattr(reference, name, state_ref.expand(shape[1:]).clone())
        sources.append(state_source)
        sources_ref.append(state_ref)
    snapshots = [value.detach().clone() for value in sources]
    actual, expected = node(x), reference(xr)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    torch.testing.assert_close(node.v, reference.v, rtol=0, atol=0)
    assert torch.ops.aten.is_non_overlapping_and_dense.default(actual)
    assert actual.data_ptr() != x.data_ptr()
    grad_s = torch.tensor(0.25, device="cuda", dtype=dtype).expand_as(actual)
    grad_v = _layout(torch.rand_like(node.v_seq), "time_inner")
    outputs, outputs_ref = (actual, node.v_seq), (expected, reference.v_seq)
    upstream = (grad_s, grad_v)
    if kind == "IzhikevichNode":
        torch.testing.assert_close(node.w, reference.w, rtol=0, atol=0)
        outputs, outputs_ref = (*outputs, node.w), (*outputs_ref, reference.w)
        upstream = (*upstream, torch.ones_like(node.w))
    grads = torch.autograd.grad(outputs, (*sources, *node.parameters()), upstream)
    grads_ref = torch.autograd.grad(
        outputs_ref,
        (*sources_ref, *reference.parameters()),
        tuple(value.contiguous() for value in upstream),
    )
    for value, expected_grad in zip(grads, grads_ref):
        assert torch.isfinite(value).all() and torch.isfinite(expected_grad).all()
        torch.testing.assert_close(
            value,
            expected_grad,
            rtol=1e-2 if dtype == torch.float16 else 1e-5,
            atol=1e-2 if dtype == torch.float16 else 1e-6,
        )
    for value, snapshot in zip(sources, snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    functional.detach_net(node)
    functional.detach_net(reference)
    with torch.no_grad():
        torch.testing.assert_close(node(x), reference(xr), rtol=0, atol=0)
        node.reset()
        reference.reset()
        torch.testing.assert_close(node(x), reference(xr), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cupy"])
@pytest.mark.parametrize("broadcast", ["time", "space", "scalar"])
def test_compiled_broadcast_lif_source_gradients(backend, broadcast):
    from spikingjelly.activation_based import surrogate

    pytest.importorskip(backend)
    op = getattr(functional, f"lif_multi_step_{backend}")
    sg = surrogate.Sigmoid()
    shape = (4, 2, 3, 2, 5)
    source, _ = _broadcast_source(shape, broadcast, torch.float32)
    state = torch.rand(1, 3, 1, 5, device="cuda", requires_grad=True)

    def run(source, state):
        spike, final, sequence = op(
            source.expand(shape),
            state.expand(shape[1:]),
            2.0,
            True,
            1.0,
            0.0,
            sg,
            store_v_seq=True,
        )
        return spike, final, sequence

    eager = run(source, state)
    compiled = torch.compile(run, fullgraph=True)(source, state)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.stride() == b.stride()
    grads = torch.autograd.grad(sum(x.sum() for x in compiled), (source, state))
    grads_ref = torch.autograd.grad(sum(x.sum() for x in eager), (source, state))
    for a, b in zip(grads, grads_ref):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    reference = neuron.LIFNode(step_mode="m", store_v_seq=True, backend="torch")
    reference.v = state.expand(shape[1:])
    spike_ref = reference(source.expand(shape))
    torch.testing.assert_close(compiled[0], spike_ref, rtol=0, atol=0)
    torch.testing.assert_close(compiled[2], reference.v_seq)
    torch_grads = torch.autograd.grad(
        spike_ref.sum() + reference.v.sum() + reference.v_seq.sum(), (source, state)
    )
    for a, b in zip(grads, torch_grads):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "kind,dtype", [("LIFNode", torch.float16), ("IzhikevichNode", torch.float32)]
)
def test_cupy_multiblock_initial_states(kind, dtype):
    pytest.importorskip("cupy")
    from spikingjelly import configure

    torch.manual_seed(23)
    x = _layout(
        torch.rand(4, 3, 2 * configure.cuda_threads + 1, device="cuda", dtype=dtype)
        * 0.8,
        "sliced",
    )
    node = getattr(neuron, kind)(backend="cupy", step_mode="m", store_v_seq=True)
    reference = copy.deepcopy(node)
    reference.backend = "torch"
    for state in ("v", "w") if kind == "IzhikevichNode" else ("v",):
        value = _layout(torch.rand_like(x[0]) * 0.1, "sliced")
        setattr(node, state, value)
        setattr(reference, state, value.contiguous())
    with torch.no_grad():
        torch.testing.assert_close(node(x), reference(x.contiguous()), rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq)
    if kind == "IzhikevichNode":
        torch.testing.assert_close(node.w, reference.w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend,kind", _CASES)
@pytest.mark.parametrize(
    "layout",
    ["channels_last", "time_inner", "sliced", "offset", "broadcast", "overlap"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_strided_neuron_forward_backward(backend, kind, layout, dtype):
    pytest.importorskip(backend)
    if backend == "cupy" and dtype == torch.bfloat16:
        pytest.skip("CuPy point neurons do not support BF16")
    if kind == "IzhikevichNode" and dtype == torch.float16:
        pytest.skip("Izhikevich CuPy supports FP32 only")
    torch.manual_seed(20261001)
    node = (
        getattr(neuron, kind)(backend=backend, step_mode="m", store_v_seq=True)
        .cuda()
        .to(dtype)
    )
    reference = copy.deepcopy(node)
    # Odd neuron count exercises masked half2 tails. PLIF's existing even-count
    # restriction is intentionally retained.
    width = 4 if kind == "ParametricLIFNode" else 5
    x = (
        _layout(torch.rand(4, 1, 3, 1, width, device="cuda", dtype=dtype) * 0.8, layout)
        .detach()
        .requires_grad_()
    )
    xr = x.detach().contiguous().requires_grad_()
    v = _layout(torch.rand_like(x[0]) * 0.1, "sliced").detach().requires_grad_()
    vr = v.detach().contiguous().requires_grad_()
    node.v, reference.v = v, vr
    if kind == "IzhikevichNode":
        w = _layout(torch.rand_like(v) * 0.1, "time_inner").detach().requires_grad_()
        wr = w.detach().contiguous().requires_grad_()
        node.w, reference.w = w, wr
    s, sr = node(x), reference(xr)
    if layout in ("channels_last", "time_inner", "offset"):
        assert all(
            a == b for size, a, b in zip(x.shape, s.stride(), x.stride()) if size != 1
        )
    torch.testing.assert_close(s, sr, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    grad_s = _layout(torch.rand_like(s), "time_inner")
    grad_v = torch.ones((), device="cuda", dtype=dtype).expand_as(node.v_seq)
    inputs = (x, v, *node.parameters())
    inputs_r = (xr, vr, *reference.parameters())
    if kind == "IzhikevichNode":
        torch.testing.assert_close(node.w, reference.w, rtol=0, atol=0)
        inputs, inputs_r = (*inputs, w), (*inputs_r, wr)
    grads = torch.autograd.grad((s, node.v_seq), inputs, (grad_s, grad_v))
    grads_r = torch.autograd.grad(
        (sr, reference.v_seq), inputs_r, (grad_s.contiguous(), grad_v.contiguous())
    )
    for a, b in zip(grads, grads_r):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(
            a,
            b,
            rtol=1e-2 if dtype != torch.float32 else 1e-5,
            atol=1e-2 if dtype != torch.float32 else 1e-6,
        )
    functional.detach_net(node)
    functional.detach_net(reference)
    torch.testing.assert_close(node(x.detach()), reference(xr.detach()), rtol=0, atol=0)
    node.reset()
    reference.reset()
    torch.testing.assert_close(node(x.detach()), reference(xr.detach()), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cupy"])
@pytest.mark.parametrize(
    "layout", ["channels_last", "time_inner", "sliced", "broadcast"]
)
def test_compiled_lif_layout_and_gradients(backend, layout):
    from spikingjelly.activation_based import surrogate

    pytest.importorskip(backend)
    operation = getattr(functional, f"lif_multi_step_{backend}")
    sg = surrogate.Sigmoid()

    def run(x, v):
        return operation(x, v, 2.0, True, 1.0, 0.0, sg, store_v_seq=True)

    torch.manual_seed(19)
    x = _layout(torch.rand(4, 2, 3, 2, 5, device="cuda"), layout).requires_grad_()
    v = _layout(torch.rand_like(x[0]), "sliced").requires_grad_()
    eager = run(x, v)
    compiled = torch.compile(run, fullgraph=True)(x, v)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.stride() == b.stride()
    ga = torch.autograd.grad(compiled[0].sum() + compiled[2].sum(), (x, v))
    gb = torch.autograd.grad(eager[0].sum() + eager[2].sum(), (x, v))
    for a, b in zip(ga, gb):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "broadcast"])
@pytest.mark.parametrize("time_steps", [1, 4])
def test_compiled_cupy_plif_parameter_gradient(layout, time_steps):
    pytest.importorskip("cupy")
    torch.manual_seed(753)
    node = neuron.ParametricLIFNode(
        backend="cupy", step_mode="m", store_v_seq=True
    ).cuda()
    reference = copy.deepcopy(node)
    shape = (time_steps, 2, 3, 2, 8)
    source_shape = (1, *shape[1:]) if layout == "broadcast" else shape
    source = torch.rand(source_shape, device="cuda") * 0.8
    if layout == "channels_last":
        source = _layout(source, layout)
    source.requires_grad_()
    source_ref = source.detach().clone().requires_grad_()
    x = source.expand(shape) if layout == "broadcast" else source
    xr = source_ref.expand(shape) if layout == "broadcast" else source_ref
    expected = reference(xr)
    actual = torch.compile(node, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    loss = actual.sum() + node.v_seq.sum()
    expected_loss = expected.sum() + reference.v_seq.sum()
    functional.reset_net(node)
    functional.reset_net(reference)
    gradients = torch.autograd.grad(loss, (source, node.w))
    reference_gradients = torch.autograd.grad(expected_loss, (source_ref, reference.w))
    assert reference_gradients[1].abs().item() > 0
    for a, b in zip(gradients, reference_gradients):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["cupy"])
@pytest.mark.parametrize("kind", ["IFNode", "LIFNode"])
@pytest.mark.parametrize("layout", ["sliced", "broadcast"])
def test_single_step_view_neuron(backend, kind, layout):
    pytest.importorskip(backend)
    torch.manual_seed(21)
    node = getattr(neuron, kind)(backend=backend, step_mode="s").cuda()
    reference = copy.deepcopy(node)
    source = torch.rand(2, 3, 2, 5, device="cuda", requires_grad=True)
    source_ref = source.detach().clone().requires_grad_()
    x = _layout(source, layout)
    xr = _layout(source_ref, layout).contiguous()
    a, b = node(x), reference(xr)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a.sum(), source)[0]
    gb = torch.autograd.grad(b.sum(), source_ref)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "layout", ["channels_last", "time_inner", "sliced", "broadcast"]
)
def test_strided_inference_point_neurons(layout):
    pytest.importorskip("triton")
    from spikingjelly.activation_based.triton_kernel.neuron_kernel.activation_aware_if import (
        _multistep_activation_aware_if,
    )
    from spikingjelly.activation_based.triton_kernel.neuron_kernel.stbif import (
        multi_step_stbif,
        single_step_stbif,
    )

    torch.manual_seed(23)
    x = _layout(torch.rand(4, 2, 3, 2, 5, device="cuda"), layout)
    q = _layout(torch.rand_like(x[0]), "sliced")
    acc = _layout(torch.zeros_like(x[0]), "channels_last")
    threshold = torch.tensor(1.0, device="cuda")
    positive = torch.tensor(8.0, device="cuda")
    negative = torch.tensor(0.0, device="cuda")
    for op, data in [(multi_step_stbif, x), (single_step_stbif, x[0])]:
        out = op(data, q, acc, threshold, positive, negative)
        ref = op(
            data.contiguous(),
            q.contiguous(),
            acc.contiguous(),
            threshold,
            positive,
            negative,
        )
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    threshold = torch.tensor([0.6, 9.0, 0.8, 9.0, 1.2, 9.0], device="cuda")[::2]
    offset = torch.tensor([-0.1, 9.0, 0.0, 9.0, 0.1, 9.0], device="cuda")[::2]
    kwargs = dict(channel_size=3, inner_size=10, v_reset=0.0, store_v_seq=True)
    out = _multistep_activation_aware_if(x, q, threshold, offset, **kwargs)
    ref = _multistep_activation_aware_if(
        x.contiguous(),
        q.contiguous(),
        threshold.contiguous(),
        offset.contiguous(),
        **kwargs,
    )
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_single_step_stbif_compacts_all_spatial_axes(device):
    from spikingjelly.activation_based.triton_kernel.neuron_kernel.stbif import (
        _single_step_stbif_fake,
        single_step_stbif,
    )

    torch.manual_seed(29)
    x = torch.empty_strided((4, 2, 3), (6, 24, 2), device=device).uniform_(-0.5, 1.5)
    q = torch.zeros_like(x)
    threshold = torch.tensor(1.0, device=device)
    positive = torch.tensor(8.0, device=device)
    negative = torch.tensor(0.0, device=device)
    fake = _single_step_stbif_fake(x, q, q, threshold, positive, negative)
    assert fake[0].stride() == (3, 12, 1)
    if device == "cuda":
        pytest.importorskip("triton")
        reference = single_step_stbif(
            x.contiguous(),
            q.contiguous(),
            q.contiguous(),
            threshold,
            positive,
            negative,
        )
        for operation in (
            single_step_stbif,
            torch.compile(single_step_stbif, fullgraph=True),
        ):
            actual = operation(x, q, q, threshold, positive, negative)
            for result, expected, declaration in zip(actual, reference, fake):
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                assert result.stride() == declaration.stride()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend,kind", _CASES)
@pytest.mark.parametrize(
    "layout", ["sliced", "channels_last", "time", "space", "scalar"]
)
def test_neuron_launch_receives_original_strided_input(
    backend, kind, layout, monkeypatch
):
    pytest.importorskip(backend)
    shape = (4, 2, 3, 2, 5)
    x = (
        _broadcast_source(shape, layout, torch.float32)[1]
        if layout in ("time", "space", "scalar")
        else _layout(torch.rand(shape, device="cuda"), layout)
    )
    launches = []
    states = []
    gradients = []
    saved = []
    restored = []
    if backend == "triton":
        module_name = {
            "IFNode": "integrate_and_fire",
            "LIFNode": "lif",
            "ParametricLIFNode": "plif",
            "ILIFNode": "ilif",
        }[kind]
        module = importlib.import_module(
            f"spikingjelly.activation_based.triton_kernel.neuron_kernel.{module_name}"
        )
        original = module.wrap_triton

        class RecordLaunch:
            def __init__(self, kernel, backward):
                self.kernel = kernel
                self.backward = backward

            def __getitem__(self, grid):
                launch = self.kernel[grid]

                def record(*args, **kwargs):
                    if self.backward:
                        gradients.append(tuple(x.data_ptr() for x in args[:2]))
                        restored.append(args[2].data_ptr())
                    else:
                        launches.append((args[0].data_ptr(), args[0].stride()))
                        states.append((args[1].data_ptr(),))
                        saved.append(args[3].data_ptr())
                    return launch(*args, **kwargs)

                return record

        monkeypatch.setattr(
            module,
            "wrap_triton",
            lambda kernel: RecordLaunch(
                original(kernel), "backward" in kernel.fn.__name__
            ),
        )
    else:
        from spikingjelly.activation_based.cuda_kernel.neuron_kernel import strides

        original = strides._get_raw_kernel

        def record_kernel(code, name, *options):
            kernel = original(code, name, *options)
            names = [
                n.removeprefix("raw_") for n in strides._parameter_names(code, name)
            ]

            def launch(grid, block, arguments, *args, **kwargs):
                values = dict(zip(names, arguments))
                if "grad_spike_seq" in values:
                    gradients.append((values["grad_spike_seq"], values["grad_v_seq"]))
                    restored.append(values["h_seq"])
                else:
                    launches.append((values["x_seq"], x.stride()))
                    states.append(
                        tuple(values[n] for n in ("v_init", "w_init") if n in values)
                    )
                    saved.append(values["h_seq"])
                return kernel(grid, block, arguments, *args, **kwargs)

            return launch

        monkeypatch.setattr(strides, "_get_raw_kernel", record_kernel)
    node = getattr(neuron, kind)(
        backend=backend, step_mode="m", store_v_seq=True
    ).cuda()
    v_source = torch.rand(1, 3, 1, 5, device="cuda", requires_grad=True)
    v = v_source.expand(shape[1:])
    node.v = v
    initial_states = [v]
    if kind == "IzhikevichNode":
        node.w = torch.rand_like(v_source).expand(shape[1:])
        initial_states.append(node.w)
    spike = node(x)
    assert launches == [(x.data_ptr(), x.stride())]
    assert states == [tuple(t.data_ptr() for t in initial_states)]
    gs = torch.tensor(0.25, device="cuda").expand_as(spike)
    gv = torch.tensor(0.5, device="cuda").expand_as(node.v_seq)
    torch.autograd.grad((spike, node.v_seq), v_source, (gs, gv))
    assert gradients == [(gs.data_ptr(), gv.data_ptr())]
    assert restored == saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cupy_half2_unaligned_external_storage():
    pytest.importorskip("cupy")
    values = torch.rand(4, 3, 5, device="cuda", dtype=torch.float16)
    x = torch.utils.dlpack.from_dlpack(_layout(values, "offset"))
    assert x.storage_offset() == 0 and x.data_ptr() % 4 == 2
    node = neuron.LIFNode(backend="cupy", step_mode="m", store_v_seq=True).cuda().half()
    reference = copy.deepcopy(node)
    torch.testing.assert_close(
        node(x), reference(x.contiguous().clone()), rtol=0, atol=0
    )
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cupy_large_address_offset_with_half2_tail():
    pytest.importorskip("cupy")
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("the strided input requires just over 4 GiB of storage")
    values = torch.tensor(
        [[0.5, 0.75, 1.25], [1.5, 0.25, 0.5]], device="cuda", dtype=torch.float16
    )
    x = torch.empty_strided(
        values.shape, ((1 << 31) + 1, 1), device="cuda", dtype=values.dtype
    )
    x.copy_(values).requires_grad_()
    expected_x = values.clone().requires_grad_()
    node = neuron.LIFNode(backend="cupy", step_mode="m", store_v_seq=True).cuda()
    reference = copy.deepcopy(node)
    actual, expected = node(x), reference(expected_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.sum(), expected_x)[0]
    assert torch.isfinite(actual_grad).all() and torch.isfinite(expected_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cupy"])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_all_small_dense_permutations(backend, order):
    pytest.importorskip(backend)
    torch.manual_seed(29)
    values = torch.rand(3, 2, 5, device="cuda") * 1.9 + 0.03
    inverse = tuple(order.index(d) for d in range(3))
    x = values.permute(order).contiguous().permute(inverse).requires_grad_()
    xr = values.clone().requires_grad_()
    node = neuron.LIFNode(backend=backend, step_mode="m", store_v_seq=True).cuda()
    reference = copy.deepcopy(node)
    state = torch.rand(5, 2, device="cuda").t().requires_grad_()
    state_r = state.detach().contiguous().requires_grad_()
    node.v, reference.v = state, state_r
    a, b = node(x), reference(xr)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    grad = torch.rand(5, 3, 2, device="cuda").permute(1, 2, 0)
    ga = torch.autograd.grad(a, (x, state), grad)
    gb = torch.autograd.grad(b, (xr, state_r), grad.contiguous())
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cupy"])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last"])
def test_compiled_convolution_chain_preserves_neuron_identity(backend, layout):
    from spikingjelly.activation_based import layer

    pytest.importorskip(backend)
    model = (
        torch.nn.Sequential(
            layer.Conv2d(64, 64, 3, padding=1, bias=False, step_mode="m"),
            layer.BatchNorm2d(64, step_mode="m"),
            neuron.LIFNode(backend=backend, step_mode="m"),
            layer.Conv2d(64, 64, 1, bias=False, step_mode="m"),
        )
        .cuda()
        .half()
    )
    model[1].eval()
    with torch.no_grad():
        torch.nn.init.dirac_(model[0].weight)
        torch.nn.init.dirac_(model[3].weight)
    reference = copy.deepcopy(model)
    shape = (4, 16, 64, 28, 28)
    values = (torch.arange(math.prod(shape), device="cuda") % 8).reshape(shape)
    x = _layout(values.half() * 0.25 + 0.03125, layout).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    state = torch.full(shape[1:], 0.0078125, device="cuda", dtype=torch.float16)
    state[:, ::2] += 0.0625
    state.requires_grad_()
    state_r = state.detach().clone().requires_grad_()
    reference[2].v = state_r
    expected = reference(xr)

    def run(data, voltage):
        model[2].v = voltage
        return model(data)

    actual = torch.compile(run, fullgraph=True)(x, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.full_like(actual, 1.0 / 256)
    ga = torch.autograd.grad(actual, (x, state), upstream)
    gb = torch.autograd.grad(expected, (xr, state_r), upstream)
    for left, right in zip(ga, gb):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=1e-2, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["IFNode", "LIFNode", "ParametricLIFNode", "ILIFNode"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_channels_last_neuron_gradients(kind, dtype):
    pytest.importorskip("triton")
    torch.manual_seed(42)
    node = getattr(neuron, kind)(backend="triton", step_mode="m", store_v_seq=True)
    node = node.cuda().to(dtype)
    reference = copy.deepcopy(node)
    # 48 channels exercise a non-power-of-two spatial extent without padding.
    x = _layout(torch.rand(4, 2, 48, 3, 5, device="cuda", dtype=dtype), "channels_last")
    x.requires_grad_()
    xr = x.detach().contiguous().requires_grad_()
    v = torch.rand_like(x[0]).requires_grad_()
    vr = v.detach().contiguous().requires_grad_()
    node.v, reference.v = v, vr
    spike, expected = node(x), reference(xr)
    torch.testing.assert_close(spike, expected, rtol=0, atol=0)
    torch.testing.assert_close(node.v_seq, reference.v_seq, rtol=0, atol=0)
    assert spike.stride() == x.stride()
    grad_s = torch.rand_like(spike)
    grad_v = torch.rand_like(node.v_seq)
    actual = torch.autograd.grad(
        (spike, node.v_seq), (x, v, *node.parameters()), (grad_s, grad_v)
    )
    expected = torch.autograd.grad(
        (expected, reference.v_seq),
        (xr, vr, *reference.parameters()),
        (grad_s.contiguous(), grad_v.contiguous()),
    )
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(
            a,
            b,
            rtol=1e-2 if dtype != torch.float32 else 1e-5,
            atol=1e-3 if dtype != torch.float32 else 1e-6,
        )
