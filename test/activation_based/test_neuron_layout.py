import importlib
import inspect
import itertools
import math

import pytest
import torch

from spikingjelly._ops.layout import _empty_like, _layout_args
from spikingjelly.activation_based import neuron

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    from spikingjelly._ops.triton_layout import _neuron_indices

    @triton.jit
    def _large_index_probe(out, N: tl.constexpr, MINOR: tl.constexpr):
        if tl.program_id(0) == tl.cdiv(N, 256) - 1:
            SIZES: tl.constexpr = () if MINOR == 1 else (16, N // 16)
            indices, mask = _neuron_indices(N, 256, SIZES, MINOR)
            lane = tl.arange(0, 256)
            tl.store(out + lane, tl.reshape(indices, (256,)))
            tl.store(out + 256 + lane, tl.reshape(mask, (256,)).to(tl.int64))


def _layout(x, kind):
    if kind == "contiguous":
        return x.contiguous()
    if kind == "time_inner":
        order = (*range(1, x.ndim), 0)
        return x.permute(order).contiguous().permute(x.ndim - 1, *range(x.ndim - 1))
    if kind == "channels_last":
        order = (0, 1, *range(3, x.ndim), 2)
        inverse = tuple(order.index(d) for d in range(x.ndim))
        return x.permute(order).contiguous().permute(inverse)
    if kind in ("batch_major_channels_last", "channels_last_time_inner"):
        order = (
            (1, 0, *range(3, x.ndim), 2)
            if kind == "batch_major_channels_last"
            else (1, *range(3, x.ndim), 2, 0)
        )
        inverse = tuple(order.index(d) for d in range(x.ndim))
        return x.permute(order).contiguous().permute(inverse)
    if kind == "singleton_strides":
        assert x.shape[1] == 1
        strides = list(x.stride())
        strides[1] = 0
        return x.as_strided(x.shape, strides)
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
        return x.reshape(-1).as_strided(x.shape, (1,) * x.ndim)
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


@pytest.mark.parametrize("kind", ["sliced", "offset", "broadcast"])
def test_nondense_output_does_not_alias(kind):
    x = _layout(torch.arange(30.0).view(2, 3, 5), kind)
    result = _empty_like(x)
    result.copy_(torch.arange(30.0).view_as(x))
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result.flatten(), torch.arange(30.0))


@pytest.mark.parametrize(
    "input_strides,output_strides",
    [
        ((1, 10, 2), (1, 15, 3)),
        ((1, 20, 4), (15, 5, 1)),
        ((0, 0, 1), (15, 1, 3)),
        ((1, 1, 1), (15, 1, 3)),
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


@pytest.mark.parametrize("kind", ["IFNode", "LIFNode", "ParametricLIFNode"])
@pytest.mark.parametrize("layout", ["time_inner", "sliced", "broadcast"])
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
def test_registered_neuron_preserves_noncontiguous_values_and_gradients(
    kind, layout, device
):
    torch.manual_seed(42)
    node_type = getattr(neuron, kind)
    node = node_type(step_mode="m", store_v_seq=False).to(device)
    reference = node_type(step_mode="m", store_v_seq=False).to(device)
    reference.load_state_dict(node.state_dict())
    x = _layout(torch.rand(4, 2, 3, 5, device=device) * 0.8, layout).requires_grad_()
    state = _layout(torch.rand(2, 3, 5, device=device) * 0.1, layout).requires_grad_()
    reference_state = state.detach().contiguous().requires_grad_()
    node.v = state
    reference.v = reference_state

    actual = node(x)
    expected = reference(x.contiguous())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(node.v, reference.v, rtol=2e-5, atol=2e-6)
    actual_grads = torch.autograd.grad(actual.sum() + node.v.sum(), (x, state))
    expected_grads = torch.autograd.grad(
        expected.sum() + reference.v.sum(), (x, reference_state)
    )
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind", ["IFNode", "LIFNode", "ParametricLIFNode"])
def test_single_step_noncontiguous_input(kind):
    node = getattr(neuron, kind)(step_mode="s")
    base = torch.rand(2, 10)
    x = base[:, ::2].requires_grad_()
    y = node(x)
    assert y.shape == x.shape
    assert torch.isfinite(torch.autograd.grad(y.sum(), x)[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["triton", "cuda"])
@pytest.mark.parametrize(
    "layout,dtype,spatial_shape",
    [
        (layout, dtype, (2, 3, 5))
        for layout in ("time_inner", "channels_last", "broadcast")
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
    ]
    + [
        ("sliced", torch.float32, (2, 3, 5)),
        ("overlap", torch.float32, (2, 3, 5)),
        ("singleton_strides", torch.float32, (1, 3, 5)),
        ("offset", torch.float16, (2, 3, 5)),
        ("broadcast_space", torch.float32, (2, 3, 5)),
        ("broadcast_scalar", torch.float32, (2, 3, 5)),
        ("batch_major_channels_last", torch.float32, (2, 32, 8)),
        ("channels_last_time_inner", torch.float32, (2, 32, 8)),
    ]
    + [
        pytest.param("channels_last", dtype, (2, channels, width), id=name)
        for channels, width, dtype, name in (
            (16, 8, torch.float32, "spatial-tile-16"),
            (32, 8, torch.float16, "spatial-tile-32"),
            (64, 8, torch.bfloat16, "spatial-tile-64"),
            (32, 7, torch.float32, "spatial-tail"),
            (33, 8, torch.float32, "spatial-nonpower"),
        )
    ],
)
@pytest.mark.parametrize(
    "family",
    [
        "if_",
        "lif",
        "plif",
        "qif",
        "eif",
        "izhikevich",
        "ilif",
        "activation_aware_if",
        "stbif",
    ],
)
@pytest.mark.parametrize("trace", [False, True])
def test_strided_execution_does_not_materialize_inputs(
    provider, layout, family, trace, dtype, spatial_shape, monkeypatch
):
    from benchmark.check_neuron_dispatch import _arguments

    package = importlib.import_module(f"spikingjelly._ops.{family}")

    monkeypatch.setattr(package._selection, "_requested", provider)
    monkeypatch.setattr(package._selection, "_selections", {})
    monkeypatch.setattr(package._selection, "_compiled_selections", {})
    torch.manual_seed(42)
    training = family not in ("activation_aware_if", "stbif")
    source_shape = {
        "broadcast": (1, *spatial_shape),
        "broadcast_space": (4, spatial_shape[0], 1, spatial_shape[2]),
        "broadcast_scalar": (1, 1, 1, 1),
        "offset": (4 * math.prod(spatial_shape) + 1,),
    }.get(layout, (4, *spatial_shape))
    source = torch.rand(
        source_shape,
        device="cuda",
        dtype=dtype,
        requires_grad=training,
    )
    if layout == "offset":
        x = source[1:].view(4, *spatial_shape)
    elif layout.startswith("broadcast"):
        x = source.expand(4, *spatial_shape)
    else:
        x = _layout(source, layout)
    state_source = torch.rand(
        1, spatial_shape[1], 1, device="cuda", requires_grad=training
    )
    v = state_source.expand(spatial_shape)
    args, _ = _arguments(
        family.removesuffix("_"), 4, math.prod(spatial_shape), x.device, dtype
    )
    args = list(args)
    args[:2] = (x, v)
    if family in ("izhikevich", "stbif"):
        args[2] = _layout(torch.zeros_like(v), "time_inner").requires_grad_(training)
    parameters = list(inspect.signature(package._cpu._forward_impl).parameters)
    if "store_v_seq" in parameters:
        args[parameters.index("store_v_seq")] = trace
    inputs = (
        (
            source,
            state_source,
            *(t for t in args[2:] if isinstance(t, torch.Tensor) and t.requires_grad),
        )
        if training
        else ()
    )
    tensors = [t for t in args if isinstance(t, torch.Tensor)]
    snapshots = [t.detach().clone() for t in tensors]
    expected = package._selection._cpu_forward(*args)
    gs = (
        torch.rand(x.shape, device="cuda", dtype=dtype)
        if spatial_shape[1] >= 16
        else torch.tensor(0.25, device="cuda", dtype=dtype).expand_as(x)
    )
    visible = 3 if family == "izhikevich" else 2
    grads = (
        gs,
        *(_layout(torch.rand_like(t), "time_inner") for t in expected[1:visible]),
    )

    def run():
        outputs = package._forward(*args)
        gradients = (
            torch.autograd.grad(outputs[:visible], inputs, grads, retain_graph=True)
            if inputs
            else ()
        )
        return outputs, gradients

    run()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        outputs, gradients = run()
    assert package._selection.diagnostics(x.device)["implementation"] == provider
    torch.testing.assert_close(outputs, expected, rtol=2e-5, atol=2e-6)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = package._forward(
            *(mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args)
        )
    assert [t.stride() for t in fake] == [t.stride() for t in outputs]
    torch.testing.assert_close(
        gradients,
        torch.autograd.grad(expected[:visible], inputs, grads) if inputs else (),
        rtol=2e-5 if dtype == torch.float32 else 0.02,
        atol=2e-6 if dtype == torch.float32 else 0.002,
    )
    names = {event.key for event in p.key_averages()}
    if layout not in ("sliced", "overlap"):
        assert not names.intersection({"aten::contiguous", "aten::clone"}), names
    if layout in ("time_inner", "channels_last"):
        assert outputs[0].stride() == x.stride()
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    storage = [t.untyped_storage().data_ptr() for t in outputs]
    assert len(set(storage)) == len(outputs)
    assert not set(storage).intersection(
        t.untyped_storage().data_ptr() for t in tensors
    )
    assert all(torch.ops.aten.is_non_overlapping_and_dense.default(t) for t in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("trace", [False, True])
def test_wide_native_lif_mixed_spatial_orders(trace, monkeypatch):
    test_strided_execution_does_not_materialize_inputs(
        "cuda",
        "channels_last",
        "lif",
        trace,
        torch.float16,
        (2, 32, 32768),
        monkeypatch,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["triton", "cuda"])
def test_strided_channel_parameters_keep_logical_channel_order(provider, monkeypatch):
    from spikingjelly._ops import activation_aware_if as package

    monkeypatch.setattr(package._selection, "_requested", provider)
    monkeypatch.setattr(package._selection, "_selections", {})
    x = _layout(torch.rand(4, 2, 3, 2, 5, device="cuda"), "channels_last")
    assert x.flatten(0, 1).is_contiguous(memory_format=torch.channels_last)
    v = torch.zeros(1, 3, 1, 5, device="cuda").expand(x.shape[1:])
    threshold = torch.linspace(0.5, 1.5, 6, device="cuda").reshape(3, 2).t()
    offset = torch.tensor(0.1, device="cuda").expand(6)
    args = (x, v, threshold, offset, 6, 5, None, True)
    package._forward(*args)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        result = package._forward(*args)
    torch.testing.assert_close(result, package._selection._cpu_forward(*args))
    assert not {event.key for event in p.key_averages()}.intersection(
        {"aten::contiguous", "aten::clone"}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["triton", "cuda"])
@pytest.mark.parametrize(
    "family",
    [
        "if_",
        "lif",
        "plif",
        "qif",
        "eif",
        "izhikevich",
        "ilif",
        "activation_aware_if",
        "stbif",
    ],
)
def test_strided_operator_registration(family, provider, monkeypatch):
    from benchmark.check_neuron_dispatch import _arguments

    package = importlib.import_module(f"spikingjelly._ops.{family}")
    monkeypatch.setattr(package._selection, "_requested", provider)
    monkeypatch.setattr(package._selection, "_selections", {})
    monkeypatch.setattr(package._selection, "_compiled_selections", {})
    args, _ = _arguments(family.removesuffix("_"), 4, 17, "cuda")
    args = list(args)
    args[0] = _layout(args[0].detach(), "time_inner").requires_grad_(
        args[0].requires_grad
    )
    args[1] = args[1][:1].expand_as(args[1])
    torch.library.opcheck(package._forward, tuple(args))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["triton", "cuda"])
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
@pytest.mark.parametrize("family", ["lif", "plif", "izhikevich"])
def test_compiled_strided_layout_transitions(family, provider, mode, monkeypatch):
    from benchmark.check_neuron_dispatch import _arguments

    torch.compiler.reset()
    package = importlib.import_module(f"spikingjelly._ops.{family}")
    monkeypatch.setattr(package._selection, "_requested", provider)
    monkeypatch.setattr(package._selection, "_selections", {})
    monkeypatch.setattr(package._selection, "_compiled_selections", {})
    run = torch.compile(package._forward, fullgraph=True, mode=mode)
    for layout in (
        "contiguous",
        "channels_last",
        "time_inner",
        "broadcast",
        "contiguous",
    ):
        args, inputs = _arguments(family, 4, 60, "cuda")
        args = list(args)
        mapped = {}
        for i, tensor in enumerate(args):
            if isinstance(tensor, torch.Tensor) and tensor.ndim:
                shape = (4, 2, 3, 2, 5) if i == 0 else (2, 3, 2, 5)
                value = tensor.detach().reshape(shape)
                if i == 0:
                    value = _layout(value, layout)
                args[i] = value.requires_grad_(tensor.requires_grad)
                mapped[id(tensor)] = args[i]
        inputs = tuple(mapped.get(id(tensor), tensor) for tensor in inputs)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            expected = package._selection._cpu_forward(*args)
            result = run(*args)
            visible = 3 if family == "izhikevich" else 2
            grads = tuple(torch.full_like(t, 0.25) for t in result[:visible])
            torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(
                torch.autograd.grad(result[:visible], inputs, grads),
                torch.autograd.grad(expected[:visible], inputs, grads),
                rtol=2e-5,
                atol=2e-6,
            )
            del result, expected, grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_launch_keeps_input_state_and_gradient_pointers(monkeypatch):
    from spikingjelly._ops.lif import triton as implementation

    seen = {}

    class RecordLaunch:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def record(*args, **kwargs):
                seen[self.name] = tuple(
                    (t.data_ptr(), t.stride())
                    for t in args
                    if isinstance(t, torch.Tensor)
                )
                return launch(*args, **kwargs)

            return record

    for name in ("_forward_kernel", "_backward_kernel"):
        monkeypatch.setattr(
            implementation, name, RecordLaunch(getattr(implementation, name), name)
        )
    x = _layout(torch.rand(4, 2, 3, 5, device="cuda"), "channels_last")
    v = torch.zeros(1, 3, 1, device="cuda").expand(x.shape[1:])
    s, voltage, h = implementation._forward_impl(x, v, 2.0, True, 1.0, 0.0, False, 4.0)
    gs = torch.tensor(0.25, device="cuda").expand_as(s)
    gv = _layout(torch.rand_like(voltage), "time_inner")
    implementation._backward_impl(gs, gv, h, 2.0, True, 1.0, 0.0, False, 4.0)
    identity = lambda t: (t.data_ptr(), t.stride())
    assert seen["_forward_kernel"][:2] == (identity(x), identity(v))
    assert seen["_backward_kernel"][:3] == (identity(gs), identity(gv), identity(h))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["triton", "cuda"])
def test_native_and_triton_large_physical_offset(provider, monkeypatch):
    from spikingjelly._ops import lif

    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("Large-address regression needs 10 GiB free")
    monkeypatch.setattr(lif._selection, "_requested", provider)
    monkeypatch.setattr(lif._selection, "_selections", {})
    storage = torch.empty((1 << 31) + 1, device="cuda", dtype=torch.float16)
    x = storage.as_strided((2, 1), (1 << 31, 1))
    x.fill_(0.25)
    x.requires_grad_()
    v = torch.zeros(1, device="cuda", requires_grad=True)
    args = (x, v, 2.0, True, 1.0, 0.0, False, 4.0)
    actual = lif._forward(*args)
    expected = lif._selection._cpu_forward(*args)
    grads = (x.detach(), torch.ones_like(actual[1]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[:2], (x, v), grads),
        torch.autograd.grad(expected[:2], (x, v), grads),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("minor", [1, 16])
def test_triton_large_logical_indices(minor):
    pytest.importorskip("triton")
    n = (1 << 31) + (17 if minor == 1 else 32)
    out = torch.empty(512, device="cuda", dtype=torch.int64)
    _large_index_probe[(triton.cdiv(n, 256),)](out, n, minor)
    expected = torch.arange(256, device="cuda", dtype=torch.int64) + (1 << 31)
    torch.testing.assert_close(out[:256], expected)
    torch.testing.assert_close(out[256:], (expected < n).to(torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fake_layout_matches_automatic_torch_fallback(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from spikingjelly._ops import lif

    original_import = importlib.import_module

    def without_acceleration(name, package=None):
        if package == lif.__name__ and name in (".native", ".triton"):
            raise ImportError("Optional CUDA implementation is not installed")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_acceleration)
    monkeypatch.setattr(lif._selection, "_requested", "auto")
    monkeypatch.setattr(lif._selection, "_selections", {})
    x = torch.rand(2, 3, 4, device="cuda").permute(2, 0, 1)
    v = torch.zeros(2, 3, device="cuda")
    args = (x, v, 2.0, True, 1.0, 0.0, False, 4.0)
    actual = lif._forward(*args)
    assert lif._selection.diagnostics(x.device)["implementation"] == "torch"
    with FakeTensorMode() as mode:
        fake = lif._forward(
            *(mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args)
        )
    assert [t.stride() for t in fake] == [t.stride() for t in actual]
