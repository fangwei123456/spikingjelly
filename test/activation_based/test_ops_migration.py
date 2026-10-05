import importlib
import math

import pytest
import torch
from torch.nn import functional as F

from spikingjelly import configure
from spikingjelly.activation_based import functional, layer, surrogate
from spikingjelly.activation_based.neuron import experimental


@pytest.fixture(params=["cpu", "cuda"], scope="module")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize(
    "shape", [(), (0,), (1,), (7,), (8,), (9,), (2, 0, 3), (2, 3, 7)]
)
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.float32, torch.float16, torch.bfloat16]
)
def test_binary_pack_roundtrip_and_format(device, shape, dtype):
    x = (torch.arange(math.prod(shape), device=device) % 2).reshape(shape).to(dtype)
    if x.ndim > 1:
        x = x.transpose(0, -1)
    packed = functional.bit_spike_compress(x)
    assert packed.dtype == torch.uint8 and packed.ndim == 1
    expected = torch.zeros((x.numel() + 7) // 8, device=device, dtype=torch.uint8)
    flat = x.flatten().to(torch.uint8)
    for bit in range(8):
        part = flat[bit::8]
        expected[: part.numel()] |= part << bit
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    decoded = functional.bit_spike_decompress(packed, tuple(x.shape), dtype)
    torch.testing.assert_close(decoded, x, rtol=0, atol=0)


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("kind", ["linear", "conv1d", "conv2d", "conv3d"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_binary_linear_convolution_gradients(device, level, kind, dtype, monkeypatch):
    if device.type == "cpu" and dtype != torch.float32:
        pytest.skip("GPU covers low-precision vendor convolution")
    monkeypatch.setattr(configure, "save_bool_spike_level", level)
    torch.manual_seed(910)
    dims = int(kind[-2]) if kind != "linear" else 0
    shape = (2, 4, *((7,) * dims)) if dims else (2, 3, 4)
    weight_shape = (6, 2, *((3,) * dims)) if dims else (6, 4)
    x = (torch.rand(shape, device=device) > 0.5).to(dtype).requires_grad_()
    w = torch.randn(weight_shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(6, device=device, dtype=dtype, requires_grad=True)
    options = dict(groups=2, padding=1, stride=2) if dims else {}
    actual = getattr(functional, "spike_" + kind)(x, w, b, **options)
    expected = getattr(F, kind)(x, w, b, **options)
    tol = (
        0.025 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 2e-5
    )
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    grad = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, (x, w, b), grad, retain_graph=True)
    for _ in range(2):
        got = torch.autograd.grad(actual, (x, w, b), grad, retain_graph=True)
        torch.testing.assert_close(got, reference, rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", [(0, 4), (2, 0), (0, 0), (4,)])
def test_dense_linear_empty_and_vector_inputs(device, shape, monkeypatch):
    monkeypatch.setattr(configure, "save_bool_spike_level", 1)
    x = torch.ones(shape, device=device, requires_grad=True)
    w = torch.randn(3, shape[-1], device=device, requires_grad=True)
    b = torch.randn(3, device=device, requires_grad=True)
    actual = functional.spike_linear(x, w, b)
    expected = F.linear(x, w, b)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), (x, w, b)),
        torch.autograd.grad(expected.sum(), (x, w, b)),
    )


@pytest.mark.parametrize("level", [0, 1])
def test_binary_operator_opcheck_and_compile(device, level, monkeypatch):
    from spikingjelly._ops.spike_conv.torch import _convolution
    from spikingjelly._ops.spike_linear.dense import _linear

    monkeypatch.setattr(configure, "save_bool_spike_level", level)
    x = torch.ones(2, 3, device=device, requires_grad=True)
    w = torch.randn(4, 3, device=device, requires_grad=True)
    b = torch.randn(4, device=device, requires_grad=True)
    functional.spike_linear(x, w, b).sum().backward()
    torch.library.opcheck(_linear, (x, w, b))
    fn = torch.compile(functional.spike_linear, fullgraph=True)
    actual = fn(x, w, b)
    expected = F.linear(x, w, b)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), (x, w, b)),
        torch.autograd.grad(expected.sum(), (x, w, b)),
    )
    x = torch.ones(2, 2, 7, device=device, requires_grad=True)
    w = torch.randn(4, 2, 3, device=device, requires_grad=True)
    torch.library.opcheck(_convolution, (x, w, None, [1], [1], [1], 1))
    fn = torch.compile(functional.spike_conv1d, fullgraph=True)
    got = fn(x, w, padding=1)
    reference = F.conv1d(x, w, padding=1)
    torch.testing.assert_close(got, reference)
    torch.testing.assert_close(
        torch.autograd.grad(got.sum(), (x, w)),
        torch.autograd.grad(reference.sum(), (x, w)),
    )


def test_registered_binary_layers_preserve_parameters_and_padding(device):
    for name, args, shape in (
        ("SpikeLinear", (3, 4), (2, 3)),
        ("SpikeConv1d", (2, 4, 3), (2, 2, 7)),
        ("SpikeConv2d", (2, 4, 3), (2, 2, 7, 7)),
    ):
        kwargs = (
            {} if name == "SpikeLinear" else dict(padding=1, padding_mode="reflect")
        )
        module = getattr(layer, name)(*args, **kwargs).to(device)
        reference_type = (
            torch.nn.Linear if name == "SpikeLinear" else getattr(torch.nn, name[5:])
        )
        reference = reference_type(*args, **kwargs).to(device)
        reference.load_state_dict(module.state_dict())
        x = torch.ones(shape, device=device, requires_grad=True)
        got, wanted = module(x), reference(x)
        torch.testing.assert_close(got, wanted)
        torch.testing.assert_close(
            torch.autograd.grad(got.sum(), (x, *module.parameters())),
            torch.autograd.grad(wanted.sum(), (x, *reference.parameters())),
        )


@pytest.mark.parametrize("kind", ["linear", "conv1d"])
def test_binary_operators_autocast_fp32_master_parameters(device, kind, monkeypatch):
    monkeypatch.setattr(configure, "save_bool_spike_level", 1)
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    shape = (2, 3) if kind == "linear" else (2, 3, 7)
    weight_shape = (4, 3) if kind == "linear" else (4, 3, 3)
    torch.manual_seed(92)
    x = (torch.rand(shape, device=device) > 0.5).float().requires_grad_()
    weight = torch.randn(weight_shape, device=device, requires_grad=True)
    bias = torch.randn(4, device=device, requires_grad=True)
    with torch.autocast(device.type, dtype=dtype):
        actual = getattr(functional, "spike_" + kind)(x, weight, bias)
        expected = getattr(F, kind)(x, weight, bias)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual.float().sum(), (x, weight, bias))
    want = torch.autograd.grad(expected.float().sum(), (x, weight, bias))
    assert all(t.dtype == torch.float32 for t in got)
    torch.testing.assert_close(got, want, rtol=0.015, atol=0.005)
    compiled = torch.compile(getattr(functional, "spike_" + kind), fullgraph=True)
    with torch.autocast(device.type, dtype=dtype):
        actual = compiled(x, weight, bias)
        expected = getattr(F, kind)(x, weight, bias)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.float().sum(), (x, weight, bias)),
        torch.autograd.grad(expected.float().sum(), (x, weight, bias)),
        rtol=0.015,
        atol=0.005,
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_binary_convolution_unbatched_inputs(device, dim):
    x = torch.ones((2, *((5,) * dim)), device=device, requires_grad=True)
    weight = torch.randn((4, 2, *((3,) * dim)), device=device, requires_grad=True)
    actual = getattr(functional, f"spike_conv{dim}d")(x, weight, padding=1)
    expected = getattr(F, f"conv{dim}d")(x, weight, padding=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), (x, weight)),
        torch.autograd.grad(expected.sum(), (x, weight)),
    )


@pytest.mark.parametrize(
    "kind", ["QIF", "EIF", "Izhikevich", "ILIF", "ActivationAwareIF", "STBIF"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_new_experimental_nodes_state_reset_and_chunks(device, kind, dtype):
    cls = getattr(experimental, "Experimental" + kind + "Node")
    node = cls(**({} if kind == "STBIF" else {"store_v_seq": True})).to(device)
    x = torch.full((5, 2, 3), 0.6, device=device, dtype=dtype)
    training = kind not in ("ActivationAwareIF", "STBIF")
    x.requires_grad_(training)
    parts = [node(part) for part in (x[:1], x[1:3], x[3:])]
    state = node.q if kind == "STBIF" else node.v
    initial_loss = torch.cat(parts).float().sum() + state.sum()
    node.reset()
    whole = node(x)
    final = node.q if kind == "STBIF" else node.v
    torch.testing.assert_close(torch.cat(parts), whole, rtol=0, atol=0)
    torch.testing.assert_close(state, final)
    assert final.dtype == torch.float32
    if training:
        g1 = torch.autograd.grad(initial_loss, x, retain_graph=True)
        g2 = torch.autograd.grad(whole.float().sum() + final.sum(), x)
        torch.testing.assert_close(g1, g2, rtol=0.015, atol=0.005)
    node.reset()
    assert getattr(node, "q" if kind == "STBIF" else "v") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
@pytest.mark.parametrize("kind", ["if", "lif", "plif", "qif", "eif", "izhikevich"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "surrogate_name", ["Sigmoid", "ATan", "PiecewiseLeakyReLU", "LogTailedReLU"]
)
def test_generated_cupy_matches_retained(kind, dtype, surrogate_name):
    if kind == "izhikevich" and dtype != torch.float32:
        pytest.skip("retained Izhikevich CuPy contract is FP32")
    name = "integrate_and_fire" if kind == "if" else kind
    package = "if_" if kind == "if" else kind
    old = importlib.import_module(
        "spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step." + name
    )
    new = importlib.import_module("spikingjelly._ops." + package + ".cupy_generated")
    torch.manual_seed(111)
    columns = 8 if kind == "plif" and dtype == torch.float16 else 7
    x = (
        (torch.rand(4, 3, columns, device="cuda", dtype=dtype) * 0.6)
        .transpose(1, 2)
        .requires_grad_()
    )
    v = torch.zeros_like(x[0], requires_grad=True)
    sg = getattr(surrogate, surrogate_name)()
    inputs = (x, v)
    if kind == "if":
        args = (x, v, 0.7, 0.2, False, sg)
    elif kind == "lif":
        args = (x, v, True, 2.3, 0.7, 0.2, False, sg)
    elif kind == "plif":
        q = torch.tensor(0.4, device="cuda", dtype=dtype, requires_grad=True)
        inputs += (q,)
        args = (x, v, q, True, 0.7, 0.2, False, sg)
    elif kind == "qif":
        args = (x, v, 2.3, 0.7, 0.2, -0.2, 0.8, 0.4, False, sg)
    elif kind == "eif":
        args = (x, v, 2.3, 0.7, 0.2, -0.2, 0.9, 0.7, False, sg)
    else:
        w = torch.zeros_like(v, requires_grad=True)
        inputs += (w,)
        args = (x, v, w, 2.3, 0.7, 0.2, -0.2, 0.2, 0.3, 3.1, 0.8, 0.4, False, sg)
    a = getattr(old, kind + "_multi_step")(*args)
    b = getattr(new, kind + "_multi_step")(*args)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(sum(t.sum() for t in a), inputs),
        torch.autograd.grad(sum(t.sum() for t in b), inputs),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize(
    "storage,compute,backward",
    [
        (torch.float32, "fp32", "fp32"),
        (torch.float16, "fp32", "fp32"),
        (torch.bfloat16, "bf16", "fp32"),
    ],
)
def test_triton_precision_matches_retained(kind, storage, compute, backward):
    old_name = "integrate_and_fire" if kind == "if" else kind
    new_name = "if_" if kind == "if" else kind
    old = importlib.import_module(
        "spikingjelly.activation_based.triton_kernel.neuron_kernel." + old_name
    )
    new = importlib.import_module("spikingjelly._ops." + new_name + ".triton_precision")
    torch.manual_seed(87)
    x = torch.rand(4, 3, 7, device="cuda").transpose(1, 2).requires_grad_()
    v = torch.zeros_like(x[0], dtype=storage, requires_grad=True)
    q = torch.tensor(0.4, device="cuda", requires_grad=True)
    args = (x, v, q) if kind == "plif" else (x, v)
    options = dict(
        storage_dtype=storage,
        compute_dtype=compute,
        backward_compute_dtype=backward,
        spike_dtype=torch.float32,
        v_threshold=0.7,
        v_reset=0.2,
        detach_reset=False,
        surrogate_function=surrogate.ATan(),
        save_intermediates=True,
    )
    if kind == "lif":
        options.update(tau=2.3, decay_input=True, store_v_seq=True)
    if kind == "plif":
        options.update(decay_input=True)
    a = getattr(old, "_multistep_" + kind + "_mp")(
        *(t.contiguous() for t in args), **options
    )
    b = getattr(new, "_multistep_" + kind + "_mp")(*args, **options)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(
            a[:2], args, tuple(torch.ones_like(t).contiguous() for t in a[:2])
        ),
        torch.autograd.grad(b[:2], args, tuple(torch.ones_like(t) for t in b[:2])),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("T", [1, 4])
def test_fused_linear_matches_retained_and_torch(kind, T):
    old = importlib.import_module(
        "spikingjelly.activation_based.cuda_kernel.neuron_linear"
    )
    new = getattr(functional, kind + "_linear")
    torch.manual_seed(25)
    x = torch.rand(T, 3, 17, device="cuda", requires_grad=True)
    v = torch.zeros(3, 17, device="cuda", requires_grad=True)
    weight = torch.randn(17, 9, device="cuda", requires_grad=True)
    bias = torch.randn(9, device="cuda", requires_grad=True)
    args = (x[0] if T == 1 else x, v, weight, bias)
    options = dict(
        v_threshold=0.7,
        v_reset=0.2,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
        threads=128,
    )
    a = getattr(old, kind + "_linear")(*args, **options)
    b = new(*args, **options)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    inputs = (x, v, weight, bias)
    torch.testing.assert_close(
        torch.autograd.grad(a[0].sum() + a[1].sum(), inputs),
        torch.autograd.grad(b[0].sum() + b[1].sum(), inputs),
        rtol=0,
        atol=0,
    )
    # Compilation must resolve the new registered op, including metadata-only use.
    fn = torch.compile(new, fullgraph=True)
    fresh = tuple(t.detach() for t in args)
    eager = new(*fresh, **options)
    compiled = fn(*fresh, **options)
    torch.testing.assert_close(eager, compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sparse_and_prepacked_linear_match_retained(dtype):
    old = importlib.import_module(
        "spikingjelly.activation_based.cuda_kernel.spike_linear"
    )
    torch.manual_seed(47)
    x = (torch.rand(5, 19, device="cuda") > 0.7).to(dtype).requires_grad_()
    w = torch.randn(7, 19, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(7, device="cuda", dtype=dtype, requires_grad=True)
    pa, pb = old.bit_pack_spike_dense(x), functional.bit_pack_spike_dense(x)
    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    for a, got, inputs in (
        (
            old.sparse_linear(x, w, b, "sparse"),
            functional.sparse_linear(x, w, b, "sparse"),
            (x, w, b),
        ),
        (
            old.cupy_spike_linear_v3_dense_forward(pa, w, b),
            functional.packed_spike_linear(pb, w, b),
            (w, b),
        ),
    ):
        torch.testing.assert_close(a, got, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(a.sum(), inputs),
            torch.autograd.grad(got.sum(), inputs),
            rtol=0,
            atol=0,
        )
    fn = torch.compile(functional.bit_pack_spike_dense, fullgraph=True)
    torch.testing.assert_close(fn(x.detach()), pb, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_migrated_cupy_uses_current_stream_and_device():
    for index in range(min(2, torch.cuda.device_count())):
        device = torch.device("cuda", index)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            x = torch.empty(4, 3, 17, device=device)
            torch.cuda._sleep(1000000)
            x.fill_(0.4).requires_grad_()
            v = torch.zeros_like(x[0], requires_grad=True)
            actual = functional.lif_multi_step_cupy(
                x, v, 2.0, True, 0.7, 0.2, surrogate.ATan(), False, True
            )
            states, spikes = [], []
            voltage = v
            for current in x:
                spike, voltage = functional.lif_step(
                    current, voltage, 2.0, True, 0.7, 0.2, surrogate.ATan(), False
                )
                spikes.append(spike)
                states.append(voltage)
            expected = torch.stack(spikes), voltage, torch.stack(states)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual[0].sum() + actual[1].sum(), (x, v)),
                torch.autograd.grad(expected[0].sum() + expected[1].sum(), (x, v)),
                rtol=2e-4,
                atol=2e-5,
            )
        stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_generated_log_tailed_relu_half_zero_and_tiny_values():
    cupy = pytest.importorskip("cupy")
    function = surrogate.LogTailedReLU(alpha=0.25)
    values = [-0.5, 0.0, -0.00001, 0.00001, 1.0, 2.0]
    code = function.cuda_codes(y="half2 result", x="value", dtype="half2")
    kernel = cupy.RawKernel(
        '#include <cuda_fp16.h>\nextern "C" __global__ void check(const half2* x, half2* y) '
        "{ int i = threadIdx.x; half2 value = x[i]; " + code + " y[i] = result; }",
        "check",
    )
    x = torch.tensor(values, device="cuda", dtype=torch.float16, requires_grad=True)
    expected = torch.autograd.grad(function(x).sum(), x)[0]
    result = torch.empty_like(x)
    kernel(
        (1,),
        (3,),
        (x.data_ptr(), result.data_ptr()),
        stream=cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream),
    )
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
