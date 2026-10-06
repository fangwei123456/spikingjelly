import math

import pytest
import torch
from torch.nn import functional as F

from spikingjelly import configure
from spikingjelly.activation_based import functional, layer


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
