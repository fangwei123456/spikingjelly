import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.functional.neuron import lif_step

from spikingjelly._ops.lif import get_cuda_implementation, lif
from spikingjelly.activation_based.neuron.experimental import ExperimentalLIFNode


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if torch.version.hip:
            pytest.skip("native experiment targets NVIDIA CUDA")
        get_cuda_implementation(torch.device("cuda", 0))
    return request.param


def _reference(x, v, tau, decay_input, threshold, reset, detach_reset, alpha):
    spikes, voltages = [], []
    for current in x:
        spike, v = lif_step(
            current,
            v,
            tau,
            decay_input,
            threshold,
            reset,
            surrogate.Sigmoid(alpha=alpha),
            detach_reset,
        )
        spikes.append(spike)
        voltages.append(v)
    return torch.stack(spikes), torch.stack(voltages)


@pytest.mark.parametrize("output", ["spikes", "last_voltage"])
def test_one_output_and_broadcast_input(device, output):
    base = torch.tensor([[2.0, 1.0, 0.0]], device=device, requires_grad=True)
    x = base.expand(4, 3)
    v = torch.zeros(3, device=device, requires_grad=True)
    params = (2.0, True, 1.0, 0.0, False, 4.0)
    actual = lif(x, v, *params)
    expected = _reference(x, v, *params)
    # Includes an exact threshold crossing in the first step.
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    losses = (
        (actual[0].sum(), expected[0].sum())
        if output == "spikes"
        else (actual[1][-1].sum(), expected[1][-1].sum())
    )
    gradients = [torch.autograd.grad(loss, (base, v)) for loss in losses]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b)


def test_module_state_and_chunking(device):
    torch.manual_seed(21)
    x = (torch.randn(8, 2, 17, device=device) + 0.6).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    node = ExperimentalLIFNode(v_reset=None, store_v_seq=True)
    reference = neuron.LIFNode(v_reset=None, step_mode="m", store_v_seq=True)
    chunks, reference_chunks, traces = [], [], []
    for a, b in [(0, 1), (1, 3), (3, 8)]:
        chunks.append(node(x[a:b]))
        reference_chunks.append(reference(reference_x[a:b]))
        traces.append(node.v_seq)
        torch.testing.assert_close(node.v, reference.v)
        torch.testing.assert_close(node.v_seq, reference.v_seq)
    torch.testing.assert_close(torch.cat(chunks), torch.cat(reference_chunks))
    loss = torch.cat(chunks).sum() + torch.cat(traces).sum() + node.v.sum()
    # Independent full-sequence reference includes every trajectory contribution.
    reference.reset()
    expected_spikes = reference(reference_x)
    reference_loss = expected_spikes.sum() + reference.v_seq.sum() + reference.v.sum()
    torch.testing.assert_close(
        torch.autograd.grad(loss, x)[0],
        torch.autograd.grad(reference_loss, reference_x)[0],
    )
    assert node.state_dict() == {}
    node.to(dtype=torch.float64)
    assert node.v.dtype == torch.float64
    node.reset()
    assert node.v is None and node.v_seq is None
    node.store_v_seq = False
    with torch.no_grad():
        fresh = node(x.detach())
    torch.testing.assert_close(fresh, expected_spikes)
    assert node.v_seq is None
    node(torch.zeros(1, 5, device=device))
    assert node.v.shape == (5,)


def test_invalid_inputs(device):
    x = torch.zeros(2, 3, device=device)
    v = torch.zeros(3, device=device)
    params = (2.0, True, 1.0, 0.0, False, 4.0)
    for bad_x, bad_v in [
        (x.half(), v.half()),
        (x.bfloat16(), v.bfloat16()),
        (x.double(), v.double()),
        (x, v.double()),
        (x, v[:2]),
        (x[:0], v),
        (x[:, :0], v[:0]),
    ]:
        with pytest.raises(RuntimeError):
            lif(bad_x, bad_v, *params)
    for name, value in [
        ("tau", 1.0),
        ("tau", float("nan")),
        ("tau", float("inf")),
        ("alpha", 0.0),
        ("alpha", float("nan")),
        ("alpha", float("inf")),
        ("threshold", float("nan")),
        ("reset", float("inf")),
    ]:
        with pytest.raises(ValueError, match=name):
            lif(x, v, **{name: value})
    if device == "cuda":
        with pytest.raises(RuntimeError, match="devices"):
            lif(x, v.cpu(), *params)


def test_nondefault_stream_and_second_gpu(device):
    if device != "cuda":
        pytest.skip("CUDA stream test")
    device_count = min(torch.cuda.device_count(), 2)
    for index in range(device_count):
        target = torch.device("cuda", index)
        stream = torch.cuda.Stream(device=target)
        with torch.cuda.device(target), torch.cuda.stream(stream):
            x = torch.full((5, 257), 0.75, device=target, requires_grad=True)
            v = torch.zeros(257, device=target, requires_grad=True)
            params = (2.0, False, 0.8, None, False, 4.0)
            # Each implementation must select the input GPU and its current stream.
            with torch.cuda.device(0):
                actual = lif(x, v, *params)
            expected = _reference(x, v, *params)
            gradients = [
                torch.autograd.grad(out[0].sum() + out[1].sum(), (x, v))
                for out in [actual, expected]
            ]
        stream.synchronize()
        torch.testing.assert_close(actual[:2], expected)
        torch.testing.assert_close(*gradients)


def test_eval_retains_surrogate_gradient(device):
    x = torch.full((2, 3), 0.75, device=device, requires_grad=True)
    node = ExperimentalLIFNode().eval()
    actual = node(x)
    expected = _reference(x, torch.zeros_like(x[0]), 2.0, True, 1.0, 0.0, False, 4.0)
    torch.testing.assert_close(actual, expected[0])
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected[0].sum(), x)[0],
    )


def test_module_compile_preserves_state_and_gradients(device):
    node = ExperimentalLIFNode(store_v_seq=True)
    reference = ExperimentalLIFNode(store_v_seq=True)
    compiled = torch.compile(
        node,
        backend="inductor" if device == "cuda" else "aot_eager",
        fullgraph=True,
    )
    torch.manual_seed(23)
    x = torch.randn(4, 2, 7, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = torch.cat([compiled(chunk) for chunk in x.chunk(2)])
    expected = torch.cat([reference(chunk) for chunk in reference_x.chunk(2)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(node.v, reference.v)
    torch.testing.assert_close(node.v_seq, reference.v_seq)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum() + node.v.sum(), x)[0],
        torch.autograd.grad(expected.sum() + reference.v.sum(), reference_x)[0],
    )


@pytest.mark.parametrize("shape", [(3,), (0, 3)])
def test_module_rejects_missing_or_empty_time(shape):
    node = ExperimentalLIFNode()
    with pytest.raises(ValueError, match="T >= 1"):
        node(torch.empty(shape))
    assert node.v is None and node.v_seq is None


@pytest.mark.parametrize("dtype", [torch.float64, torch.float8_e4m3fn])
def test_module_rejects_unsupported_dtype(dtype):
    node = ExperimentalLIFNode()
    with pytest.raises(RuntimeError, match="float32"):
        node(torch.zeros(2, 3, dtype=dtype))
    assert node.v is None and node.v_seq is None


def test_state_returns_to_input_dtype(device):
    node = ExperimentalLIFNode(store_v_seq=True)
    x = torch.full((2, 3), 0.3, device=device)
    node(x)
    state = node.v.clone()
    node.double()
    assert node.v.dtype == torch.float64
    expected = _reference(x, state, 2.0, True, 1.0, 0.0, False, 4.0)
    torch.testing.assert_close(node(x), expected[0])
    torch.testing.assert_close(node.v, expected[1][-1])
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32
