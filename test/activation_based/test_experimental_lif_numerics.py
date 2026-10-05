import pytest
import torch

from spikingjelly._ops.lif import lif
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.functional.neuron import lif_step


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initial_state_gradient_near_unit_tau(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.zeros(1, 2, device=device, requires_grad=True)
    v = torch.tensor([0.1, 0.2], device=device, requires_grad=True)
    tau = 1.00000006
    _, actual, _ = lif(x, v, tau, True, 1.0, None, True, 4.0)
    _, expected = lif_step(x[0], v, tau, True, 1.0, None, surrogate.Sigmoid(), True)
    actual_grad = torch.autograd.grad(actual.sum() * 1e8, (x, v))
    expected_grad = torch.autograd.grad(expected.sum() * 1e8, (x, v))
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_atan_large_denominator_gradient():
    pytest.importorskip("triton")
    from spikingjelly._ops.lif import _forward, get_cuda_implementation

    if get_cuda_implementation(torch.device("cuda", 0))["implementation"] != "triton":
        pytest.skip("requires SJ_LIF_CUDA_IMPLEMENTATION=triton")

    # A subnormal surrogate derivative can still produce a normal input gradient.
    x = torch.tensor([[2.0**62, 2.0**63, 2.0**64]], requires_grad=True)
    v = torch.zeros(3, requires_grad=True)
    gs = torch.tensor([[1e38, 3e38, 1e38]])
    expected, _ = lif_step(
        x[0], v, 2.0, True, 1.0, 0.0, surrogate.ATan(alpha=2.0), True
    )
    expected_grad = torch.autograd.grad(expected, (x, v), gs[0])
    cx = x.detach().cuda().requires_grad_()
    cv = v.detach().cuda().requires_grad_()
    actual, _, _ = _forward(cx, cv, 2.0, True, 1.0, 0.0, True, 2.0, False, 1)
    actual_grad = torch.autograd.grad(actual, (cx, cv), gs.cuda())
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got.cpu(), want, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_wide_atan_fullgraph():
    pytest.importorskip("triton")
    from spikingjelly._ops.lif import _forward, get_cuda_implementation

    if get_cuda_implementation(torch.device("cuda", 0))["implementation"] != "triton":
        pytest.skip("requires SJ_LIF_CUDA_IMPLEMENTATION=triton")

    x = torch.full((4, 2097153), 0.3, device="cuda", requires_grad=True)
    v = torch.zeros(2097153, device="cuda", requires_grad=True)

    def run(x, v):
        return _forward(x, v, 2.0, True, 1.0, 0.0, True, 2.0, False, 1)[:2]

    expected = run(x, v)
    expected_grad = torch.autograd.grad(sum(t.sum() for t in expected), (x, v))
    actual = torch.compile(run, fullgraph=True)(x, v)
    actual_grad = torch.autograd.grad(sum(t.sum() for t in actual), (x, v))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
