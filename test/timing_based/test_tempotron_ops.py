import pytest
import torch

from spikingjelly._ops.tempotron_states import cpu as states
from spikingjelly._ops.tempotron_roots import cpu as roots
from spikingjelly._ops.tempotron_weight_gradient import cpu as gradient
from spikingjelly.timing_based.functional import _parameters, tempotron


def _normalizer(slow, delta):
    peak = torch.tensor(delta * slow, dtype=torch.float64).log1p() / delta
    return torch.exp(-peak / slow) * (-torch.expm1(-delta * peak)) / delta


@pytest.fixture
def cuda_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from spikingjelly._ops.tempotron_states import triton as states_cuda
    from spikingjelly._ops.tempotron_roots import triton as roots_cuda
    from spikingjelly._ops.tempotron_weight_gradient import triton as gradient_cuda

    return states_cuda, roots_cuda, gradient_cuda


@pytest.mark.parametrize("fast", [0.5, 2.0 - 1e-10])
def test_fused_states_and_roots_match_fp64(cuda_ops, fast):
    states_cuda, roots_cuda, _ = cuda_ops
    slow, fast, delta = _parameters(5.0, 2.0, fast, 0.6)
    norm = _normalizer(slow, delta)
    norm = norm.cuda()
    times = torch.tensor([[0.1, 0.4, 0.4, 1.7]], device="cuda", dtype=torch.float64)
    times = times.repeat_interleave(2, dim=1)[:, ::2]
    incoming = torch.tensor(
        [[[0.2, -0.1, 0.8, 0.3], [0.8, 0.2, -0.4, 0.1]]],
        device="cuda",
        dtype=torch.float64,
    )
    carry = torch.tensor([[0.1, -0.2]], device="cuda", dtype=torch.float64)
    origin = torch.tensor([0.0], device="cuda", dtype=torch.float64)
    args = times, incoming, carry, carry, origin, slow, fast, delta, norm
    for actual, expected in zip(
        states_cuda._forward_impl(*args), states._forward_impl(*args)
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    shape = (2, 3)
    voltage = torch.zeros(shape, dtype=torch.float64, device="cuda")
    current = torch.ones_like(voltage)
    lower = torch.zeros_like(voltage)
    upper = torch.full_like(voltage.t(), 1.0 if fast < 1.0 else 2.0).t()
    mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
    args = voltage, current, lower, upper, mask, slow, fast, delta, norm, 0.6, 5.0
    torch.testing.assert_close(
        roots_cuda._forward_impl(*args),
        roots._forward_impl(*args),
        rtol=1e-12,
        atol=1e-14,
    )
    empty = (
        voltage[:0],
        current[:0],
        lower[:0],
        upper[:0],
        mask[:0],
        slow,
        fast,
        delta,
        norm,
        0.6,
        5.0,
    )
    assert roots_cuda._forward_impl(*empty).shape == (0, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_weight_gradient_with_repeated_strided_events(cuda_ops, dtype):
    _, _, gradient_cuda = cuda_ops
    slow, _, delta = _parameters(5.0, 2.0, 0.5, 0.6)
    norm = _normalizer(slow, delta)
    times = torch.tensor([0.3, 0.1, 0.1, 2.0, 0.2], device="cuda", dtype=dtype)
    times = times.repeat_interleave(2)[::2]
    indices = torch.tensor([0, 1, 1, 0, 2], device="cuda").repeat_interleave(2)[::2]
    weight = torch.randn(3, 2, device="cuda", dtype=dtype).t()
    order = torch.tensor([1, 2, 0, 3, 4], device="cuda")
    offsets = torch.tensor([0, 4, 4, 5], device="cuda")
    peak = torch.tensor(
        [[1.5, 2.8], [0.0, 0.0], [1.0, 1.3]], device="cuda", dtype=torch.float64
    )
    accepted = torch.tensor([[3, 4], [0, 0], [1, 1]], device="cuda")
    upstream = torch.tensor(
        [[0.2, -0.3], [0.0, 0.7], [1.1, -0.4]], device="cuda", dtype=torch.float64
    )
    args = (
        times,
        indices,
        weight,
        order,
        offsets,
        peak,
        accepted,
        upstream,
        slow,
        delta,
        norm.cuda(),
    )
    torch.testing.assert_close(
        gradient_cuda._forward_impl(*args),
        gradient._forward_impl(*args),
        rtol=1e-12,
        atol=1e-14,
    )
    empty = (
        times[:0],
        indices[:0],
        weight,
        order[:0],
        offsets.new_zeros(4),
        peak,
        accepted.new_zeros(accepted.shape),
        upstream,
        slow,
        delta,
        norm.cuda(),
    )
    torch.testing.assert_close(
        gradient_cuda._forward_impl(*empty), torch.zeros_like(weight).double()
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_weight_only_peak_gradient_matches_full_analytic_branch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    times = torch.tensor([0.1, 0.3, 1.5, 0.2, 0.4], dtype=torch.float64, device=device)
    indices = torch.tensor([0, 0, 1, 1, 0], device=device)
    offsets = torch.tensor([0, 3, 3, 5], device=device)
    weight = torch.tensor(
        [[0.8, 0.3], [0.1, -0.2]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    upstream = torch.tensor(
        [[0.7, -0.3], [0.2, 0.4], [-0.8, 0.1]], dtype=torch.float64, device=device
    )
    fast = tempotron(times, indices, offsets, weight, 5.0, 2.0, 0.5, 0.6)[0]
    actual = torch.autograd.grad(fast, weight, upstream)[0]
    full = tempotron(
        times.requires_grad_(), indices, offsets, weight, 5.0, 2.0, 0.5, 0.6
    )[0]
    expected = torch.autograd.grad(full, weight, upstream)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
