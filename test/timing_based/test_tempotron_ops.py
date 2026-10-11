import pytest
import torch

from spikingjelly._ops.tempotron import _roots, _states, _weight_gradient, cpu
from spikingjelly.timing_based.functional import _parameters, tempotron


def _normalizer(slow, delta):
    peak = torch.tensor(delta * slow, dtype=torch.float64).log1p() / delta
    return torch.exp(-peak / slow) * (-torch.expm1(-delta * peak)) / delta


def test_shared_package_registers_each_operator_fake_signature():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        state = torch.empty(2, 3, dtype=torch.float64)
        norm = torch.ones((), dtype=torch.float64)
        root = _roots(
            state,
            state,
            state,
            state,
            torch.empty(2, 3, dtype=torch.bool),
            2.0,
            0.5,
            1.5,
            norm,
            0.6,
            5.0,
        )
        incoming = torch.empty(2, 3, 4, dtype=torch.float64)
        states = _states(
            torch.empty(2, 4, dtype=torch.float64),
            incoming,
            state,
            state,
            torch.empty(2, dtype=torch.float64),
            2.0,
            0.5,
            1.5,
            norm,
        )
        events = torch.empty(4, dtype=torch.int64)
        gradient = _weight_gradient(
            torch.empty(4, dtype=torch.float32),
            events,
            torch.empty(3, 7, dtype=torch.float32),
            events,
            torch.empty(3, dtype=torch.int64),
            state,
            torch.empty(2, 3, dtype=torch.int64),
            state,
            2.0,
            1.5,
            norm,
        )
        assert root.shape == (2, 3) and root.dtype == torch.float64
        assert all(x.shape == incoming.shape for x in states)
        assert gradient.shape == (3, 7) and gradient.dtype == torch.float64


@pytest.fixture
def cuda_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from spikingjelly._ops.tempotron import triton

    return triton


@pytest.mark.parametrize("fast", [0.5, 2.0 - 1e-10])
def test_fused_states_and_roots_match_fp64(cuda_ops, fast):
    triton = cuda_ops
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
    for actual, expected in zip(triton._states_impl(*args), cpu._states_impl(*args)):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    shape = (2, 3)
    voltage = torch.zeros(shape, dtype=torch.float64, device="cuda")
    current = torch.ones_like(voltage)
    lower = torch.zeros_like(voltage)
    upper = torch.full_like(voltage.t(), 1.0 if fast < 1.0 else 2.0).t()
    mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
    args = voltage, current, lower, upper, mask, slow, fast, delta, norm, 0.6, 5.0
    torch.testing.assert_close(
        triton._roots_impl(*args),
        cpu._roots_impl(*args),
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
    assert triton._roots_impl(*empty).shape == (0, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_weight_gradient_with_repeated_strided_events(cuda_ops, dtype):
    triton = cuda_ops
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
        triton._weight_gradient_impl(*args),
        cpu._weight_gradient_impl(*args),
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
        triton._weight_gradient_impl(*empty), torch.zeros_like(weight).double()
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
