from decimal import Decimal, localcontext

import pytest
import torch

from spikingjelly.activation_based.encoding import LatencyEncoder


def _reference_firing_times(x: torch.Tensor, period: int) -> torch.Tensor:
    # Evaluate the documented equation independently, without float overflow.
    with localcontext() as context:
        context.prec = 80
        alpha = Decimal(period - 1).exp() - 1
        times = [
            int(round(Decimal(period - 1) - (1 + alpha * Decimal(value)).ln()))
            for value in x.flatten().tolist()
        ]
    return torch.tensor(times).reshape(x.shape)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("period", [1, 4, 20, 100, 1000])
def test_log_latency_matches_high_precision_equation(dtype, period):
    x = torch.tensor([[0.0, 0.001, 0.1], [0.5, 0.9, 1.0]], dtype=dtype)
    original = x.clone()
    encoder = LatencyEncoder(period, enc_function="log")

    spikes = encoder.single_step_encode(x)

    assert spikes.shape == (period, *x.shape)
    assert spikes.dtype == dtype
    assert spikes.device == x.device
    torch.testing.assert_close(spikes.sum(0), torch.ones_like(x))
    assert torch.equal(spikes.argmax(0), _reference_firing_times(x, period))
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("period", [4, 20, 100])
def test_log_latency_cycles_and_resets_after_encoding(period):
    encoder = LatencyEncoder(period, enc_function="log")
    x = torch.tensor([0.0, 0.1, 0.5, 1.0])
    times = _reference_firing_times(x, period)
    expected = (torch.arange(period)[:, None] == times).to(x)

    actual = torch.stack([encoder(x)] + [encoder() for _ in range(period - 1)])

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(encoder(), expected[0])
    encoder.reset()
    next_x = 1 - x
    torch.testing.assert_close(
        encoder(next_x), (_reference_firing_times(next_x, period) == 0).to(x)
    )


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
@pytest.mark.parametrize("period", [4, 1000])
def test_log_latency_preserves_binary_input_support(dtype, period):
    x = torch.tensor([0, 1], dtype=dtype)
    spikes = LatencyEncoder(period, "log").single_step_encode(x)
    expected = torch.zeros((period, 2), dtype=dtype)
    expected[-1, 0] = 1
    expected[0, 1] = 1
    torch.testing.assert_close(spikes, expected)
