import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.encoding import WeightedPhaseEncoder


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_weighted_phase_preserves_input_dtype(dtype, step_mode):
    """Preserve dtype, binary fractions and state cycles in both step modes."""
    x = torch.tensor([[0, 0.25, 0.5], [0.75, 0.9375, 0.125]], dtype=dtype)
    original = x.clone()
    expected = torch.tensor(
        [
            [[0, 0, 1], [1, 1, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 0]],
        ],
        dtype=dtype,
    ).repeat(2, 1, 1)
    encoder = WeightedPhaseEncoder(4, step_mode=step_mode)
    if step_mode == "s":
        actual = torch.stack([encoder(x)] + [encoder() for _ in range(7)])
    else:
        actual = encoder(x.unsqueeze(0).repeat(8, 1, 1))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, original)
    encoder.reset()
    assert encoder.spike is None and encoder.t == 0


def test_weighted_phase_can_feed_double_linear_and_backpropagate():
    """Train a double Linear layer directly from double encoded spikes."""
    encoder = WeightedPhaseEncoder(4)
    linear = torch.nn.Linear(2, 1, dtype=torch.float64)
    x = torch.tensor([[0.5, 0.25]], dtype=torch.float64)
    encoded = torch.stack([encoder(x)] + [encoder() for _ in range(3)])
    reference = torch.tensor([[[1, 0]], [[0, 1]], [[0, 0]], [[0, 0]]], dtype=x.dtype)

    actual = linear(encoded).square().sum()
    expected = F.linear(reference, linear.weight, linear.bias).square().sum()
    expected_grads = torch.autograd.grad(expected, tuple(linear.parameters()))
    actual.backward()

    torch.testing.assert_close(actual, expected)
    for parameter, gradient in zip(linear.parameters(), expected_grads):
        torch.testing.assert_close(parameter.grad, gradient)
