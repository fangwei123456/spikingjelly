import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn.recipes.spikezip_qann import (
    SpikeZIPConv2d,
    SpikeZIPLinear,
)


def test_spikezip_bias_updates_explicit_state():
    y = torch.randn(2, 3)
    bias = torch.tensor([0.5, -1.0, 1.5])

    y_next, remaining_steps, released = functional.spikezip_bias_step(
        y,
        bias,
        remaining_steps=2,
        bias_steps=4,
    )
    assert released
    assert remaining_steps == 1
    assert torch.allclose(y_next, y + bias / 4)

    y_seq = torch.randn(5, 2, 3)
    original = y_seq.clone()
    y_seq_next, remaining_steps, released_steps = functional.spikezip_bias_multi_step(
        y_seq,
        bias.view(1, 1, 3),
        remaining_steps=3,
        bias_steps=4,
    )
    expected = y_seq.clone()
    expected[:3] += bias.view(1, 1, 3) / 4
    assert released_steps == 3
    assert remaining_steps == 0
    assert torch.allclose(y_seq_next, expected)
    assert torch.allclose(y_seq, original)


def test_spikezip_bias_free_multi_step_is_idle_without_input():
    linear = SpikeZIPLinear(
        torch.nn.Linear(3, 2, bias=False),
        level=4,
        bias_steps=3,
        step_mode="m",
    )
    linear_out = linear(torch.zeros(2, 1, 3))
    assert torch.equal(linear_out, torch.zeros_like(linear_out))
    assert not linear.is_work
    assert linear.realize_time == 3

    conv = SpikeZIPConv2d(
        torch.nn.Conv2d(2, 3, kernel_size=1, bias=False),
        level=4,
        bias_steps=3,
        step_mode="m",
    )
    conv_out = conv(torch.zeros(2, 1, 2, 4, 4))
    assert torch.equal(conv_out, torch.zeros_like(conv_out))
    assert not conv.is_work
    assert conv.realize_time == 3
