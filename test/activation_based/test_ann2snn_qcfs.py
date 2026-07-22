from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    Qwen2SNNModel,
    SignedQCFSSequenceEncoder,
)


def test_signed_qcfs_sequence_encoder_preserves_time_and_reconstructs_counts():
    scale = torch.tensor([0.25, 0.5, 1.0])
    value = torch.tensor(
        [
            [[-0.62, 0.74, 1.51], [0.13, -1.26, 0.49]],
            [[0.87, -0.24, -2.40], [-0.51, 1.76, 0.02]],
        ]
    )
    encoder = SignedQCFSSequenceEncoder(scale=scale, time_steps=4)

    sequence = encoder.encode(value)
    expected = torch.round(value / scale).clamp(-4, 4) * scale

    assert sequence.shape == (4, *value.shape)
    torch.testing.assert_close(sequence.sum(0), expected, atol=0.0, rtol=0.0)

    functional.reset_net(encoder)
    replay = encoder.encode(value)
    torch.testing.assert_close(replay, sequence, atol=0.0, rtol=0.0)


def test_signed_qcfs_sequence_encoder_supports_non_last_channel_dimension():
    scale = torch.tensor([0.25, 0.5, 1.0])
    value = torch.tensor([[[-0.62, 0.74], [1.51, 0.13], [-1.26, 0.49]]])
    encoder = SignedQCFSSequenceEncoder(
        scale=scale,
        time_steps=4,
        channel_dim=1,
        collect_statistics=False,
    )

    sequence = encoder.encode(value)
    expected = torch.round(value / scale.reshape(1, 3, 1)).clamp(-4, 4)
    expected = expected * scale.reshape(1, 3, 1)

    assert sequence.shape == (4, 1, 3, 2)
    torch.testing.assert_close(sequence.sum(0), expected, atol=0.0, rtol=0.0)


def test_signed_qcfs_sequence_encoder_validates_rank_and_metric_mask_shape():
    encoder = SignedQCFSSequenceEncoder(
        scale=torch.tensor([0.25, 0.5, 1.0]),
        time_steps=4,
        channel_dim=1,
    )

    with pytest.raises(ValueError, match="at least one dimension"):
        encoder.encode(torch.tensor(1.0))
    with pytest.raises(ValueError, match="metric_mask shape"):
        encoder.encode(torch.ones(2, 3, 4), metric_mask=torch.ones(2, 3))
    with pytest.raises(ValueError, match="select at least one"):
        encoder.encode(
            torch.ones(2, 3, 4),
            metric_mask=torch.zeros(2, 4, dtype=torch.bool),
        )


def test_signed_qcfs_metric_mask_supports_non_last_channel_dimension():
    encoder = SignedQCFSSequenceEncoder(
        scale=torch.tensor([0.25, 0.5, 1.0]),
        time_steps=4,
        channel_dim=1,
    )
    value = torch.ones(2, 3, 4)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    sequence = encoder.encode(value, metric_mask=mask)

    assert sequence.shape == (4, 2, 3, 4)
    assert encoder.spike_value_count == 4 * int(mask.sum()) * 3


def test_signed_qcfs_metric_mask_filters_boundary_corrections():
    value = torch.tensor([0.5, 0.125, 1.5, 0.375]).reshape(1, 4, 1)
    encoder = SignedQCFSSequenceEncoder(torch.tensor([1.0]), time_steps=4)

    encoder.encode(
        value,
        metric_mask=torch.tensor([[False, True, True, True]]),
    )

    assert encoder.boundary_correction_fraction == 0.0


def test_count_domain_reconstruction_equals_multistep_temporal_sum():
    encoder = SignedQCFSSequenceEncoder(
        torch.tensor([0.25, 0.5, 1.0]),
        time_steps=8,
        neuron_backend="torch",
    )
    value = torch.tensor([[0.125, -1.25, 9.0], [-0.5, 0.75, -2.5]])

    sequence = encoder.encode(value)

    assert torch.equal(encoder.reconstruct(value), sequence.sum(0))


def test_signed_qcfs_bfloat16_replays_large_time_step_counts_exactly():
    scale = torch.exp(torch.linspace(-8, 4, 17)).to(torch.bfloat16)
    counts = torch.arange(17, dtype=torch.bfloat16).reshape(1, 17) * 10
    value = counts * scale
    encoder = SignedQCFSSequenceEncoder(
        scale=scale,
        time_steps=160,
        neuron_backend="torch",
    )

    sequence = encoder.encode(value)

    expected = torch.round(value / scale).clamp(-160, 160) * scale
    torch.testing.assert_close(sequence.sum(0), expected, atol=0.0, rtol=0.0)


def test_qwen2_snn_forward_rejects_autograd_before_computation():
    model = Qwen2SNNModel.__new__(Qwen2SNNModel)
    nn.Module.__init__(model)
    model.eval()

    with pytest.raises(RuntimeError, match="does not support autograd"):
        model(torch.ones(1, 2, dtype=torch.long))


def test_qwen2_snn_generate_owns_inference_mode():
    class StubQwen2SNN(Qwen2SNNModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.grad_enabled = []
            self.eval()

        def forward(self, input_ids, **kwargs):
            del kwargs
            self.grad_enabled.append(torch.is_grad_enabled())
            logits = torch.zeros(*input_ids.shape, 3)
            logits[..., 2] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    model = StubQwen2SNN()
    generated = model.generate(torch.tensor([[0, 1]]), max_new_tokens=2)

    assert generated.tolist() == [[0, 1, 2, 2]]
    assert model.grad_enabled == [False, False]
