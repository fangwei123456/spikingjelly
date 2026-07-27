import pytest
import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn.recipes.spikezip_qann import (
    SpikeZIPConv2d,
    SpikeZIPEmbedding,
    SpikeZIPLinear,
)
from spikingjelly.activation_based.ann2snn.recipes.sta_transformer import (
    _STAConstant,
    _STASpikeEncoder,
)


def _temporal_difference(y_cum: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(y_cum)
    y[0] = y_cum[0]
    y[1:] = y_cum[1:] - y_cum[:-1]
    return y


def test_sta_spike_encoder_single_step_matches_module_and_state_reset():
    threshold = torch.tensor([0.25, 0.5, 1.0]).view(1, 3, 1)
    x0 = torch.tensor([[[0.3, -0.4], [0.6, 0.1], [1.8, -1.2]]])
    x1 = torch.tensor([[[0.2, -0.2], [-0.9, 0.8], [0.4, 0.5]]])
    mem = torch.zeros_like(x0)

    spike0, mem = functional.sta_spike_encoder_single_step(x0, mem, threshold)
    spike1, mem = functional.sta_spike_encoder_single_step(x1, mem, threshold)

    module = _STASpikeEncoder(torch.tensor([0.25, 0.5, 1.0]), channel_dim=1)
    y0 = module(x0)
    y1 = module(x1)

    assert torch.allclose(y0, spike0)
    assert torch.allclose(y1, spike1)
    assert torch.allclose(module.mem, mem)

    x_changed = torch.randn(2, 3)
    module(x_changed)
    assert module.mem.shape == x_changed.shape


def test_sta_spike_encoder_multi_step_matches_single_step_loop():
    x_seq = torch.randn(4, 2, 3)
    threshold = torch.full_like(x_seq[0], 0.5)
    mem = torch.zeros_like(x_seq[0])
    expected = []
    for x in x_seq:
        spike, mem = functional.sta_spike_encoder_single_step(x, mem, threshold)
        expected.append(spike)

    module = _STASpikeEncoder(torch.tensor(0.5), step_mode="m")
    actual = module(x_seq)

    assert torch.allclose(actual, torch.stack(expected))
    assert torch.allclose(module.mem, mem)


def test_sta_constant_single_and_multi_step_match_module():
    value = torch.tensor([1.0, -2.0])

    y0, t = functional.sta_constant_single_step(value, 0)
    y1, t = functional.sta_constant_single_step(value, t)
    assert y0 is value
    assert torch.equal(y1, torch.zeros_like(value))
    assert t == 2

    seq0, t = functional.sta_constant_multi_step(value, 0, 4)
    seq1, t = functional.sta_constant_multi_step(value, t, 4)

    module = _STAConstant(value, time_steps=4, step_mode="m")
    assert torch.equal(module(), seq0)
    assert module.t == 4
    assert torch.equal(module(), seq1)
    assert module.t == 8

    with pytest.raises(ValueError, match="time_steps must be positive"):
        functional.sta_constant_multi_step(value, 0, 0)


def test_spikezip_bias_release_helpers_update_explicit_state():
    y = torch.randn(2, 3)
    bias = torch.tensor([0.5, -1.0, 1.5])

    y0, realize_time, released = functional.spikezip_release_bias_single_step(
        y,
        bias,
        realize_time=2,
        bias_steps=4,
        bias_view_shape=(3,),
    )
    assert released
    assert realize_time == 1
    assert torch.allclose(y0, y + bias / 4)

    y_seq = torch.randn(5, 2, 3)
    y_seq_original = y_seq.clone()
    y_seq_next, realize_time, released_steps = (
        functional.spikezip_release_bias_multi_step(
            y_seq,
            bias,
            realize_time=3,
            bias_steps=4,
            bias_view_shape=(1, 1, 3),
        )
    )
    expected = y_seq.clone()
    expected[:3] = expected[:3] + bias.view(1, 1, 3) / 4
    assert released_steps == 3
    assert realize_time == 0
    assert torch.allclose(y_seq_next, expected)
    assert torch.allclose(y_seq, y_seq_original)


def test_spikezip_embedding_helpers_match_module_state():
    embedding = torch.nn.Embedding(5, 3)
    x0 = torch.tensor([[1, 2], [3, 4]])
    x1 = torch.tensor([[0, 1], [2, 3]])

    y0, t = functional.spikezip_embedding_single_step(
        x0,
        0,
        embedding,
        embedding.embedding_dim,
        embedding.weight.dtype,
    )
    y1, t = functional.spikezip_embedding_single_step(
        x1,
        t,
        embedding,
        embedding.embedding_dim,
        embedding.weight.dtype,
    )

    module = SpikeZIPEmbedding(embedding)
    assert torch.allclose(module(x0), y0)
    assert torch.allclose(module(x1), y1)
    assert module.t == t

    module.reset()
    module.step_mode = "m"
    x_seq = torch.stack((x0, x1, x0))
    y_seq, t = functional.spikezip_embedding_multi_step(x_seq, embedding)
    assert torch.allclose(module(x_seq), y_seq)
    assert module.t == t
    assert torch.allclose(module(x_seq), y_seq)
    assert module.t == t


def test_spikezip_bias_free_multi_step_preserves_scheduled_work_state():
    linear = SpikeZIPLinear(
        torch.nn.Linear(3, 2, bias=False),
        level=4,
        bias_steps=3,
        step_mode="m",
    )
    linear_out = linear(torch.zeros(2, 1, 3))
    assert torch.equal(linear_out, torch.zeros_like(linear_out))
    assert linear.is_work
    assert linear.realize_time == 3

    conv = SpikeZIPConv2d(
        torch.nn.Conv2d(2, 3, kernel_size=1, bias=False),
        level=4,
        bias_steps=3,
        step_mode="m",
    )
    conv_out = conv(torch.zeros(2, 1, 2, 4, 4))
    assert torch.equal(conv_out, torch.zeros_like(conv_out))
    assert conv.is_work
    assert conv.realize_time == 3


def test_spikezip_matmul_delta_helpers_match_cumulative_difference():
    a_seq = torch.randn(4, 2, 3, 5)
    b_seq = torch.randn(4, 2, 7, 5)
    a_sum = a_seq.cumsum(0)
    b_sum = b_seq.cumsum(0)

    y0 = functional.spikezip_matmul_delta(
        a_seq[0],
        b_seq[0],
        a_sum[0],
        b_sum[0],
        transpose_b=True,
    )
    expected0 = a_sum[0] @ b_sum[0].transpose(-2, -1)
    assert torch.allclose(y0, expected0, atol=1e-6, rtol=1e-6)

    y_seq = functional.spikezip_matmul_sequence_delta(
        a_seq,
        b_seq,
        transpose_b=True,
    )
    expected_seq = _temporal_difference(a_sum @ b_sum.transpose(-2, -1))
    assert torch.allclose(y_seq, expected_seq, atol=1e-6, rtol=1e-6)


def test_spikezip_vit_token_helpers_emit_first_step_only():
    cls_token = torch.randn(1, 1, 4)
    pos_embed = torch.randn(1, 5, 4)

    cls0, pos0, t = functional.spikezip_vit_single_tokens(
        cls_token,
        pos_embed,
        batch_size=2,
        t=0,
    )
    cls1, pos1, t = functional.spikezip_vit_single_tokens(
        cls_token,
        pos_embed,
        batch_size=2,
        t=t,
    )
    assert torch.allclose(cls0, cls_token.expand(2, -1, -1))
    assert pos0 is pos_embed
    assert torch.equal(cls1, torch.zeros_like(cls0))
    assert torch.equal(pos1, torch.zeros_like(pos_embed))
    assert t == 2

    cls_seq, pos_seq = functional.spikezip_vit_token_sequence(
        cls_token,
        pos_embed,
        time_steps=3,
        batch_size=2,
    )
    assert torch.allclose(cls_seq[0], cls0)
    assert torch.equal(cls_seq[1:], torch.zeros_like(cls_seq[1:]))
    assert torch.allclose(pos_seq[0], pos_embed)
    assert torch.equal(pos_seq[1:], torch.zeros_like(pos_seq[1:]))
