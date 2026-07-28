import torch

from spikingjelly.activation_based import base
from spikingjelly.activation_based.encoding import PopEncoder, PopSpikeEncoderRandom
from spikingjelly.activation_based.layer.attention import (
    MultiDimensionalAttention,
    QKAttention,
    SpikingSelfAttention,
)


def test_population_encoder_matches_gaussian_reference_and_propagates_gradients():
    encoder = PopEncoder(
        obs_dim=2,
        pop_dim=3,
        spike_ts=4,
        mean_range=(-1.0, 1.0),
        std=0.5,
    )
    obs = torch.tensor([[0.0, 0.5]], requires_grad=True)

    output = encoder(obs)
    population = torch.exp(
        -0.5 * ((obs.view(-1, 2, 1) - encoder.mean) / encoder.std).square()
    ).flatten(1)
    expected = population.unsqueeze(0).repeat(4, 1, 1)

    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert obs.grad is not None
    assert encoder.mean.grad is not None
    assert encoder.std.grad is not None


def test_random_population_encoder_matches_per_step_sampling_order():
    encoder = PopSpikeEncoderRandom(
        obs_dim=2,
        pop_dim=3,
        spike_ts=4,
        mean_range=(-1.0, 1.0),
        std=0.5,
    )
    obs = torch.tensor([[0.0, 0.5]])
    population = torch.exp(
        -0.5 * ((obs.view(-1, 2, 1) - encoder.mean) / encoder.std).square()
    ).flatten(1)

    torch.manual_seed(7)
    expected = torch.stack([torch.bernoulli(population) for _ in range(4)])
    torch.manual_seed(7)
    actual = encoder(obs)

    torch.testing.assert_close(actual, expected)


def test_multidimensional_attention_matches_the_direct_three_stage_equation():
    torch.manual_seed(3)
    attention = MultiDimensionalAttention(
        T=4,
        C=6,
        reduction_t=2,
        reduction_c=3,
        kernel_size=3,
    )
    x = torch.randn(4, 2, 6, 5, 5)
    batch_first = x.transpose(0, 1)

    time_score = torch.sigmoid(
        attention.ta.sharedMLP(attention.ta.avg_pool(batch_first))
        + attention.ta.sharedMLP(attention.ta.max_pool(batch_first))
    )
    after_time = time_score * batch_first
    channel_input = after_time.transpose(1, 2)
    channel_score = torch.sigmoid(
        attention.ca.sharedMLP(attention.ca.avg_pool(channel_input))
        + attention.ca.sharedMLP(attention.ca.max_pool(channel_input))
    ).transpose(1, 2)
    after_channel = channel_score * after_time
    spatial_input = after_channel.flatten(1, 2)
    spatial_score = torch.sigmoid(
        attention.sa.conv(
            torch.cat(
                (
                    spatial_input.mean(dim=1, keepdim=True),
                    spatial_input.max(dim=1, keepdim=True).values,
                ),
                dim=1,
            )
        )
    ).unsqueeze(1)
    expected = torch.relu(spatial_score * after_channel).transpose(0, 1)

    torch.testing.assert_close(attention(x), expected)
    assert set(attention.state_dict()) == {
        "ta.sharedMLP.0.weight",
        "ta.sharedMLP.2.weight",
        "ca.sharedMLP.0.weight",
        "ca.sharedMLP.2.weight",
        "sa.conv.weight",
    }


def test_spiking_attention_backend_updates_all_internal_neurons(monkeypatch):
    monkeypatch.setattr(base, "check_backend_library", lambda _backend: None)

    ssa = SpikingSelfAttention(dim=8, num_heads=2)
    qka = QKAttention(dim=8, num_heads=2)
    for attention in (ssa, qka):
        attention.backend = "cupy"
        assert attention.backend == "cupy"
        assert attention.attn_lif.backend == "cupy"
        assert attention.proj_lif.backend == "cupy"
    assert ssa.qkv_lif.backend == "cupy"
    assert qka.qk_lif.backend == "cupy"
