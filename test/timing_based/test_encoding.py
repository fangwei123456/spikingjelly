import pytest
import torch

from spikingjelly.timing_based.encoding import GaussianTuning


def _reference_spike_times(
    encoder: GaussianTuning, x: torch.Tensor, max_spike_time: int
) -> torch.Tensor:
    spikes = torch.empty(*x.shape, encoder.m, dtype=x.dtype, device=x.device)
    for batch in range(x.shape[0]):
        for feature in range(encoder.n):
            for sample in range(x.shape[2]):
                for neuron in range(encoder.m):
                    distance = x[batch, feature, sample] - encoder.mu[feature, neuron]
                    response = torch.exp(
                        -distance.square() / (2 * encoder.sigma2[feature, neuron])
                    )
                    spike_time = torch.round(max_spike_time * (1 - response))
                    spikes[batch, feature, sample, neuron] = (
                        -1 if spike_time >= max_spike_time else spike_time
                    )
    return spikes


def test_receptive_fields_match_reference_values():
    encoder = GaussianTuning(
        n=1,
        m=4,
        x_min=torch.tensor([0.0]),
        x_max=torch.tensor([1.0]),
    )

    torch.testing.assert_close(encoder.mu, torch.tensor([[-0.25, 0.25, 0.75, 1.25]]))
    torch.testing.assert_close(encoder.sigma2, torch.full((1, 4), 1.0 / 9.0))
    assert encoder.n == 1
    assert encoder.m == 4


def test_encode_produces_expected_values():
    encoder = GaussianTuning(
        n=1,
        m=4,
        x_min=torch.tensor([0.0]),
        x_max=torch.tensor([1.0]),
    )
    x = torch.tensor([[[0.1, 0.5, 0.9]]])

    spikes = encoder.encode(x, max_spike_time=100)

    expected = torch.tensor(
        [
            [
                [
                    [42.0, 10.0, 85.0, -1.0],
                    [92.0, 25.0, 25.0, 92.0],
                    [-1.0, 85.0, 10.0, 42.0],
                ]
            ]
        ]
    )
    torch.testing.assert_close(spikes, expected)


@pytest.mark.parametrize(
    ("n", "m", "x_min", "x_max", "dtype"),
    [
        (1, 4, [0.0], [1.0], torch.float32),
        (2, 5, [-2.0, 1.0], [3.0, 4.0], torch.float32),
        (3, 7, [-1.0, 0.0, 2.0], [1.0, 5.0, 9.0], torch.float64),
    ],
)
def test_encode_matches_loop_reference(n, m, x_min, x_max, dtype):
    torch.manual_seed(0)
    encoder = GaussianTuning(
        n=n,
        m=m,
        x_min=torch.tensor(x_min, dtype=dtype),
        x_max=torch.tensor(x_max, dtype=dtype),
    )
    x = torch.randn(3, n, 4, dtype=dtype)

    actual = encoder.encode(x, max_spike_time=37)

    torch.testing.assert_close(actual, _reference_spike_times(encoder, x, 37))


def test_center_values_spike_at_zero():
    encoder = GaussianTuning(
        n=2,
        m=5,
        x_min=torch.tensor([0.0, -1.0]),
        x_max=torch.tensor([1.0, 1.0]),
    )
    center_input = encoder.mu[:, 2].view(1, 2, 1)

    spikes = encoder.encode(center_input, max_spike_time=100)

    assert torch.all(spikes[..., 2] == 0)


def test_far_out_of_range_values_are_inactive():
    encoder = GaussianTuning(
        n=2,
        m=5,
        x_min=torch.tensor([0.0, -1.0]),
        x_max=torch.tensor([1.0, 1.0]),
    )

    spikes = encoder.encode(torch.tensor([[[-100.0], [100.0]]]), max_spike_time=100)

    assert (spikes == -1).float().mean() > 0.95


def test_zero_max_spike_time_preserves_default_fallback():
    encoder = GaussianTuning(
        n=1,
        m=4,
        x_min=torch.tensor([0.0]),
        x_max=torch.tensor([1.0]),
    )
    x = torch.tensor([[[0.1, 0.5, 0.9]]])

    assert torch.equal(
        encoder.encode(x, max_spike_time=0),
        encoder.encode(x, max_spike_time=100),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "n": 0,
                "m": 5,
                "x_min": torch.zeros(0),
                "x_max": torch.ones(0),
            },
            "n must be a positive integer",
        ),
        (
            {
                "n": 1,
                "m": 2,
                "x_min": torch.zeros(1),
                "x_max": torch.ones(1),
            },
            "m must be greater than 2",
        ),
        (
            {
                "n": 1,
                "m": 5,
                "x_min": torch.ones(1),
                "x_max": torch.zeros(1),
            },
            "x_min must be less than x_max",
        ),
        (
            {
                "n": 2,
                "m": 5,
                "x_min": torch.zeros(1),
                "x_max": torch.ones(2),
            },
            "must both have shape",
        ),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GaussianTuning(**kwargs)


def test_encode_rejects_invalid_input_shape():
    encoder = GaussianTuning(
        n=2,
        m=5,
        x_min=torch.zeros(2),
        x_max=torch.ones(2),
    )

    with pytest.raises(AssertionError, match="must have shape"):
        encoder.encode(torch.rand(2, 5))
    with pytest.raises(AssertionError, match=r"shape \[batch_size, 2, samples_count\]"):
        encoder.encode(torch.rand(4, 3, 10))
