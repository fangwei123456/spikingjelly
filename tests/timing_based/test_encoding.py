import random

import pytest
import torch

from spikingjelly.timing_based.encoding import GaussianTuning as NewGaussianTuning
from spikingjelly.timing_based.orig_encoding import (
    GaussianTuning as OriginalGaussianTuning,
)


def generate_regression_test_cases(num_random_cases: int = 10) -> list:
    fixed_cases = [
        (1, 3, 1, 4, 50),
        (2, 9, 4, 4, 100),
        (3, 15, 8, 10, 75),
        (5, 20, 16, 5, 60),
    ]

    random_cases = list()
    for _ in range(num_random_cases):
        random_cases.append(
            (
                random.randint(1, 10),  # num_dims
                random.randint(3, 20),  # neurons_per_dim
                random.randint(1, 16),  # batch_size
                random.randint(1, 10),  # num_samples
                random.choice([50, 75, 100]),  # max_spike_time
            )
        )
    return fixed_cases + random_cases


REGRESSION_TEST_CASES = generate_regression_test_cases(50)


class TestValidation:
    def test_invalid_neuron_count_raises_error(self):
        with pytest.raises(ValueError, match="Input should be greater than 2"):
            NewGaussianTuning(n=1, m=2, x_min=torch.zeros(1), x_max=torch.ones(1))

    def test_invalid_min_max_range_raises_error(self):
        with pytest.raises(
            ValueError, match="All elements of input_min must be less than input_max"
        ):
            NewGaussianTuning(n=1, m=5, x_min=torch.ones(1), x_max=torch.zeros(1))

    def test_mismatched_min_max_shapes_raise_error(self):
        with pytest.raises(
            ValueError, match="input_min and input_max should have shape"
        ):
            NewGaussianTuning(n=2, m=5, x_min=torch.zeros(1), x_max=torch.ones(2))

    def test_invalid_encode_input_shape_raises_error(self):
        encoder = NewGaussianTuning(n=2, m=5, x_min=torch.zeros(2), x_max=torch.ones(2))

        with pytest.raises(
            AssertionError, match="Input tensor x must be 3-dimensional"
        ):
            encoder.encode(torch.rand(2, 5))  # 2D input

        with pytest.raises(
            AssertionError, match="Input tensor x must be 3-dimensional"
        ):
            encoder.encode(torch.rand(4, 3, 10))  # 3 channels instead of 2


class TestBehavior:
    @pytest.fixture(scope="class")
    def encoder(self):
        return NewGaussianTuning(
            n=2,
            m=5,
            x_min=torch.tensor([0.0, -1.0]),
            x_max=torch.tensor([1.0, 1.0]),
        )

    def test_center_values_yield_zero_spike_time(self, encoder):
        center_input = encoder.mu[:, 2].unsqueeze(0).unsqueeze(2)  # Shape: (1, n, 1)
        spikes = encoder.encode(center_input, max_spike_time=100)
        assert torch.all(
            spikes[:, :, :, 2] == 0
        ), "Spike time at neuron centers must be 0"

    def test_far_out_of_range_values_are_inactive(self, encoder):
        far_input = torch.tensor([[[-100.0], [100.0]]])  # Shape: (1, 2, 1)
        spikes = encoder.encode(far_input, max_spike_time=100)
        inactive_ratio = (spikes == -1).float().mean()
        assert (
            inactive_ratio > 0.95
        ), "Expected most neurons to be inactive for out-of-range inputs"


class TestRegression:
    @pytest.mark.parametrize(
        "num_dims, neurons_per_dim, batch_size, num_samples, max_spike_time",
        REGRESSION_TEST_CASES,
    )
    def test_equivalence_with_original(
        self,
        num_dims,
        neurons_per_dim,
        batch_size,
        num_samples,
        max_spike_time,
    ):
        x_min = torch.zeros(num_dims)
        x_max = torch.ones(num_dims)
        original_encoder = OriginalGaussianTuning(
            n=num_dims, m=neurons_per_dim, x_min=x_min, x_max=x_max
        )
        new_encoder = NewGaussianTuning(
            n=num_dims, m=neurons_per_dim, x_min=x_min, x_max=x_max
        )

        assert torch.allclose(
            original_encoder.mu, new_encoder.mu
        ), "Internal 'mu' attribute mismatched"
        assert torch.allclose(
            original_encoder.sigma2, new_encoder.sigma2
        ), "Internal 'sigma2' attribute mismatched"
        assert original_encoder.n == new_encoder.n, "Internal 'n' attribute mismatched"
        assert original_encoder.m == new_encoder.m, "Internal 'm' attribute mismatched"

        input_tensor = torch.rand(batch_size, num_dims, num_samples)
        out_spikes_orig = original_encoder.encode(
            input_tensor, max_spike_time=max_spike_time
        )
        out_spikes_new = new_encoder.encode(input_tensor, max_spike_time=max_spike_time)

        assert torch.equal(
            out_spikes_orig, out_spikes_new
        ), "encode() output mismatched"

    def test_edge_case_min_max_values(self):
        x_min = torch.tensor([-1.0, -1.0])
        x_max = torch.tensor([1.0, 1.0])
        original_encoder = OriginalGaussianTuning(n=2, m=5, x_min=x_min, x_max=x_max)
        new_encoder = NewGaussianTuning(n=2, m=5, x_min=x_min, x_max=x_max)
        input_tensor = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]).unsqueeze(0)
        out_spikes_orig = original_encoder.encode(input_tensor, max_spike_time=100)
        out_spikes_new = new_encoder.encode(input_tensor, max_spike_time=100)
        assert torch.equal(
            out_spikes_orig, out_spikes_new
        ), "encode() output mismatched for edge min/max values"
