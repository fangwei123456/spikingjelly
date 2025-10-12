import os
import sys

import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from spikingjelly.timing_based.encoding import GaussianTuning
from src.coders.sparse_ttfs import SparseTTFS

# Each tuple represents one test case:
# (num_dims, neurons_per_dim, batch_size, num_samples, beta, max_spike_time)
TEST_CASES = [
    (1, 3, 1, 4, 50),
    (2, 9, 1, 4, 100),
    (2, 9, 4, 4, 100),
    (3, 15, 8, 10, 75),
    (5, 20, 16, 5, 60),
    (10, 30, 32, 2, 50),
]


@pytest.mark.parametrize(
    "num_dims, neurons_per_dim, batch_size, num_samples, max_spike_time",
    TEST_CASES,
)
def test_encoder_outputs_are_identical(
    num_dims,
    neurons_per_dim,
    batch_size,
    num_samples,
    max_spike_time,
):
    x_min = torch.zeros(num_dims)
    x_max = torch.ones(num_dims)

    original_encoder = GaussianTuning(
        n=num_dims,
        m=neurons_per_dim,
        x_min=x_min,
        x_max=x_max,
    )

    new_encoder = SparseTTFS(
        {
            "channels_count": num_dims,
            "input_min": x_min,
            "input_max": x_max,
            "neurons_count": neurons_per_dim,
            "max_spike_time": max_spike_time,
            "beta": 1.5,  # hardcoded beta to match GaussianTuning behavior
        }
    )

    input_tensor = torch.rand(batch_size, num_dims, num_samples)

    out_spikes_orig = original_encoder.encode(
        input_tensor,
        max_spike_time=max_spike_time,
    )
    out_spikes_new = new_encoder.encode(input_tensor)

    assert torch.equal(out_spikes_orig, out_spikes_new), (
        "Outputs mismatched for"
        f"config: dims={num_dims}, neurons={neurons_per_dim}, batch={batch_size}"
    )
