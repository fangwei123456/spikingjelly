import random

import pytest
import torch

from spikingjelly.timing_based.neuron import Tempotron as NewTempotron
from spikingjelly.timing_based.orig_neuron import Tempotron as OriginalTempotron


def generate_test_cases(num_random_cases: int = 10) -> list:
    fixed_cases = [
        # (in_features, out_features, T, batch_size)
        (4, 3, 50, 1),
        (8, 5, 100, 4),
        (16, 10, 75, 8),
        (10, 8, 60, 2),
    ]
    random_cases = [
        (
            random.randint(1, 20),
            random.randint(1, 10),
            random.choice([50, 75, 100]),
            random.choice([1, 2, 4]),
        )
        for _ in range(num_random_cases)
    ]
    return fixed_cases + random_cases


REGRESSION_TEST_CASES = generate_test_cases(20)


@pytest.fixture
def setup_neurons(request):
    in_features, out_features, T, _ = request.param

    original_neuron = OriginalTempotron(
        in_features=in_features,
        out_features=out_features,
        T=T,
    )
    new_neuron = NewTempotron(in_features=in_features, out_features=out_features, T=T)

    # Fully Connected Layer Weights need to be initialized the same
    new_neuron.model.summation_layer.weight.data = (
        original_neuron.fc.weight.data.clone()
    )

    return original_neuron, new_neuron


class TestRegression:
    @pytest.mark.parametrize("setup_neurons", REGRESSION_TEST_CASES, indirect=True)
    def test_output_voltage_trace(self, setup_neurons):
        original_neuron, new_neuron = setup_neurons
        in_features = original_neuron.fc.in_features
        T = original_neuron.T
        _, _, _, batch_size = REGRESSION_TEST_CASES[0]

        input_spikes = (torch.rand(batch_size, in_features) * (T + 20)) - 10

        original_output = original_neuron(input_spikes, ret_type="v")
        new_output = new_neuron(input_spikes, ret_type="v")

        assert torch.allclose(original_output, new_output, atol=1e-6)

    @pytest.mark.parametrize("setup_neurons", REGRESSION_TEST_CASES, indirect=True)
    def test_output_max_voltage(self, setup_neurons):
        original_neuron, new_neuron = setup_neurons
        in_features = original_neuron.fc.in_features
        T = original_neuron.T
        _, _, _, batch_size = REGRESSION_TEST_CASES[0]

        input_spikes = torch.rand(batch_size, in_features) * T

        original_output = original_neuron(input_spikes, ret_type="v_max")
        new_output = new_neuron(input_spikes, ret_type="v_max")

        assert torch.allclose(original_output, new_output, atol=1e-6)

    @pytest.mark.parametrize("setup_neurons", REGRESSION_TEST_CASES, indirect=True)
    def test_output_spikes(self, setup_neurons):
        original_neuron, new_neuron = setup_neurons
        in_features = original_neuron.fc.in_features
        T = original_neuron.T
        _, _, _, batch_size = REGRESSION_TEST_CASES[0]
        input_spikes = torch.rand(batch_size, in_features) * T

        original_output = original_neuron(input_spikes, ret_type="spikes")
        new_output = new_neuron(input_spikes, ret_type="spikes")

        assert torch.allclose(original_output, new_output, atol=1e-4)

    @pytest.mark.parametrize("setup_neurons", REGRESSION_TEST_CASES, indirect=True)
    def test_mse_loss(self, setup_neurons):
        original_neuron, new_neuron = setup_neurons
        out_features = original_neuron.fc.out_features
        _, _, _, batch_size = REGRESSION_TEST_CASES[0]

        v_max = torch.rand(batch_size, out_features) * 2.0
        labels = torch.randint(0, out_features, (batch_size,))
        v_threshold = original_neuron.v_threshold

        original_loss = original_neuron.mse_loss(
            v_max, v_threshold, labels, out_features
        )
        new_loss = new_neuron.mse_loss(v_max, labels)

        assert torch.allclose(original_loss, new_loss, atol=1e-6)


class TestValidation:
    def test_invalid_ret_type(self):
        neuron = NewTempotron(1, 1, 100)
        with pytest.raises(ValueError, match="Invalid out_voltage_type"):
            neuron(torch.tensor([[10.0]]), ret_type="invalid_string")

    def test_input_shape_mismatch(self):
        neuron = NewTempotron(in_features=5, out_features=1, T=100)
        wrong_input = torch.rand(1, 3) * 100
        with pytest.raises(RuntimeError):
            neuron(wrong_input, ret_type="v")
