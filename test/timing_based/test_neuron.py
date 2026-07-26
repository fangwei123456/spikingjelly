import pytest
import torch
import torch.nn.functional as F

from spikingjelly.timing_based.neuron import Tempotron


def _reference_voltage(neuron: Tempotron, in_spikes: torch.Tensor) -> torch.Tensor:
    voltage = []
    for time_step in range(neuron.T):
        delta = time_step - in_spikes
        active = (delta >= 0) & (in_spikes >= 0)
        psp = torch.where(
            active,
            torch.exp(-delta / neuron.tau) - torch.exp(-delta / neuron.tau_s),
            0.0,
        )
        voltage.append(
            F.linear(
                neuron.v0 * psp,
                neuron.model.summation_layer.weight,
            )
        )
    return torch.stack(voltage, dim=-1)


@pytest.mark.parametrize(
    ("in_features", "out_features", "T", "batch_size", "dtype"),
    [
        (4, 3, 50, 1, torch.float32),
        (8, 5, 20, 4, torch.float32),
        (1, 1, 10, 2, torch.float32),
        (3, 7, 101, 5, torch.float64),
        (11, 2, 7, 3, torch.float64),
    ],
)
def test_voltage_trace_matches_reference(
    in_features, out_features, T, batch_size, dtype
):
    torch.manual_seed(0)
    neuron = Tempotron(in_features, out_features, T).to(dtype)
    in_spikes = torch.rand(batch_size, in_features, dtype=dtype) * (T + 4) - 2

    actual = neuron(in_spikes, ret_type="v")

    torch.testing.assert_close(actual, _reference_voltage(neuron, in_spikes))


def test_voltage_views_are_consistent():
    torch.manual_seed(1)
    neuron = Tempotron(4, 3, 20)
    in_spikes = torch.rand(2, 4) * 20

    voltage = neuron(in_spikes, ret_type="v")
    v_max = neuron(in_spikes, ret_type="v_max")

    torch.testing.assert_close(v_max, voltage.max(dim=2).values)


@pytest.mark.parametrize(("batch_size", "out_features"), [(1, 1), (4, 1), (1, 3)])
def test_max_voltage_preserves_batch_and_output_dimensions(batch_size, out_features):
    neuron = Tempotron(2, out_features, 10)

    v_max = neuron(torch.rand(batch_size, 2) * 10, ret_type="v_max")

    assert v_max.shape == (batch_size, out_features)


def test_spike_times_match_reference_interpretation():
    torch.manual_seed(2)
    neuron = Tempotron(3, 2, 12)
    in_spikes = torch.rand(2, 3) * 12
    voltage = neuron(in_spikes, ret_type="v")
    times = torch.arange(12, dtype=voltage.dtype).view(1, 1, 12)
    hard_index = voltage.argmax(dim=2)
    soft_index = (F.softmax(voltage * 12, dim=2) * times).sum(dim=2)
    sign = (voltage.max(dim=2).values >= neuron.v_threshold).to(voltage.dtype) * 2 - 1
    expected = soft_index * sign + (hard_index * sign - soft_index * sign).detach()

    actual = neuron(in_spikes, ret_type="spikes")

    torch.testing.assert_close(actual, expected)


def test_single_output_spike_times_keep_output_dimension():
    neuron = Tempotron(1, 1, 10)

    assert neuron(torch.rand(4, 1) * 10, ret_type="spikes").shape == (4, 1)


def test_spike_times_keep_soft_gradient():
    neuron = Tempotron(3, 2, 12)
    in_spikes = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]])

    neuron(in_spikes, ret_type="spikes").sum().backward()

    grad = neuron.model.summation_layer.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad)


def test_mse_loss_uses_only_misclassified_outputs():
    neuron = Tempotron(2, 3, 10, v_threshold=1.0)
    v_max = torch.tensor([[1.5, 0.5, 0.5], [0.5, 1.2, 1.4]])
    labels = torch.tensor([0, 1])

    loss = neuron.mse_loss(v_max, labels)

    expected = (0.4**2) / 2
    assert loss.item() == pytest.approx(expected)


def test_state_dict_preserves_tempotron_parameter_name():
    neuron = Tempotron(2, 3, 10)

    assert set(neuron.state_dict()) == {"model.summation_layer.weight"}


def test_invalid_ret_type():
    neuron = Tempotron(1, 1, 10)

    with pytest.raises(ValueError, match="Invalid out_voltage_type"):
        neuron(torch.tensor([[1.0]]), ret_type="invalid")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_features": 0, "out_features": 1, "T": 10}, "must be positive"),
        ({"in_features": 1, "out_features": 1, "T": 0}, "must be positive"),
        ({"in_features": 1, "out_features": 1, "T": 10, "tau": 0}, "must be positive"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Tempotron(**kwargs)


def test_input_shape_mismatch():
    neuron = Tempotron(5, 1, 10)

    with pytest.raises(RuntimeError):
        neuron(torch.rand(1, 3), ret_type="v")
