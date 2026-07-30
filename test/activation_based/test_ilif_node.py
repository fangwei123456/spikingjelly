import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate


def test_ilif_rejects_invalid_max_spike_count():
    for value in (0, -1):
        with pytest.raises(ValueError):
            neuron.ILIFNode(max_spike_count=value)
    for value in (1.5, True):
        with pytest.raises(TypeError):
            neuron.ILIFNode(max_spike_count=value)


def test_ilif_training_outputs_integer_counts_and_updates_voltage():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25)
    x = torch.tensor([[3.2, 0.4, 5.8]])

    y = node(x)

    expected = torch.tensor([[3.0, 0.0, 4.0]])
    assert torch.equal(y, expected)
    torch.testing.assert_close(node.v, x - expected)


def test_ilif_uses_spike_count_surrogate():
    node = neuron.ILIFNode(max_spike_count=4)

    assert isinstance(node.surrogate_function, surrogate.MultiLevelSpikeCount)
    assert node.surrogate_function.max_spike_count == 4


def test_ilif_decay_is_applied_during_charge():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25)

    node(torch.tensor([[3.2]]))
    y = node(torch.tensor([[0.0]]))

    assert torch.equal(y, torch.tensor([[0.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.05]]))


def test_ilif_eval_releases_only_binary_spikes_over_user_virtual_steps():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=1.0).eval()
    x_seq = torch.tensor([[[1.2]], [[1.2]], [[1.2]], [[0.0]]])

    y = torch.stack([node(x) for x in x_seq])

    assert torch.equal(y.flatten(), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(node.v, torch.tensor([[0.6]]))


def test_ilif_eval_does_not_return_integer_counts():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25).eval()

    y = node(torch.tensor([[3.2, 5.8]]))

    assert torch.equal(y, torch.ones_like(y))
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_ilif_multistep_uses_user_provided_virtual_length_without_expansion():
    node = neuron.ILIFNode(
        v_threshold=1.0,
        max_spike_count=4,
        decay=0.25,
        step_mode="m",
        store_v_seq=True,
    ).eval()
    x_seq = torch.tensor([[[1.2]], [[1.2]], [[1.2]], [[0.0]]])

    y = node(x_seq)

    assert y.shape == x_seq.shape
    assert node.v_seq.shape == x_seq.shape
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_ilif_training_straight_through_gradient_is_windowed():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.0)
    x = torch.tensor([[-0.5, 0.5, 4.5]], requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 0.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[0.0, 1.0, 0.0]]))


def test_multi_level_spike_count_supports_custom_gradient_window():
    spike_count = surrogate.MultiLevelSpikeCount(
        max_spike_count=4,
        grad_window=(-0.5, 4.5),
    )
    x = torch.tensor([[-0.25, 4.25, 4.75]], requires_grad=True)

    y = spike_count(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 4.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))


def test_multi_level_spike_count_rejects_invalid_parameters():
    for value in (0, -1):
        with pytest.raises(ValueError):
            surrogate.MultiLevelSpikeCount(value)
    for value in (1.5, False):
        with pytest.raises(TypeError):
            surrogate.MultiLevelSpikeCount(value)
    with pytest.raises(ValueError):
        surrogate.MultiLevelSpikeCount(4, grad_window=(1.0, 0.0))
