import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import rnn


class _AccumulatingCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, state):
        return state + x[..., : self.hidden_size]


class _AccumulatingRNN(rnn.SpikingRNNBase):
    @staticmethod
    def base_cell():
        return _AccumulatingCell

    @staticmethod
    def states_num():
        return 1


class _TwoStateCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, states):
        hidden, memory = states
        return torch.stack((hidden + x[..., : self.hidden_size], memory + 1))


class _TwoStateRNN(rnn.SpikingRNNBase):
    @staticmethod
    def base_cell():
        return _TwoStateCell

    @staticmethod
    def states_num():
        return 2


@pytest.mark.parametrize("bidirectional", [False, True])
def test_spiking_rnn_preserves_initial_state_for_every_layer(bidirectional):
    model = _AccumulatingRNN(
        input_size=1,
        hidden_size=1,
        num_layers=2,
        bidirectional=bidirectional,
    )
    x = torch.zeros(3, 1, 1)
    if bidirectional:
        states = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1)
        expected = torch.tensor([1.0, 5.0, 3.0, 7.0]).reshape(4, 1, 1)
    else:
        states = torch.tensor([1.0, 2.0]).reshape(2, 1, 1)
        expected = torch.tensor([1.0, 5.0]).reshape(2, 1, 1)

    _, final_states = model(x, states)

    torch.testing.assert_close(final_states, expected)


def test_spiking_rnn_preserves_multiple_state_tensors():
    model = _TwoStateRNN(input_size=1, hidden_size=1, num_layers=2)
    x = torch.zeros(3, 1, 1)
    states = (
        torch.tensor([1.0, 2.0]).reshape(2, 1, 1),
        torch.tensor([10.0, 20.0]).reshape(2, 1, 1),
    )

    _, final_states = model(x, states)

    assert isinstance(final_states, tuple)
    torch.testing.assert_close(
        final_states[0], torch.tensor([1.0, 5.0]).reshape(2, 1, 1)
    )
    torch.testing.assert_close(
        final_states[1], torch.tensor([13.0, 23.0]).reshape(2, 1, 1)
    )


@pytest.mark.parametrize("invariant_dropout_mask", [False, True])
def test_spiking_rnn_applies_dropout_between_layers(invariant_dropout_mask):
    model = _AccumulatingRNN(
        input_size=1,
        hidden_size=1,
        num_layers=2,
        dropout_p=1.0,
        invariant_dropout_mask=invariant_dropout_mask,
    )
    model.train()

    output, final_states = model(torch.ones(3, 1, 1))

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(final_states[1], torch.zeros_like(final_states[1]))


@pytest.mark.parametrize(
    ("rnn_type", "bidirectional"),
    [
        (rnn.SpikingVanillaRNN, False),
        (rnn.SpikingGRU, False),
        (rnn.SpikingLSTM, True),
    ],
)
def test_spiking_rnn_public_types_forward_and_backward(rnn_type, bidirectional):
    model = rnn_type(
        input_size=3,
        hidden_size=2,
        num_layers=2,
        bidirectional=bidirectional,
    )
    x = torch.randn(4, 2, 3, requires_grad=True)

    output, states = model(x)
    state_tensors = states if isinstance(states, tuple) else (states,)
    loss = output.sum() + sum(state.sum() for state in state_tensors)
    loss.backward()

    directions = 2 if bidirectional else 1
    assert output.shape == (4, 2, 2 * directions)
    assert all(state.shape == (2 * directions, 2, 2) for state in state_tensors)
    assert x.grad is not None
