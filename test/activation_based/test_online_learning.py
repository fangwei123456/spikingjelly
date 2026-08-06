import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import base
from spikingjelly.activation_based.functional.online_learning import (
    fptt_online_training,
    fptt_online_training_init_w_ra,
)
from spikingjelly.activation_based.layer.online_learning import (
    GradwithTrace,
    OTTTSequential,
)
from spikingjelly.activation_based.neuron.online_learning import (
    OTTTLIFNode,
    SLTTLIFNode,
)


def test_online_lif_nodes_share_spike_dynamics_and_reset_state():
    ottt = OTTTLIFNode(tau=2.0, decay_input=False, v_threshold=0.5, v_reset=0.0)
    sltt = SLTTLIFNode(tau=2.0, decay_input=False, v_threshold=0.5, v_reset=0.0)
    inputs = torch.tensor([[0.4], [0.8], [0.2], [0.9]])

    trace = torch.zeros(1)
    for x in inputs:
        spike, actual_trace = ottt(x)
        torch.testing.assert_close(spike, sltt(x))
        trace = trace * 0.5 + spike
        torch.testing.assert_close(actual_trace, trace)

    ottt.reset()
    sltt.reset()
    assert ottt.v == 0.0
    assert ottt.trace is None
    assert sltt.v == 0.0

    ottt.eval()
    sltt.eval()
    for x in inputs:
        torch.testing.assert_close(ottt(x), sltt(x))


@pytest.mark.parametrize("node_type", [OTTTLIFNode, SLTTLIFNode])
def test_online_lif_nodes_support_functional_conversion(node_type):
    node = node_type()
    functional_forward = base.to_functional_forward(node)
    states = tuple(node._memories.values())

    outputs, updated_states = functional_forward((torch.zeros(1),), states)

    assert len(outputs) == (2 if node_type is OTTTLIFNode else 1)
    assert len(updated_states) == len(states)
    assert tuple(node._memories.values()) == states


def test_grad_with_trace_uses_spikes_for_values_and_traces_for_gradients():
    linear = nn.Linear(2, 1, bias=False)
    linear.weight.data.copy_(torch.tensor([[2.0, -1.0]]))
    spike = torch.tensor([[1.0, 0.0]], requires_grad=True)
    trace = torch.tensor([[0.25, 0.75]], requires_grad=True)

    output = GradwithTrace(linear)([spike, trace])
    torch.testing.assert_close(output, torch.tensor([[2.0]]))
    output.sum().backward()

    torch.testing.assert_close(linear.weight.grad, trace)
    torch.testing.assert_close(spike.grad, linear.weight.detach())
    torch.testing.assert_close(trace.grad, linear.weight.detach())


def test_ottt_sequential_keeps_the_public_trace_gradient_behavior():
    linear = nn.Linear(2, 1, bias=False)
    linear.weight.data.copy_(torch.tensor([[1.5, -0.5]]))
    model = OTTTSequential(nn.Identity(), linear)
    spike = torch.tensor([[1.0, 2.0]], requires_grad=True)
    trace = torch.tensor([[3.0, 4.0]], requires_grad=True)

    output = model([spike, trace])
    torch.testing.assert_close(output, linear(spike))
    output.sum().backward()

    torch.testing.assert_close(linear.weight.grad, trace)


def test_fptt_does_not_silently_skip_parameters_missing_from_w_ra():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(IndexError):
        fptt_online_training(
            model,
            optimizer,
            x_seq=torch.ones(1, 1, 2),
            target_seq=torch.ones(1, 1, 1),
            f_loss_t=nn.functional.mse_loss,
            alpha=0.1,
            w_ra=[],
        )


def test_fptt_running_average_starts_from_independent_parameter_snapshots():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshots = fptt_online_training_init_w_ra(optimizer)
    original = [snapshot.clone() for snapshot in snapshots]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    for snapshot, expected, parameter in zip(snapshots, original, model.parameters()):
        assert snapshot.data_ptr() != parameter.data_ptr()
        torch.testing.assert_close(snapshot, expected)
