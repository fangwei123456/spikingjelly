import copy
import importlib.util

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import base, layer
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.functional.net_config import detach_net, reset_net
from spikingjelly.activation_based.learning.online_learning import (
    fptt_online_training,
    fptt_online_training_init_w_ra,
    ottt_online_training,
)
from spikingjelly.activation_based.layer.online_learning import (
    GradwithTrace,
    OTTTSequential,
)
from spikingjelly.activation_based.neuron.online_learning import (
    OTTTLIFNode,
    SLTTLIFNode,
)
from spikingjelly.activation_based.model.spiking_vggws_ottt import OTTTSpikingVGG


def test_learning_public_locations_preserve_plasticity_and_remove_old_training_paths():
    from spikingjelly.activation_based import functional, learning

    for name in learning.plasticity.__all__:
        assert getattr(learning, name) is getattr(learning.plasticity, name)
    for name in (
        "stdp_linear_single_step",
        "mstdp_linear_single_step",
        "mstdpet_linear_single_step",
        "stdp_conv1d_single_step",
        "stdp_conv2d_single_step",
        "stdp_multi_step",
    ):
        assert not hasattr(learning, name)
        assert not hasattr(learning.plasticity, name)
    for name in learning.online_learning.__all__:
        assert not hasattr(functional, name)
    assert (
        importlib.util.find_spec(
            "spikingjelly.activation_based.functional.online_learning"
        )
        is None
    )
    for name in ("stdp_trace", "mstdp", "mstdpet", "stdp_mnist"):
        assert (
            importlib.util.find_spec(f"spikingjelly.activation_based.examples.{name}")
            is None
        )
        assert importlib.util.find_spec(
            f"spikingjelly.activation_based.examples.plasticity.{name}"
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
def test_online_lif_nodes_accumulate_v_seq_in_single_step_mode(node_type):
    node = node_type(store_v_seq=True)
    x_seq = torch.rand(3, 2)
    v_steps = []
    for x in x_seq:
        node(x)
        v_steps.append(node.v)
    torch.testing.assert_close(node.v_seq, torch.stack(v_steps))
    node.reset()
    assert node.v_seq is None


def test_ottt_spiking_vgg_applies_requested_dropout_rate():
    model = OTTTSpikingVGG(
        cfg=[4],
        num_classes=2,
        init_weights=False,
        spiking_neuron=OTTTLIFNode,
        drop_rate=0.25,
    )

    dropouts = [
        module for module in model.features if isinstance(module, layer.Dropout)
    ]
    assert len(dropouts) == 1
    assert dropouts[0].p == 0.25


@pytest.mark.parametrize("node_type", [OTTTLIFNode, SLTTLIFNode])
def test_online_lif_nodes_support_functional_conversion(node_type):
    node = node_type()
    functional_forward = base.to_functional_forward(node)
    states = tuple(node._memories.values())

    outputs, updated_states = functional_forward((torch.zeros(1),), states)

    assert len(outputs) == (2 if node_type is OTTTLIFNode else 1)
    assert len(updated_states) == len(states)
    assert tuple(node._memories.values()) == states


@pytest.mark.parametrize("linear_type", [nn.Linear, layer.WSLinear])
def test_grad_with_trace_uses_spikes_for_values_and_traces_for_gradients(linear_type):
    linear = linear_type(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    reference = copy.deepcopy(linear)
    spike = torch.tensor([[1.0, 0.0]], requires_grad=True)
    trace = torch.tensor([[0.25, 0.75]], requires_grad=True)
    reference_trace = trace.detach().clone().requires_grad_()

    output = GradwithTrace(linear)([spike, trace])
    torch.testing.assert_close(output, reference(spike))
    output.sum().backward()
    reference(reference_trace).sum().backward()

    for actual, expected in zip(linear.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(spike.grad, reference_trace.grad)
    torch.testing.assert_close(trace.grad, reference_trace.grad)


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


@pytest.mark.parametrize("online", [True, False])
@pytest.mark.parametrize("node_type", [OTTTLIFNode, SLTTLIFNode, neuron.LIFNode])
def test_online_training_matches_manual_updates(online, node_type):
    torch.manual_seed(780)
    net = OTTTSequential(
        layer.WSLinear(4, 3),
        SLTTLIFNode() if node_type is SLTTLIFNode else OTTTLIFNode(),
        layer.WSLinear(3, 2),
        node_type(),
    )
    reference = copy.deepcopy(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    x_seq = torch.rand(2, 3, 4)
    target_seq = torch.rand(2, 3, 2)

    for _ in range(2):
        expected_outputs, expected_losses = [], []
        reference_optimizer.zero_grad()
        for t in range(x_seq.shape[1]):
            output = reference(x_seq[:, t])
            if isinstance(output, list):
                output = output[0]
            loss = nn.functional.mse_loss(output, target_seq[:, t])
            loss.backward()
            expected_outputs.append(output.detach())
            expected_losses.append(loss.detach())
            detach_net(reference)
            if online:
                reference_optimizer.step()
                reference_optimizer.zero_grad()
        if not online:
            reference_optimizer.step()

        batch_loss, outputs = ottt_online_training(
            net, optimizer, x_seq, target_seq, nn.functional.mse_loss, online
        )
        torch.testing.assert_close(outputs, torch.stack(expected_outputs, dim=1))
        torch.testing.assert_close(batch_loss, torch.stack(expected_losses).sum())
        assert not outputs.requires_grad
        assert not batch_loss.requires_grad
        for actual, expected in zip(net.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(net.modules(), reference.modules()):
            if isinstance(actual, neuron.LIFNode):
                torch.testing.assert_close(actual.v, expected.v)
                assert actual.v.grad_fn is None
        reset_net(net)
        reset_net(reference)


@pytest.mark.parametrize("use_subclass", [False, True])
def test_ottt_sequential_calls_stacked_ottt_lif_once_on_spike(use_subclass):
    class CustomOTTTLIFNode(OTTTLIFNode):
        pass

    torch.manual_seed(754)
    net = OTTTSequential(
        nn.Linear(8, 2),
        OTTTLIFNode(),
        CustomOTTTLIFNode() if use_subclass else OTTTLIFNode(),
    )
    reference = copy.deepcopy(net)
    for x in torch.rand(4, 2, 8) * 3:
        y = net(x)
        expected = reference[2](reference[1](reference[0](x))[0])
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(net[2].v, reference[2].v)
        torch.testing.assert_close(net[2].trace, reference[2].trace)
        y[0].sum().backward()
        expected[0].sum().backward()
        torch.testing.assert_close(net[0].weight.grad, reference[0].weight.grad)
        torch.testing.assert_close(net[0].bias.grad, reference[0].bias.grad)
    reset_net(net)


@pytest.mark.parametrize("node_type", [OTTTLIFNode, SLTTLIFNode])
@pytest.mark.parametrize("v_reset", [None, 0.0])
def test_online_lif_cuts_temporal_gradients(node_type, v_reset):
    node = node_type(tau=2.0, decay_input=False, v_reset=v_reset)
    x_previous = torch.tensor([1.2], requires_grad=True)
    x_current = torch.tensor([0.3], requires_grad=True)
    node(x_previous)
    output = node(x_current)
    spike = output[0] if isinstance(output, list) else output
    spike.sum().backward()
    assert x_previous.grad is None
    voltage = 0.4 if v_reset is None else 0.3
    sigmoid = torch.sigmoid(4 * torch.tensor([voltage - 1.0]))
    torch.testing.assert_close(x_current.grad, 4 * sigmoid * (1 - sigmoid))
    if node_type is OTTTLIFNode:
        assert not node.trace.requires_grad


def test_fptt_updates_parameters_and_running_average():
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    w_ra = fptt_online_training_init_w_ra(optimizer)
    x_seq = torch.tensor([1.0, 2.0, 0.5], dtype=torch.double).reshape(3, 1, 1)
    targets = torch.tensor([0.2, 0.8, -0.1], dtype=torch.double).reshape(3, 1, 1)
    weight, average, previous_gradient = 0.5, 0.5, 0.0
    alpha = 0.3
    for t, (x, target) in enumerate(zip(x_seq.flatten(), targets.flatten())):
        gradient = 2 * (weight * x - target) * x
        gradient += alpha * (weight - average) - previous_gradient / 2
        weight -= 0.1 * gradient
        if t < 2:
            previous_gradient = 2 * (weight * x - target) * x
            average = (average + weight) / 2 - previous_gradient / (2 * alpha)

    fptt_online_training(
        model, optimizer, x_seq, targets, nn.functional.mse_loss, alpha, w_ra
    )
    torch.testing.assert_close(model.weight.squeeze(), weight)
    torch.testing.assert_close(w_ra[0].squeeze(), average)


def test_fptt_preserves_frozen_parameter_and_running_average():
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    ).double()
    with torch.no_grad():
        model[0].weight.fill_(0.5)
        model[1].weight.fill_(0.8)
    optimizer = torch.optim.SGD(
        [{"params": model[0].parameters()}, {"params": model[1].parameters()}], lr=0.1
    )
    averages = fptt_online_training_init_w_ra(optimizer)
    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.double).reshape(3, 1, 1)
    targets = torch.tensor([0.2, 0.8, -0.1], dtype=torch.double).reshape(3, 1, 1)
    fptt_online_training(
        model, optimizer, x, targets, nn.functional.mse_loss, 0.3, averages
    )
    model[0].requires_grad_(False)
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference[1].parameters(), lr=0.1)
    reference_averages = [averages[1].clone()]
    frozen_snapshot = model[0].weight.detach().clone()
    trainable_snapshot = model[1].weight.detach().clone()
    average_snapshot = averages[0].clone()
    assert not torch.equal(frozen_snapshot, average_snapshot)
    for net, opt, state in [
        (reference, reference_optimizer, reference_averages),
        (model, optimizer, averages),
    ]:
        fptt_online_training(net, opt, x, targets, nn.functional.mse_loss, 0.3, state)
    torch.testing.assert_close(model[0].weight, frozen_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(model[1].weight, reference[1].weight, rtol=0, atol=0)
    torch.testing.assert_close(averages[0], average_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(averages[1], reference_averages[0], rtol=0, atol=0)
    assert model[0].weight.grad is None
    assert not torch.equal(model[1].weight, trainable_snapshot)
