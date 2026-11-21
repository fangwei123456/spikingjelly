import pytest
import torch

from spikingjelly.activation_based import base, neuron


@pytest.mark.parametrize(
    "node_type,kwargs",
    [
        (neuron.IFNode, {}),
        (neuron.LIFNode, {}),
        (neuron.ParametricLIFNode, {}),
        (neuron.ILIFNode, {}),
        (neuron.QIFNode, {}),
        (neuron.EIFNode, {}),
        (neuron.IzhikevichNode, {}),
        (neuron.HalfThresholdIFNode, {}),
        (neuron.TernaryIFNode, {}),
        (neuron.TernaryLIFNode, {}),
        (neuron.KLIFNode, {}),
        (neuron.RAFNode, {}),
        (neuron.CUBALIFNode, {}),
        (neuron.ComplementaryLIFNode, {}),
        (neuron.MPBNLIFNode, {"out_features": 4}),
        (neuron.LIAFNode, {"act": torch.relu, "threshold_related": False}),
        (neuron.ActivationAwareIFNode, {}),
    ],
)
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_sequence_dynamics_are_independent_of_single_step(
    node_type, kwargs, record, device
):
    class SingleStepOverride(node_type):
        def single_step_functional_forward(self, *args, **kwargs):
            raise AssertionError("sequence called overridable single-step transition")

    node = SingleStepOverride(**kwargs).to(device)
    node.step_mode = "m"
    if isinstance(node, neuron.ComplementaryLIFNode):
        node.store_state_seqs = record
    else:
        node.store_v_seq = record
    reference = node_type(**kwargs).to(device)
    reference.load_state_dict(node.state_dict())
    reference.step_mode = "s"
    x = torch.linspace(-0.4, 1.3, 24, device=device).reshape(3, 2, 4).requires_grad_()
    xs = x.detach().clone().requires_grad_()
    outputs, voltages = [], []
    for xt in xs:
        outputs.append(reference(xt))
        voltages.append(reference.v)
    expected = torch.stack(outputs)
    initial_states = tuple(base.extract_memories(node))
    (explicit,), final = base.to_functional_forward(node)((x,), initial_states)
    assert tuple(base.extract_memories(node)) == initial_states
    actual = node(x)
    for y in (actual, explicit):
        torch.testing.assert_close(y, expected)
    for group in (base.extract_memories(node), final):
        torch.testing.assert_close(
            tuple(group), tuple(base.extract_memories(reference))
        )
    if record:
        trace = (
            node.state_seqs[0]
            if isinstance(node, neuron.ComplementaryLIFNode)
            else node.v_seq
        )
        torch.testing.assert_close(trace, torch.stack(voltages))
    if expected.requires_grad:
        expected_grads = torch.autograd.grad(
            expected.sum(), (xs, *reference.parameters()), allow_unused=True
        )
        for y in (actual, explicit):
            gradients = torch.autograd.grad(
                y.sum(), (x, *node.parameters()), allow_unused=True
            )
            torch.testing.assert_close(gradients, expected_grads)


@pytest.mark.parametrize("node_type", [neuron.SLTTLIFNode, neuron.OTTTLIFNode])
def test_online_lif_rejects_sequence_entry_points(node_type):
    with pytest.raises(ValueError):
        node_type(step_mode="m")
    node = node_type()
    with pytest.raises(ValueError):
        node.step_mode = "m"
    x = torch.ones(2, 1)
    with pytest.raises(ValueError, match="single-step"):
        node.multi_step_forward(x)
    with pytest.raises(ValueError, match="single-step"):
        node.multi_step_functional_forward((x,), tuple(base.extract_memories(node)))


def test_base_neuron_requires_sequence_dynamics():
    class SingleOnly(neuron.BaseNode):
        def single_step_functional_forward(self, inputs, states, **kwargs):
            return inputs, states

    node = SingleOnly(step_mode="m")
    x = torch.ones(3, 2)
    for record in (False, True):
        node.store_v_seq = record
        with pytest.raises(NotImplementedError, match="multi-step"):
            node(x)
        with pytest.raises(NotImplementedError, match="multi-step"):
            node.functional_forward((x,), (0.0,))


def test_custom_sequence_transition_drives_both_entry_points():
    class SumNode(neuron.BaseNode):
        def _multi_step(self, inputs, states, store_v_seq, **kwargs):
            voltage = inputs[0].cumsum(0) + states[0]
            return (2 * voltage,), (voltage[-1],), voltage if store_v_seq else None

    node = SumNode(step_mode="m", store_v_seq=True)
    x = torch.tensor([[1.0], [2.0], [4.0]])
    expected = torch.tensor([[1.0], [3.0], [7.0]])
    (y,), (v,) = node.functional_forward((x,), (0.0,))
    torch.testing.assert_close(y, 2 * expected)
    torch.testing.assert_close(v, expected[-1])
    assert node.v == 0.0 and node.v_seq is None
    torch.testing.assert_close(node(x), y)
    torch.testing.assert_close(node.v, v)
    torch.testing.assert_close(node.v_seq, expected)
