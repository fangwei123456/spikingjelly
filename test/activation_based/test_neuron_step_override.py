import copy

import pytest
import torch

from spikingjelly.activation_based import functional, neuron


def _analog_override(base_type):
    # A subclass that overrides only the single-step transition, as recommended
    # by the neuron tutorial: emit the membrane potential instead of spikes.
    class AnalogNode(base_type):
        def single_step_functional_forward(self, inputs, states, **kwargs):
            (spike,), states = super().single_step_functional_forward(
                inputs, states, **kwargs
            )
            return (spike + 0.5 * states[0],), states

    return AnalogNode


def _first(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _run(node, x_seq, step_mode):
    node = copy.deepcopy(node)
    functional.set_step_mode(node, step_mode)
    functional.reset_net(node)
    x_seq = x_seq.clone().requires_grad_(True)
    if step_mode == "m":
        out = _first(node(x_seq))
    else:
        out = torch.stack([_first(node(x)) for x in x_seq])
    (grad,) = torch.autograd.grad(out.sum(), x_seq)
    return out.detach(), grad, node.v.detach()


def _assert_single_and_multi_step_match(node, x_seq):
    out_s, grad_s, v_s = _run(node, x_seq, "s")
    out_m, grad_m, v_m = _run(node, x_seq, "m")
    torch.testing.assert_close(out_m, out_s)
    torch.testing.assert_close(grad_m, grad_s)
    torch.testing.assert_close(v_m, v_s)


@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize(
    "base_type", [neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode]
)
def test_multi_step_honors_single_step_override(base_type, store_v_seq):
    torch.manual_seed(0)
    node = _analog_override(base_type)(store_v_seq=store_v_seq)
    x_seq = torch.randn(6, 3, 4) + 0.5

    _assert_single_and_multi_step_match(node, x_seq)


@pytest.mark.parametrize(
    "base_type", [neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode]
)
def test_multi_step_functional_forward_honors_single_step_override(base_type):
    torch.manual_seed(0)
    node = _analog_override(base_type)()
    x_seq = torch.randn(6, 3, 4) + 0.5
    states = node.materialize_states((x_seq,), tuple(node._memories.values()), "m")

    (out_m,), (v_m,) = node.multi_step_functional_forward((x_seq,), states)
    out_s, v_s = [], states[0]
    for x in x_seq:
        (out,), (v_s,) = node.single_step_functional_forward((x,), (v_s,))
        out_s.append(out)

    torch.testing.assert_close(out_m, torch.stack(out_s))
    torch.testing.assert_close(v_m, v_s)


@pytest.mark.parametrize("store_v_seq", [False, True])
def test_liaf_multi_step_returns_analog_output(store_v_seq):
    torch.manual_seed(0)
    node = neuron.LIAFNode(
        act=torch.relu, threshold_related=False, store_v_seq=store_v_seq
    )
    x_seq = torch.randn(6, 3, 4) + 0.5

    _assert_single_and_multi_step_match(node, x_seq)


@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize("node_type", [neuron.OTTTLIFNode, neuron.SLTTLIFNode])
def test_online_lif_multi_step_keeps_the_online_gradient(node_type, store_v_seq):
    torch.manual_seed(0)
    node = node_type(store_v_seq=store_v_seq)
    x_seq = torch.randn(6, 2, 4) + 0.5

    _assert_single_and_multi_step_match(node, x_seq)
