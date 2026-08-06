import copy

import pytest
import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import DSRIFNode, DSRLIFNode, MPBNLIFNode


@pytest.mark.parametrize(
    ("node_type", "kwargs"),
    [
        (DSRIFNode, {}),
        (DSRLIFNode, {"tau": 2.0, "delta_t": 0.05}),
    ],
)
def test_dsr_neuron_backward_does_not_require_distributed(node_type, kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    node = node_type(T=4, **kwargs).to(device)
    x = torch.randn(4, 2, 3, device=device, requires_grad=True)

    node(x).sum().backward()

    assert x.grad is not None
    assert node.v_threshold.grad is not None
    assert torch.isfinite(node.v_threshold.grad)


@pytest.mark.parametrize("shape", [(8, 4), (8, 4, 3, 3)])
@pytest.mark.parametrize("learnable_vth", [False, True])
def test_mpbn_threshold_reparameterization_preserves_eval_result(shape, learnable_vth):
    width = shape[1]
    kwargs = {"out_features": width} if len(shape) == 2 else {"out_channels": width}
    node = MPBNLIFNode(
        tau=2.0,
        decay_input=False,
        learnable_vth=learnable_vth,
        **kwargs,
    ).train()
    for _ in range(3):
        functional.reset_net(node)
        node(torch.randn(*shape))

    functional.reset_net(node)
    node.eval()
    folded = copy.deepcopy(node)
    folded.re_parameterize_v_threshold(normalize_residual=True)
    x = torch.randn(*shape, requires_grad=True)
    functional.reset_net(node)
    functional.reset_net(folded)

    expected = node(x)
    actual = folded(x)

    assert torch.equal(actual, expected)
    assert torch.allclose(folded.v, node.v, atol=1e-5, rtol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_mpbn_functional_forward_keeps_running_statistics_pure(step_mode):
    node = MPBNLIFNode(tau=2.0, out_features=4, step_mode=step_mode).train()
    x = torch.randn(3, 4) if step_mode == "s" else torch.randn(5, 3, 4)
    state_dict_before = {
        name: value.clone() for name, value in node.state_dict().items()
    }

    (actual,), (v,) = node.functional_forward((x,), (0.0,))

    for name, value in node.state_dict().items():
        assert torch.equal(value, state_dict_before[name])

    stateful_node = copy.deepcopy(node)
    expected = stateful_node(x)
    assert torch.allclose(actual, expected)
    assert torch.allclose(v, stateful_node.v)
    expected_batches = 1 if step_mode == "s" else x.shape[0]
    assert stateful_node.vbn.num_batches_tracked == expected_batches


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 2, 2)])
def test_mpbn_forward_matches_torch_batch_norm(shape):
    kwargs = (
        {"out_features": shape[1]} if len(shape) == 2 else {"out_channels": shape[1]}
    )
    node = MPBNLIFNode(tau=2.0, **kwargs).train()
    reference_bn = copy.deepcopy(node.vbn)
    x = torch.randn(*shape)
    charged = functional.lif_charge(x, torch.zeros_like(x), 2.0, False, 0.0)
    normalized = reference_bn(charged)
    expected_spike = node.surrogate_function(normalized - node.v_threshold)
    expected_v = functional.voltage_reset(
        normalized,
        expected_spike,
        node.v_threshold,
        node.v_reset,
        node.detach_reset,
    )

    actual_spike = node(x)

    assert torch.allclose(actual_spike, expected_spike)
    assert torch.allclose(node.v, expected_v)
    assert torch.equal(node.vbn.running_mean, reference_bn.running_mean)
    assert torch.equal(node.vbn.running_var, reference_bn.running_var)
    assert torch.equal(
        node.vbn.num_batches_tracked,
        reference_bn.num_batches_tracked,
    )


def test_folded_mpbn_multi_step_functional_forward_threads_local_statistics():
    node = MPBNLIFNode(
        tau=2.0,
        out_features=4,
        step_mode="m",
    ).train()
    for _ in range(3):
        functional.reset_net(node)
        node(torch.randn(5, 3, 4))
    node.re_parameterize_v_threshold(
        normalize_residual=True,
        running_stats=True,
    )
    functional.reset_net(node)
    stateful_node = copy.deepcopy(node)
    mu_before = node.mu.clone()
    sigma2_before = node.sigma2.clone()
    momentum_before = node.bn_momentum
    x = torch.randn(5, 3, 4)

    (actual,), (v,) = node.functional_forward((x,), (0.0,))
    expected = stateful_node(x)

    assert torch.allclose(actual, expected)
    assert torch.allclose(v, stateful_node.v)
    assert torch.equal(node.mu, mu_before)
    assert torch.equal(node.sigma2, sigma2_before)
    assert node.bn_momentum == momentum_before
    assert not torch.equal(stateful_node.mu, mu_before)
