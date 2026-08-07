import inspect
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.functional import layer as functional_layer


def test_delay_step_matches_sequence_delay_and_gradients():
    x_seq = torch.randn(5, 2, 3, requires_grad=True)
    queue = ()
    outputs = []
    for x in x_seq:
        y, queue = functional.delay_step(x, queue, delay_steps=2)
        outputs.append(y)
    y_seq = torch.stack(outputs)
    y_seq.sum().backward()

    x_ref = x_seq.detach().clone().requires_grad_()
    functional.delay(x_ref, 2).sum().backward()

    assert torch.allclose(y_seq, functional.delay(x_seq.detach(), 2))
    assert torch.allclose(x_seq.grad, x_ref.grad)
    assert torch.equal(queue[0], x_seq[-2])
    assert torch.equal(queue[1], x_seq[-1])


def test_delay_step_does_not_mutate_queue():
    x = torch.randn(2, 3)
    queue = (torch.randn(2, 3), torch.randn(2, 3))
    snapshot = tuple(item.clone() for item in queue)

    y, queue_next = functional.delay_step(x, queue, delay_steps=2)

    assert y is queue[0]
    assert queue_next[0] is queue[1]
    assert queue_next[1] is x
    for actual, expected in zip(queue, snapshot):
        assert torch.equal(actual, expected)


def test_delay_step_rejects_negative_delay():
    with pytest.raises(ValueError, match="non-negative integer"):
        functional.delay_step(torch.randn(2, 3), (), -1)


def test_delay_module_keeps_public_queue_list():
    x0 = torch.randn(2, 3)
    x1 = torch.randn(2, 3)
    module = layer.Delay(delay_steps=1)
    queue = module.queue

    y0 = module(x0)
    y1 = module(x1)

    assert module.queue is queue
    assert torch.equal(y0, torch.zeros_like(x0))
    assert y1 is x0
    assert module.queue == [x1]


@pytest.mark.parametrize("learnable", [False, True])
def test_synapse_filter_step_matches_module_and_gradients(learnable):
    x = torch.rand(2, 3)
    state = torch.rand(2, 3)
    tau = 3.5

    module = layer.SynapseFilter(tau=tau, learnable=learnable)
    x_module = x.clone().requires_grad_()
    state_module = state.clone().requires_grad_()
    module.out_i = state_module
    module(x_module).sum().backward()

    x_functional = x.clone().requires_grad_()
    state_functional = state.clone().requires_grad_()
    if learnable:
        w = torch.tensor(-math.log(tau - 1.0), requires_grad=True)
        reciprocal_tau = w.sigmoid()
    else:
        w = None
        reciprocal_tau = 1.0 / tau
    output = functional.synapse_filter_step(
        x_functional, state_functional, reciprocal_tau
    )
    output.sum().backward()

    assert torch.allclose(output, module.out_i)
    assert torch.allclose(x_functional.grad, x_module.grad)
    assert torch.allclose(state_functional.grad, state_module.grad)
    if learnable:
        assert torch.allclose(w.grad, module.w.grad)


def test_synapse_filter_step_does_not_mutate_state():
    x = torch.rand(2, 3)
    state = torch.rand(2, 3)
    x_before = x.clone()
    state_before = state.clone()

    functional.synapse_filter_step(x, state, 0.25)

    assert torch.equal(x, x_before)
    assert torch.equal(state, state_before)


def test_neunorm_step_matches_module():
    in_spikes = torch.randn(2, 3, 4, 5)
    state = torch.randn(2, 1, 4, 5)
    module = layer.NeuNorm(3, 4, 5)
    module.x = state.clone()

    output, state_next = functional.neunorm_step(
        in_spikes,
        state,
        module.w,
        module.k0,
        module.k1,
    )
    module_output = module(in_spikes)
    expected_state = module.k0 * state + module.k1 * in_spikes.sum(dim=1, keepdim=True)

    torch.testing.assert_close(output, module_output)
    torch.testing.assert_close(state_next, module.x)
    torch.testing.assert_close(state_next, expected_state)
    torch.testing.assert_close(output, in_spikes - module.w * expected_state)


@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_neunorm_materializes_channel_shared_state(step_mode):
    module = layer.NeuNorm(3, 4, 5, step_mode=step_mode)
    x = torch.randn(2, 3, 4, 5)
    output = module(x if step_mode == "s" else x.unsqueeze(0))

    assert output.shape == (x.shape if step_mode == "s" else (1, *x.shape))
    assert module.x.shape == (2, 1, 4, 5)


@pytest.mark.parametrize(
    ("spiking_type", "torch_type", "shape"),
    [
        (layer.BatchNorm1d, nn.BatchNorm1d, (3, 2, 4)),
        (layer.BatchNorm2d, nn.BatchNorm2d, (3, 2, 4, 5, 6)),
        (layer.BatchNorm3d, nn.BatchNorm3d, (3, 2, 4, 2, 3, 5)),
    ],
)
def test_multistep_batch_norm_matches_flattened_torch(spiking_type, torch_type, shape):
    spiking_bn = spiking_type(4, step_mode="m").eval()
    torch_bn = torch_type(4).eval()
    torch_bn.load_state_dict(spiking_bn.state_dict())
    x_seq = torch.randn(shape)

    actual = spiking_bn(x_seq)
    expected = torch_bn(x_seq.flatten(0, 1)).view_as(actual)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("bn_type", "torch_type", "shape"),
    [
        (layer.ThresholdDependentBatchNorm1d, nn.BatchNorm1d, (3, 2, 4)),
        (layer.ThresholdDependentBatchNorm2d, nn.BatchNorm2d, (3, 2, 4, 5, 6)),
        (
            layer.ThresholdDependentBatchNorm3d,
            nn.BatchNorm3d,
            (3, 2, 4, 2, 3, 5),
        ),
    ],
)
def test_threshold_dependent_batch_norm_uses_threshold_scaled_affine_weight(
    bn_type, torch_type, shape
):
    module = bn_type(alpha=0.5, v_th=1.5, num_features=4).eval()
    x_seq = torch.randn(shape)
    reference = torch_type(4).eval()
    reference.load_state_dict(module.state_dict())

    output = module(x_seq)
    expected = reference(x_seq.flatten(0, 1)).view_as(output)

    assert output.shape == x_seq.shape
    torch.testing.assert_close(module.weight, torch.full((4,), 0.75))
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("bn_type", "torch_type", "shape"),
    [
        (layer.TemporalEffectiveBatchNorm1d, nn.BatchNorm1d, (3, 2, 4)),
        (layer.TemporalEffectiveBatchNorm2d, nn.BatchNorm2d, (3, 2, 4, 5, 6)),
        (
            layer.TemporalEffectiveBatchNorm3d,
            nn.BatchNorm3d,
            (3, 2, 4, 2, 3, 5),
        ),
    ],
)
def test_temporal_effective_batch_norm_applies_per_step_scale(
    bn_type, torch_type, shape
):
    module = bn_type(3, num_features=4).eval()
    with torch.no_grad():
        module.scale.copy_(torch.tensor([1.0, 2.0, 3.0]))
    x_seq = torch.randn(shape)
    unscaled = torch_type(4).eval()(x_seq.flatten(0, 1)).view_as(x_seq)
    scale_shape = (3,) + (1,) * (x_seq.ndim - 1)

    torch.testing.assert_close(module(x_seq), unscaled * module.scale.view(scale_shape))


@pytest.mark.parametrize(
    ("bn_type", "shape"),
    [
        (layer.BatchNormThroughTime1d, (3, 2, 4)),
        (layer.BatchNormThroughTime2d, (3, 2, 4, 5, 6)),
        (layer.BatchNormThroughTime3d, (3, 2, 4, 2, 3, 5)),
    ],
)
def test_batch_norm_through_time_uses_independent_step_modules_and_resets(
    bn_type, shape
):
    module = bn_type(3, 4, step_mode="m").eval()
    with torch.no_grad():
        for index, bn in enumerate(module.bn_list, start=1):
            bn.weight.fill_(index)
    x_seq = torch.randn(shape)
    expected = torch.stack(
        [index * x / (1.0 + 1e-5) ** 0.5 for index, x in enumerate(x_seq, 1)]
    )

    torch.testing.assert_close(module(x_seq), expected)
    module.reset()
    torch.testing.assert_close(module(x_seq), expected)


def test_multistep_and_seq_to_ann_containers_match_their_execution_models():
    linear = nn.Linear(4, 3)
    x_seq = torch.randn(5, 2, 4)
    loop_container = layer.MultiStepContainer(linear)
    batch_container = layer.SeqToANNContainer(linear)
    expected = torch.stack([linear(x) for x in x_seq])

    torch.testing.assert_close(loop_container(x_seq), expected)
    torch.testing.assert_close(batch_container(x_seq), expected)


def test_elementwise_recurrent_container_accumulates_and_reset_clears_state():
    module = layer.ElementWiseRecurrentContainer(
        nn.Identity(), lambda previous, current: previous + current, step_mode="m"
    )
    x_seq = torch.tensor([[1.0], [2.0], [3.0]])

    assert torch.equal(module(x_seq), torch.tensor([[1.0], [3.0], [6.0]]))
    module.reset()
    assert torch.equal(module(x_seq), torch.tensor([[1.0], [3.0], [6.0]]))


def test_drop_connect_p_zero_keeps_connections_when_sampler_returns_zero(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)
    module = layer.DropConnectLinear(3, 2, p=0.0, activation=None).train()
    x = torch.randn(4, 3)

    torch.testing.assert_close(module(x), F.linear(x, module.weight, module.bias))


def test_voting_layer_averages_fixed_size_groups_in_both_step_modes():
    x = torch.tensor([[1.0, 3.0, 2.0, 6.0]])
    expected = torch.tensor([[2.0, 4.0]])

    assert torch.equal(layer.VotingLayer(2)(x), expected)
    assert torch.equal(layer.VotingLayer(2, step_mode="m")(x.unsqueeze(0))[0], expected)


def test_functional_layer_exports():
    for name in functional_layer.__all__:
        assert getattr(functional, name) is getattr(functional_layer, name)


def test_functional_layer_public_api_documentation():
    for name in functional_layer.__all__:
        function = getattr(functional_layer, name)
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
        doc = inspect.getdoc(function)
        assert f".. _{name}-cn:" in doc
        assert f".. _{name}-en:" in doc
