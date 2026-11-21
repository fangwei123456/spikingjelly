import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, layer, neuron


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
    alpha = 0.5
    v_th = 1.5
    module = bn_type(alpha=alpha, v_th=v_th, num_features=4).eval()
    x_seq = torch.randn(shape)
    reference = torch_type(4).eval()
    with torch.no_grad():
        reference.weight.fill_(alpha * v_th)

    output = module(x_seq)
    expected = reference(x_seq.flatten(0, 1)).view_as(output)

    assert output.shape == x_seq.shape
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
    module = bn_type(3, 4, eps=0.0, step_mode="m").eval()
    with torch.no_grad():
        for index, bn in enumerate(module.bn_list, start=1):
            bn.weight.fill_(index)
    x_seq = torch.randn(shape)
    expected = torch.stack([index * x for index, x in enumerate(x_seq, 1)])

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


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda sub_module: layer.LinearRecurrentContainer(sub_module, 3, 3),
        lambda sub_module: layer.ElementWiseRecurrentContainer(sub_module, torch.add),
    ],
)
def test_recurrent_containers_expose_wrapped_neuron_v_seq_in_multi_step_mode(
    container_factory,
):
    container = container_factory(neuron.LIFNode(store_v_seq=True))
    functional.set_step_mode(container, "m")
    reference = copy.deepcopy(container)
    functional.set_step_mode(reference, "s")
    x_seq = torch.rand(5, 2, 3) + 0.5

    y_seq = container(x_seq)
    expected_y_steps = []
    expected_v_steps = []
    for x in x_seq:
        expected_y_steps.append(reference(x))
        expected_v_steps.append(reference.sub_module.v)

    inner = container.sub_module
    assert inner.step_mode == "s"
    assert inner.v_seq.shape == x_seq.shape
    torch.testing.assert_close(y_seq, torch.stack(expected_y_steps))
    torch.testing.assert_close(inner.v_seq, torch.stack(expected_v_steps))
    torch.testing.assert_close(inner.v_seq[-1], inner.v)

    functional.reset_net(container)
    assert inner.v_seq is None


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


def test_linear_recurrent_container_rejects_batch_change_until_reset():
    module = layer.LinearRecurrentContainer(nn.Identity(), 3, 3, bias=False)
    module(torch.zeros(1, 3))
    previous = module.y.clone()
    with pytest.raises(ValueError, match="reset_net"):
        module(torch.zeros(2, 3))
    torch.testing.assert_close(module.y, previous)
    functional.reset_net(module)
    assert module(torch.zeros(2, 3)).shape == (2, 3)


@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "circular", "replicate"])
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_ws_conv2d_honors_padding_mode(padding_mode, step_mode):
    torch.manual_seed(0)
    conv = layer.WSConv2d(
        2, 3, 3, padding=1, padding_mode=padding_mode, step_mode=step_mode
    )
    x = torch.randn(2, 2, 2, 5, 5) if step_mode == "m" else torch.randn(2, 2, 5, 5)
    pad_mode = "constant" if padding_mode == "zeros" else padding_mode
    x_flat = x.flatten(0, 1) if step_mode == "m" else x
    expected = F.conv2d(
        F.pad(x_flat, (1, 1, 1, 1), mode=pad_mode), conv.get_weight(), conv.bias
    )
    expected = expected.view(*x.shape[:-3], *expected.shape[1:])

    torch.testing.assert_close(conv(x), expected)


@pytest.mark.parametrize("gain", [True, False])
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_ws_linear_matches_standardized_weight_values_and_gradients(gain, step_mode):
    torch.manual_seed(780)
    linear = layer.WSLinear(4, 3, gain=gain, step_mode=step_mode).double()
    if gain:
        with torch.no_grad():
            linear.gain.copy_(torch.tensor([[0.5], [1.0], [1.5]]))
    shape = (2, 4) if step_mode == "s" else (3, 2, 4)
    x = torch.randn(shape, dtype=torch.double, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    parameters = list(linear.parameters())
    reference = [p.detach().clone().requires_grad_() for p in parameters]
    weight, bias = reference[:2]
    centered = weight - weight.mean(dim=1, keepdim=True)
    variance = centered.square().sum(dim=1, keepdim=True) / 3
    standardized = centered / (4 * variance + linear.eps).sqrt()
    if gain:
        standardized = standardized * reference[2]
    expected = F.linear(x_ref, standardized, bias)

    actual = linear(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for actual_input, expected_input in zip([x, *parameters], [x_ref, *reference]):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_drop_connect_eval_mixed_zero_variance():
    activation = nn.PReLU(init=0.5)
    module = layer.DropConnectLinear(
        2, 2, bias=False, p=0.5, samples_num=8, activation=activation
    ).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.0, 0.0], [-1.0, -2.0]]))
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True)
    mu = 0.5 * F.linear(x, module.weight)
    std = (0.25 * F.linear(x.square(), module.weight.square())).sqrt()
    torch.manual_seed(918)
    samples = torch.distributions.Normal(mu, std, validate_args=False).sample((8,))
    expected = activation(samples).mean(0)
    torch.manual_seed(918)

    actual = module(x)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[0], torch.zeros(2))
    assert torch.equal(actual[:, 0], torch.zeros(2))
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    assert activation.weight.grad is not None
    assert x.grad is None
    assert module.weight.grad is None


@pytest.mark.parametrize("p", [-0.25, 1.25, float("nan")])
def test_drop_connect_eval_rejects_invalid_probability_with_zero_variance(p):
    module = layer.DropConnectLinear(2, 1, bias=False, p=p).eval()
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        module(torch.zeros(1, 2))


def test_dropconnect_public_drop_uses_fresh_masks_without_caching(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)
    module = layer.DropConnectLinear(3, 2, p=0.5, invariant=True, activation=None)
    weights, biases = module.drop(4)
    torch.testing.assert_close(weights, module.weight.expand(4, 2, 3))
    torch.testing.assert_close(biases, module.bias.expand(4, 2))
    assert module.weight_mask is None
    assert module.bias_mask is None


def test_dropconnect_non_invariant_forward_uses_overridden_drop():
    class KeepConnections(layer.DropConnectLinear):
        def drop(self, batch_size):
            return self.weight.expand(batch_size, -1, -1), None

    module = KeepConnections(3, 2, bias=False, p=1, activation=None, step_mode="m")
    x = torch.ones(2, 4, 3)
    with torch.no_grad():
        module.weight.fill_(1)

    torch.testing.assert_close(module(x), F.linear(x, module.weight))
    assert module.weight_mask is None

    module.step_mode = "s"
    module.invariant = True
    with pytest.raises(ValueError, match="_sample_masks"):
        module(x[0])


def test_dropconnect_invariant_subclass_reuses_custom_mask_views():
    class KeepConnections(layer.DropConnectLinear):
        def _sample_masks(self, batch_size):
            return (
                self.weight.new_ones((1, 2, 3), dtype=torch.bool).expand(
                    batch_size, -1, -1
                ),
                self.bias.new_ones((1, 2), dtype=torch.bool).expand(batch_size, -1),
            )

    module = KeepConnections(3, 2, p=1, invariant=True, activation=None)
    x = torch.ones(4, 3)
    torch.testing.assert_close(module(x), F.linear(x, module.weight, module.bias))
    functional.detach_net(module)

    with torch.no_grad():
        module.weight.add_(1)
    outputs, states = module.single_step_functional_forward(
        (x,), (module.weight_mask, module.bias_mask)
    )
    torch.testing.assert_close(outputs[0], F.linear(x, module.weight, module.bias))
    assert states[0] is module.weight_mask
    assert states[1] is module.bias_mask
    outputs[0].sum().backward()
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 4))
    torch.testing.assert_close(module.bias.grad, torch.full_like(module.bias, 4))


def test_dropconnect_rejects_legacy_cached_parameter_products():
    module = layer.DropConnectLinear(3, 2, invariant=True, activation=None)
    with pytest.raises(ValueError, match="boolean connection masks"):
        module.single_step_functional_forward(
            (torch.randn(4, 3),),
            (module.weight.expand(4, 2, 3), module.bias.expand(4, 2)),
        )


@pytest.mark.parametrize("conversion", ["double", "type"])
def test_dropconnect_cached_masks_survive_parameter_dtype_conversion(
    monkeypatch, conversion
):
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)
    module = layer.DropConnectLinear(3, 2, p=0.5, invariant=True, activation=None)
    x = torch.randn(4, 3)
    module(x)
    if conversion == "type":
        module.type(torch.DoubleTensor)
    else:
        module.double()
    actual = module(x.double())
    expected = F.linear(x.double(), module.weight, module.bias)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert module.weight.grad is not None
