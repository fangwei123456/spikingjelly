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


@pytest.mark.parametrize(
    (
        "p",
        "bias",
        "zero_input",
        "zero_weight",
        "step_mode",
        "dtype",
        "activation_factory",
    ),
    [
        (0.5, False, True, False, "s", torch.float32, lambda: None),
        (0.5, True, True, False, "m", torch.float64, nn.Sigmoid),
        (0.5, False, False, True, "m", torch.float32, nn.ReLU),
        (0.5, True, False, True, "s", torch.float64, lambda: None),
        (0.0, False, False, False, "s", torch.float64, nn.ReLU),
        (0.0, True, False, False, "m", torch.float32, lambda: None),
        (1.0, False, False, False, "m", torch.float64, lambda: None),
        (1.0, True, False, False, "s", torch.float32, nn.Sigmoid),
    ],
)
def test_drop_connect_eval_zero_variance(
    step_mode, dtype, activation_factory, p, bias, zero_input, zero_weight
):
    module = (
        layer.DropConnectLinear(
            3,
            2,
            bias=bias,
            p=p,
            samples_num=8,
            activation=activation_factory(),
            step_mode=step_mode,
        )
        .to(dtype)
        .eval()
    )
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]]))
        if zero_weight:
            module.weight.zero_()
        if bias:
            module.bias.fill_(0.0 if zero_input or zero_weight else 0.25)
    x = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, 0.5]], dtype=dtype)
    if zero_input:
        x.zero_()
    if step_mode == "m":
        x = torch.stack([x, -x])
    x = x.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    original = x.detach().clone()
    expected = (1 - p) * F.linear(x, module.weight, module.bias)
    if module.activation is not None:
        expected = module.activation(expected)

    actual = module(x)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (*x.shape[:-1], 2)
    assert actual.dtype == dtype
    assert actual.device == x.device
    assert not actual.requires_grad
    assert torch.equal(x, original)
    assert module.dropped_w is None
    if bias:
        assert module.dropped_b is None


@pytest.mark.parametrize(
    ("step_mode", "dtype", "bias", "activation_factory"),
    [
        ("s", torch.float32, False, lambda: None),
        ("m", torch.float64, True, nn.Sigmoid),
    ],
)
def test_drop_connect_eval_mixed_zero_variance(
    step_mode, dtype, bias, activation_factory
):
    module = (
        layer.DropConnectLinear(
            3,
            2,
            bias=bias,
            samples_num=8,
            activation=activation_factory(),
            step_mode=step_mode,
        )
        .to(dtype)
        .eval()
    )
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.5, 2.0]]))
        if bias:
            module.bias.zero_()
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]], dtype=dtype)
    if step_mode == "m":
        x = torch.stack([x, -x])

    actual = module(x)

    expected = torch.zeros_like(actual)
    if module.activation is not None:
        expected = module.activation(expected)
    assert torch.equal(actual[..., 0], expected[..., 0])
    assert torch.equal(actual[..., 0, :], expected[..., 0, :])
    assert torch.isfinite(actual).all()
    assert actual.dtype == dtype
    assert actual.shape == expected.shape == (*x.shape[:-1], 2)


@pytest.mark.parametrize(
    ("step_mode", "dtype", "bias", "activation_factory"),
    [
        ("s", torch.float32, False, lambda: None),
        ("m", torch.float64, True, nn.PReLU),
    ],
)
def test_drop_connect_eval_preserves_sampling_and_gradients(
    step_mode, dtype, bias, activation_factory
):
    torch.manual_seed(918)
    module = (
        layer.DropConnectLinear(
            3,
            2,
            bias=bias,
            p=0.25,
            samples_num=11,
            activation=activation_factory(),
            step_mode=step_mode,
        )
        .to(dtype)
        .eval()
    )
    shape = (2, 3) if step_mode == "s" else (3, 2, 3)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    activation = copy.deepcopy(module.activation)
    rng_state = torch.get_rng_state()
    expected_steps = []
    for x_step in [x] if step_mode == "s" else x:
        mu = (1 - module.p) * F.linear(x_step, module.weight, module.bias)
        variance = (
            module.p
            * (1 - module.p)
            * F.linear(
                x_step.square(),
                module.weight.square(),
                None if module.bias is None else module.bias.square(),
            )
        )
        samples = torch.distributions.Normal(mu, variance.sqrt()).sample(
            (module.samples_num,)
        )
        if activation is not None:
            samples = activation(samples)
        expected_steps.append(samples.mean(0))
    expected = expected_steps[0] if step_mode == "s" else torch.stack(expected_steps)
    expected_rng_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)

    actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    if activation is None:
        assert not actual.requires_grad
    else:
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(
            module.activation.weight.grad, activation.weight.grad
        )
    assert x.grad is None
    assert module.weight.grad is None
    if bias:
        assert module.bias.grad is None


@pytest.mark.parametrize(
    ("invariant", "bias", "step_mode"), [(False, False, "s"), (True, True, "m")]
)
def test_drop_connect_eval_preserves_training_state(invariant, bias, step_mode):
    torch.manual_seed(123)
    module = layer.DropConnectLinear(
        3,
        2,
        bias=bias,
        p=0.5,
        samples_num=8,
        invariant=invariant,
        activation=None,
        step_mode=step_mode,
    ).train()
    x = torch.ones((2, 3) if step_mode == "s" else (3, 2, 3), requires_grad=True)
    training_output = module(x)
    training_output.sum().backward()
    assert x.grad is not None
    assert module.weight.grad is not None
    if bias:
        assert module.bias.grad is not None
    memories = tuple(module.memories())
    snapshots = tuple(state.detach().clone() for state in memories)
    parameters = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }

    module.eval()(torch.zeros_like(x))

    for state, previous, snapshot in zip(module.memories(), memories, snapshots):
        assert state is previous
        assert torch.equal(state, snapshot)
    for name, value in module.state_dict().items():
        assert torch.equal(value, parameters[name])
    module.train()
    if invariant:
        torch.testing.assert_close(module(x), training_output)
    module.reset()
    assert module.dropped_w is None
    if bias:
        assert module.dropped_b is None


@pytest.mark.parametrize(
    ("p", "input_value", "weight_values", "invalid_parameter"),
    [
        (0.5, 1e30, [1e30, -1e30], "loc"),
        (0.5, float("nan"), [1.0, 1.0], "loc"),
        (float("nan"), 1.0, [1.0, 1.0], "loc"),
        (-0.25, 1.0, [1.0, 1.0], "scale"),
        (1.25, 1.0, [1.0, 1.0], "scale"),
    ],
)
def test_drop_connect_eval_validates_before_sampling(
    p, input_value, weight_values, invalid_parameter
):
    module = (
        layer.DropConnectLinear(2, 1, bias=False, p=p, samples_num=8, activation=None)
        .float()
        .eval()
    )
    with torch.no_grad():
        module.weight.copy_(torch.tensor([weight_values], dtype=torch.float32))
    x = torch.full((1, 2), input_value, dtype=torch.float32)
    rng_state = torch.get_rng_state()

    with pytest.raises(ValueError, match=invalid_parameter):
        module(x)

    assert torch.equal(torch.get_rng_state(), rng_state)
