import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _node(kind, **kwargs):
    node_type = neuron.TernaryIFNode if kind == "if" else neuron.TernaryLIFNode
    return node_type(**kwargs)


def _scalar_rollout(x_seq, kind, threshold, reset, tau=2.5, decay_input=True):
    """Independent scalar threshold decisions, without surrogate/reset helpers."""
    voltage = [0.0 if reset is None else reset] * x_seq[0].numel()
    spike_steps = []
    voltage_steps = []
    for inputs in x_seq.flatten(1).tolist():
        spikes = []
        for index, current in enumerate(inputs):
            if kind == "if":
                charged = voltage[index] + current
            else:
                rest = 0.0 if reset is None else reset
                charged = rest + (voltage[index] - rest) * (1.0 - 1.0 / tau)
                charged += current / tau if decay_input else current
            if charged >= threshold:
                spike = 1.0
            elif charged <= -threshold:
                spike = -1.0
            else:
                spike = 0.0
            voltage[index] = (
                reset
                if reset is not None and spike != 0.0
                else charged - threshold * spike
                if reset is None
                else charged
            )
            spikes.append(spike)
        spike_steps.append(spikes)
        voltage_steps.append(voltage.copy())
    return (
        x_seq.new_tensor(spike_steps).reshape_as(x_seq),
        x_seq.new_tensor(voltage_steps).reshape_as(x_seq),
    )


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("reset", [None, 0.0, 0.2])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_matches_scalar_dynamics(kind, reset, decay_input, dtype):
    x_seq = torch.tensor(
        [[0.7, -0.7], [0.9, -0.9], [-2.8, 2.8], [3.4, -3.4], [0.0, 0.0]],
        dtype=dtype,
    )
    kwargs = {"tau": 2.5, "decay_input": decay_input} if kind == "lif" else {}
    node = _node(
        kind, v_threshold=0.8, v_reset=reset, step_mode="m", store_v_seq=True, **kwargs
    )
    expected_spikes, expected_voltages = _scalar_rollout(
        x_seq, kind, 0.8, reset, 2.5, decay_input
    )

    actual_spikes = node(x_seq)

    torch.testing.assert_close(actual_spikes, expected_spikes)
    torch.testing.assert_close(node.v_seq, expected_voltages)
    torch.testing.assert_close(node.v, expected_voltages[-1])
    assert actual_spikes.dtype == dtype
    assert actual_spikes.device == x_seq.device
    assert set(actual_spikes.flatten().tolist()) <= {-1.0, 0.0, 1.0}


def test_lif_matches_guo_equations_seven_and_eight():
    # Guo et al. use the pre-reset voltage and the preceding signed spike.
    # Their retained fraction 0.25 maps to SpikingJelly tau = 4/3.
    inputs = [0.8, 0.4, -1.2, -0.2, 0.0, 5.0, -5.0, 0.9, -1.0]
    previous_voltage, previous_spike = 0.0, 0.0
    expected_spikes, expected_voltages = [], []
    for current in inputs:
        voltage = 0.25 * previous_voltage * (1.0 - abs(previous_spike)) + current
        spike = 1.0 if voltage >= 1.0 else -1.0 if voltage <= -1.0 else 0.0
        expected_spikes.append(spike)
        expected_voltages.append(0.0 if spike else voltage)
        previous_voltage, previous_spike = voltage, spike
    assert expected_spikes == [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]

    x_seq = torch.tensor(inputs, dtype=torch.float64).unsqueeze(1)
    node = neuron.TernaryLIFNode(
        tau=4.0 / 3.0, decay_input=False, v_reset=0.0, step_mode="m", store_v_seq=True
    )
    actual = node(x_seq)

    torch.testing.assert_close(actual, x_seq.new_tensor(expected_spikes).unsqueeze(1))
    torch.testing.assert_close(
        node.v_seq, x_seq.new_tensor(expected_voltages).unsqueeze(1)
    )


@pytest.mark.parametrize("reset", [None, 0.0, 0.2])
def test_if_threshold_boundaries_and_one_spike_per_step(reset):
    node = neuron.TernaryIFNode(v_threshold=1.0, v_reset=reset)
    node.v = torch.zeros(7, dtype=torch.float64)
    charged = torch.tensor(
        [-5.0, -1.0, -0.99, 0.0, 0.99, 1.0, 5.0], dtype=torch.float64
    )

    spikes = node(charged)

    expected = charged.new_tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(spikes, expected)
    expected_v = charged - expected if reset is None else charged.clone()
    if reset is not None:
        expected_v[expected != 0] = reset
    torch.testing.assert_close(node.v, expected_v)


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("reset", [None, 0.0])
def test_zero_rest_dynamics_are_sign_symmetric(kind, reset):
    x_seq = torch.tensor(
        [[0.8, -0.6], [1.9, -2.7], [-3.1, 1.2], [0.5, 0.7]], dtype=torch.float64
    )
    positive = _node(kind, v_reset=reset, step_mode="m", store_v_seq=True)
    negative = _node(kind, v_reset=reset, step_mode="m", store_v_seq=True)

    torch.testing.assert_close(negative(-x_seq), -positive(x_seq))
    torch.testing.assert_close(negative.v_seq, -positive.v_seq)


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("reset", [None, 0.2])
def test_single_multi_and_functional_forward_and_gradients(kind, reset):
    x = torch.tensor(
        [[0.9, -0.7], [2.2, -2.0], [-1.7, 1.5], [0.3, -0.2]], dtype=torch.float64
    )
    x_single = x.clone().requires_grad_()
    x_multi = x.clone().requires_grad_()
    x_functional = x.clone().requires_grad_()
    single = _node(kind, v_reset=reset, store_v_seq=True)
    multi = _node(kind, v_reset=reset, store_v_seq=True, step_mode="m")
    pure = _node(kind, v_reset=reset, store_v_seq=True, step_mode="m")
    original_memories = tuple(pure._memories.values())

    (out_functional,), (state_functional,) = pure.functional_forward(
        (x_functional,), original_memories
    )
    assert all(
        actual is original
        for actual, original in zip(pure._memories.values(), original_memories)
    )
    assert pure.v_seq is None
    out_single = torch.stack([single(current) for current in x_single])
    out_multi = multi(x_multi)

    torch.testing.assert_close(out_single, out_multi)
    torch.testing.assert_close(out_functional, out_multi)
    torch.testing.assert_close(single.v_seq, multi.v_seq)
    torch.testing.assert_close(state_functional, multi.v)
    weights = x.new_tensor([[0.5, 0.8], [1.0, -0.3], [0.9, 0.2], [-0.7, 1.1]])
    for output, state in [
        (out_single, single.v),
        (out_multi, multi.v),
        (out_functional, state_functional),
    ]:
        ((output * weights).sum() + 0.4 * state.sum()).backward()
    torch.testing.assert_close(x_single.grad, x_multi.grad)
    torch.testing.assert_close(x_functional.grad, x_multi.grad)


@pytest.mark.parametrize("kind", ["if", "lif"])
def test_explicit_tensor_state_is_not_mutated(kind):
    node = _node(kind, store_v_seq=True)
    original_memory = torch.tensor([0.3, -0.2], dtype=torch.float64)
    node.v = original_memory
    supplied = torch.tensor([0.6, -0.5], dtype=torch.float64, requires_grad=True)
    supplied_before = supplied.detach().clone()
    x = torch.tensor([0.5, -0.4], dtype=torch.float64)

    (_,), (next_state,) = node.functional_forward((x,), (supplied,))

    assert node.v is original_memory
    assert node.v_seq is None
    assert next_state is not supplied
    torch.testing.assert_close(supplied, supplied_before)
    next_state.sum().backward()
    assert supplied.grad is not None


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("reset", [None, 0.2])
def test_reset_detach_and_training_evaluation_contract(kind, reset):
    node = _node(kind, v_reset=reset, step_mode="m", store_v_seq=True)
    x = torch.tensor([[0.8, -0.7], [2.5, -2.3]], requires_grad=True)
    training_output = node(x)
    assert node.v_seq.shape == x.shape
    assert node.v.requires_grad
    functional.detach_net(node)
    assert not node.v.requires_grad
    assert not node.v_seq.requires_grad

    functional.reset_net(node)
    assert node.v == (0.0 if reset is None else reset)
    assert node.v_seq is None
    node.eval()
    torch.testing.assert_close(node(x), training_output)


def _sigmoid_slope(x, alpha=4.0):
    probability = torch.sigmoid(alpha * x)
    return alpha * probability * (1.0 - probability)


@pytest.mark.parametrize("detach_reset", [False, True])
def test_signed_spike_has_positive_surrogate_gradient_at_both_thresholds(detach_reset):
    x = torch.tensor(
        [-1.2, -0.35, 0.0, 0.35, 1.2], dtype=torch.float64, requires_grad=True
    )
    node = neuron.TernaryIFNode(
        surrogate_function=surrogate.Sigmoid(alpha=4.0), detach_reset=detach_reset
    )

    node(x).sum().backward()

    expected = _sigmoid_slope(x.detach() - 1.0) + _sigmoid_slope(-x.detach() - 1.0)
    torch.testing.assert_close(x.grad, expected)
    assert torch.all(x.grad > 0)


@pytest.mark.parametrize("reset", [None, 0.0, 0.2])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_reset_gradient_uses_signed_spike_or_two_sided_reset_event(reset, detach_reset):
    x = torch.tensor([-1.2, -0.35, 0.35, 1.2], dtype=torch.float64, requires_grad=True)
    node = neuron.TernaryIFNode(
        v_reset=reset,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset=detach_reset,
    )
    node.v = torch.zeros_like(x)
    node(x)
    node.v.sum().backward()

    positive_slope = _sigmoid_slope(x.detach() - 1.0)
    negative_slope = _sigmoid_slope(-x.detach() - 1.0)
    if reset is None:
        expected = torch.ones_like(x)
        if not detach_reset:
            expected -= positive_slope + negative_slope
    else:
        event = (x.detach().abs() >= 1.0).to(x)
        expected = 1.0 - event
        if not detach_reset:
            expected += (reset - x.detach()) * (positive_slope - negative_slope)
            # abs(signed_spike) has zero derivative at silent, nonzero voltage.
            assert not torch.allclose(expected[1:3], torch.ones_like(expected[1:3]))
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("training", [False, True])
def test_continuous_surrogate_mode_matches_smooth_equations(training):
    x = torch.tensor([-1.2, -0.35, 0.35, 1.2], dtype=torch.float64, requires_grad=True)
    node = neuron.TernaryIFNode(
        surrogate_function=surrogate.Sigmoid(alpha=4.0, spiking=False)
    ).train(training)
    actual = node(x)
    positive = torch.sigmoid(4.0 * (x - 1.0))
    negative = torch.sigmoid(4.0 * (-x - 1.0))
    expected_spikes = positive - negative
    expected_state = x * (1.0 - positive - negative)
    torch.testing.assert_close(actual, expected_spikes)
    torch.testing.assert_close(node.v, expected_state)
    (actual.sum() + node.v.sum()).backward()
    expected_grad = 1.0 - positive.detach() - negative.detach()
    expected_grad += _sigmoid_slope(x.detach() - 1.0) + _sigmoid_slope(
        -x.detach() - 1.0
    )
    expected_grad -= x.detach() * (
        _sigmoid_slope(x.detach() - 1.0) - _sigmoid_slope(-x.detach() - 1.0)
    )
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(kind, threshold):
    with pytest.raises(ValueError):
        _node(kind, v_threshold=threshold)


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("reset", [-1.0, 1.0, float("nan"), float("inf")])
def test_reset_must_be_finite_and_inside_thresholds(kind, reset):
    with pytest.raises(ValueError):
        _node(kind, v_reset=reset)


@pytest.mark.parametrize("tau", [0.0, 1.0, -1.0, float("nan"), float("inf")])
def test_invalid_lif_time_constant_is_rejected(tau):
    with pytest.raises(ValueError):
        neuron.TernaryLIFNode(tau=tau)
