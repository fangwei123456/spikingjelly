import inspect

import pytest
import torch

from spikingjelly.activation_based import functional, lava_exchange, neuron, surrogate
from spikingjelly.activation_based.functional import neuron as functional_neuron


def _surrogate():
    return surrogate.Sigmoid(alpha=4.0)


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("v_reset", [0.0, None])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_if_step_matches_module(v_reset, detach_reset):
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    function_surrogate = _surrogate()
    module = neuron.IFNode(
        v_threshold=0.7,
        v_reset=v_reset,
        surrogate_function=_surrogate(),
        detach_reset=detach_reset,
    )
    module.v = v.clone()

    spike, v_next = functional.if_step(
        x,
        v,
        0.7,
        v_reset,
        function_surrogate,
        detach_reset,
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


@pytest.mark.parametrize("decay_input", [False, True])
def test_lif_step_matches_module(decay_input):
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    module = neuron.LIFNode(
        tau=2.5,
        decay_input=decay_input,
        v_threshold=0.8,
        v_reset=None,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()

    spike, v_next = functional.lif_step(
        x,
        v,
        2.5,
        decay_input,
        0.8,
        None,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_lif_step_is_composed_from_charge_and_voltage_reset(decay_input, v_reset):
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    surrogate_function = _surrogate()

    charged = functional.lif_charge(x, v, 2.5, decay_input, v_reset)
    spike = surrogate_function(charged - 0.8)
    v_next = functional.voltage_reset(charged, spike, 0.8, v_reset, False)
    expected_spike, expected_v = functional.lif_step(
        x,
        v,
        2.5,
        decay_input,
        0.8,
        v_reset,
        surrogate_function,
    )

    _assert_close(spike, expected_spike)
    _assert_close(v_next, expected_v)


@pytest.mark.parametrize("threshold_related", [False, True])
def test_liaf_step_matches_module(threshold_related):
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    module = neuron.LIAFNode(
        torch.tanh,
        threshold_related,
        tau=2.5,
        decay_input=True,
        v_threshold=0.8,
        v_reset=None,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()

    output, v_next = functional.liaf_step(
        x,
        v,
        2.5,
        True,
        0.8,
        None,
        torch.tanh,
        threshold_related,
        _surrogate(),
    )
    module_output = module(x)

    _assert_close(output, module_output)
    _assert_close(v_next, module.v)


def test_ilif_step_matches_module():
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    module = neuron.ILIFNode(tau=1.5, v_threshold=0.8)
    module.v = v.clone()

    spike, v_next = functional.ilif_step(
        x,
        v,
        1.5,
        0.8,
        module.surrogate_function,
        module.detach_reset,
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


def test_materialize_states_is_pure():
    module = neuron.IFNode()
    x = torch.randn(2, 3)

    states = module.materialize_states((x,), (module.v,), "s")

    assert module.v == 0.0
    assert torch.equal(states[0], torch.zeros_like(x))


def test_plif_step_matches_module_and_parameter_gradient():
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    module = neuron.ParametricLIFNode(
        init_tau=2.5,
        decay_input=True,
        v_threshold=0.8,
        v_reset=0.0,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()

    w = module.w.detach().clone().requires_grad_()
    spike, v_next = functional.plif_step(
        x,
        v,
        w,
        True,
        0.8,
        0.0,
        _surrogate(),
    )
    (spike.sum() + v_next.sum()).backward()
    module_spike = module(x)
    (module_spike.sum() + module.v.sum()).backward()

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)
    _assert_close(w.grad, module.w.grad)


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: neuron.IFNode(v_reset=None, surrogate_function=_surrogate()),
        lambda: neuron.LIFNode(
            tau=2.5, decay_input=False, v_reset=0.0, surrogate_function=_surrogate()
        ),
        lambda: neuron.ParametricLIFNode(
            init_tau=2.5,
            decay_input=True,
            v_reset=None,
            surrogate_function=_surrogate(),
        ),
        lambda: neuron.LIAFNode(
            torch.relu,
            True,
            tau=2.5,
            decay_input=True,
            v_reset=0.0,
            surrogate_function=_surrogate(),
        ),
        lambda: neuron.ActivationAwareIFNode(
            v_threshold=0.8,
            v_offset=0.1,
            v_reset=None,
            surrogate_function=_surrogate(),
        ),
    ],
)
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_neuron_functional_forward_is_pure_and_backs_regular_forward(
    module_factory, step_mode
):
    module = module_factory()
    module.step_mode = step_mode
    module.store_v_seq = step_mode == "m"
    x = torch.randn(4, 2, 3) if step_mode == "m" else torch.randn(2, 3)
    v = torch.randn_like(x[0] if step_mode == "m" else x, requires_grad=True)
    module.v = v
    if step_mode == "m":
        module.v_seq = None

    original_states = tuple(module._memories.values())
    outputs, updated_states = module.functional_forward((x,), original_states)

    assert all(
        actual is expected
        for actual, expected in zip(module._memories.values(), original_states)
    )
    actual = module(x)
    _assert_close(actual, outputs[0])
    for actual_state, expected_state in zip(module._memories.values(), updated_states):
        if actual_state is not None:
            _assert_close(actual_state, expected_state)

    loss = outputs[0].sum() + updated_states[0].sum()
    loss.backward()
    assert v.grad is not None


@pytest.mark.parametrize("node", [neuron.IFNode(), neuron.LIFNode()])
def test_store_v_seq_does_not_change_functional_state_layout(node):
    node.step_mode = "m"
    node.store_v_seq = True
    assert tuple(name for name, _ in node.named_memories()) == ("v",)

    x_seq = torch.randn(3, 2)
    states = tuple(node._memories.values())
    node.functional_forward((x_seq,), states)
    assert node.v_seq is None

    node(x_seq)
    assert node.v_seq.shape == x_seq.shape

    node.store_v_seq = False
    assert tuple(name for name, _ in node.named_memories()) == ("v",)
    assert node.v_seq is None


def test_save_v_lif_keeps_observed_voltage_out_of_functional_state():
    from spikingjelly.activation_based.model.spike_dhs import save_v_LIFNode

    node = save_v_LIFNode(
        tau=2.5,
        decay_input=False,
        store_v_seq=True,
        step_mode="m",
        surrogate_function=_surrogate(),
    )
    x_seq = torch.randn(3, 2, 4)
    states = tuple(node._memories.values())

    outputs, updated_states = node.functional_forward((x_seq,), states)

    assert tuple(node._memories.values()) == states
    assert tuple(node._memories) == ("v",)
    assert node.v_before_spike is None
    torch.testing.assert_close(node(x_seq), outputs[0])
    for actual, expected in zip(node._memories.values(), updated_states):
        torch.testing.assert_close(actual, expected)
    assert node.v_seq.shape == (x_seq.shape[0],)

    reference_v = states[0]
    expected_observations = []
    for x in x_seq:
        charged = functional.lif_charge(
            x, reference_v, node.tau, node.decay_input, node.v_reset
        )
        expected_observations.append((charged - node.v_threshold).mean())
        spike = node.surrogate_function(charged - node.v_threshold)
        reference_v = functional.voltage_reset(
            charged,
            spike,
            node.v_threshold,
            node.v_reset,
            node.detach_reset,
        )
    torch.testing.assert_close(node.v_seq, torch.stack(expected_observations))


@pytest.mark.parametrize(
    ("module", "function"),
    [
        (
            neuron.QIFNode(
                tau=2.5,
                a0=0.6,
                v_rest=-0.2,
                v_c=0.4,
                v_threshold=0.9,
                v_reset=-0.3,
                surrogate_function=_surrogate(),
            ),
            lambda x, v: functional.qif_step(
                x, v, 2.5, 0.6, -0.2, 0.4, 0.9, -0.3, _surrogate()
            ),
        ),
        (
            neuron.EIFNode(
                tau=2.5,
                delta_T=0.7,
                theta_rh=0.4,
                v_rest=-0.2,
                v_threshold=0.9,
                v_reset=-0.3,
                surrogate_function=_surrogate(),
            ),
            lambda x, v: functional.eif_step(
                x, v, 2.5, 0.7, 0.4, -0.2, 0.9, -0.3, _surrogate()
            ),
        ),
    ],
)
def test_nonlinear_if_steps_match_modules(module, function):
    x = torch.randn(2, 3)
    v = torch.randn(2, 3) * 0.2
    module.v = v.clone()

    spike, v_next = function(x, v)
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


def test_izhikevich_step_matches_module():
    x = torch.randn(2, 3)
    v = torch.randn(2, 3) * 0.2
    w = torch.randn(2, 3) * 0.1
    module = neuron.IzhikevichNode(
        tau=2.5,
        v_c=0.4,
        a0=0.6,
        v_threshold=0.9,
        v_reset=-0.3,
        v_rest=-0.2,
        tau_w=3.0,
        a=0.1,
        b=0.2,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()
    module.w = w.clone()

    spike, v_next, w_next = functional.izhikevich_step(
        x,
        v,
        w,
        2.5,
        0.6,
        -0.2,
        0.4,
        3.0,
        0.1,
        0.2,
        0.9,
        -0.3,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)
    _assert_close(w_next, module.w)


@pytest.mark.parametrize("scale_reset", [False, True])
def test_klif_step_matches_module(scale_reset):
    x = torch.randn(2, 3)
    v = torch.rand(2, 3)
    module = neuron.KLIFNode(
        scale_reset=scale_reset,
        tau=2.5,
        decay_input=True,
        v_threshold=0.8,
        v_reset=0.0,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()

    spike, v_next = functional.klif_step(
        x,
        v,
        module.k,
        2.5,
        True,
        scale_reset,
        0.8,
        0.0,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


def test_cuba_lif_step_matches_module():
    x = torch.randn(2, 3)
    current = torch.randn(2, 3)
    v = torch.randn(2, 3)
    module = neuron.CUBALIFNode(
        c_decay=0.4,
        v_decay=0.7,
        v_threshold=0.8,
        v_reset=None,
        surrogate_function=_surrogate(),
    )
    module.c = current.clone()
    module.v = v.clone()

    spike, current_next, v_next = functional.cuba_lif_step(
        x,
        current,
        v,
        0.4,
        0.7,
        0.8,
        None,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(current_next, module.c)
    _assert_close(v_next, module.v)


def test_lava_cuba_lif_step_matches_norm_free_module():
    x = torch.randn(2, 3)
    current = torch.randn(2, 3)
    voltage = torch.randn(2, 3)
    module = lava_exchange.CubaLIFNode(
        current_decay=0.25,
        voltage_decay=0.5,
        surrogate_function=_surrogate(),
    )
    module.current_state = current.clone()
    module.voltage_state = voltage.clone()

    spike, current_next, voltage_next = functional.lava_cuba_lif_step(
        x,
        current,
        voltage,
        module.current_decay,
        module.voltage_decay,
        module.s_scale,
        module.v_threshold,
        module.v_threshold_eps,
        module.v_reset,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(current_next, module.current_state)
    _assert_close(voltage_next, module.voltage_state)


def test_cuba_sequence_caches_are_not_functional_states():
    module = lava_exchange.CubaLIFNode(
        current_decay=0.25,
        voltage_decay=0.5,
        step_mode="m",
        store_v_seq=True,
        store_i_seq=True,
    )
    x_seq = torch.randn(3, 2, 4)
    states = tuple(module._memories.values())

    outputs, updated_states = module.functional_forward((x_seq,), states)

    assert module.v_seq is None
    assert module.i_seq is None
    assert tuple(module._memories.values()) == states
    torch.testing.assert_close(module(x_seq), outputs[0])
    assert module.v_seq.shape == x_seq.shape
    assert module.i_seq.shape == x_seq.shape
    for actual, expected in zip(module._memories.values(), updated_states):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected)


def test_flexsn_sequence_cache_is_not_a_functional_state():
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def core(x, state):
        state = state + x
        return state, state

    module = FlexSN(
        core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
    )
    x_seq = torch.randn(3, 2, 4)

    outputs, updated_states = module.functional_forward((x_seq,), (module.states,))

    assert module.states is None
    assert module.state_seqs is None
    torch.testing.assert_close(module(x_seq), outputs[0])
    assert len(module.state_seqs) == 1
    torch.testing.assert_close(module.states[0], updated_states[0][0])


def test_activation_aware_if_step_matches_module():
    x = torch.randn(2, 3, 4, 4)
    v = torch.randn_like(x)
    module = neuron.ActivationAwareIFNode(
        v_threshold=torch.tensor([0.6, 0.8, 1.0]),
        v_offset=torch.tensor([-0.1, 0.0, 0.1]),
        channel_dim=1,
        v_reset=0.0,
        surrogate_function=_surrogate(),
    )
    module.v = v.clone()
    threshold = module._broadcast_parameter(module.v_threshold, x, "v_threshold")
    offset = module._broadcast_parameter(module.v_offset, x, "v_offset")

    spike, v_next = functional.activation_aware_if_step(
        x,
        v,
        threshold,
        offset,
        0.0,
        _surrogate(),
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(v_next, module.v)


def test_sliding_psn_step_matches_module():
    x_seq = torch.randn(5, 2, 3)
    module = neuron.SlidingPSN(k=3, surrogate_function=_surrogate())
    queue = ()

    expected = []
    for x in x_seq:
        spike, queue = functional.sliding_psn_step(
            x,
            queue,
            module.weight,
            module.bias,
            module.surrogate_function,
        )
        expected.append(spike)
    actual = torch.stack([module(x) for x in x_seq])

    _assert_close(actual, torch.stack(expected))
    assert len(module.queue) == len(queue)
    for actual_state, expected_state in zip(module.queue, queue):
        _assert_close(actual_state, expected_state)


def test_masked_psn_step_matches_module_sequence():
    x_seq = torch.randn(4, 2, 3)
    module = neuron.MaskedPSN(k=3, T=4, lambda_init=1.0)
    time_step = 0
    queue = ()
    expected = []

    for x in x_seq:
        spike, time_step, queue = functional.masked_psn_step(
            x,
            time_step,
            queue,
            module.masked_weight(),
            module.bias,
            module.k,
            module.surrogate_function,
        )
        expected.append(spike)
    actual = torch.stack([module(x) for x in x_seq])
    module.reset()
    module.step_mode = "m"
    multi_step = module(x_seq)

    _assert_close(actual, torch.stack(expected))
    _assert_close(actual, multi_step)
    assert time_step == x_seq.shape[0]
    assert len(queue) == module.k


def test_gated_lif_step_matches_module_sequence():
    x_seq = torch.randn(3, 2, 2, 3, 3)
    module = neuron.GatedLIFNode(T=x_seq.shape[0], inplane=x_seq.shape[2])
    v = torch.randn_like(x_seq[0])
    module.v = v.clone()

    alpha = module.alpha.view(1, -1, 1, 1).sigmoid()
    beta = module.beta.view(1, -1, 1, 1).sigmoid()
    gamma = module.gamma.view(1, -1, 1, 1).sigmoid()
    tau = module.tau.view(1, -1, 1, 1).sigmoid()
    v_threshold = module.v_threshold.view(1, -1, 1, 1).sigmoid()
    linear_decay = module.linear_decay.view(1, -1, 1, 1).sigmoid()
    v_subreset = module.v_subreset.view(1, -1, 1, 1).sigmoid()
    spike = torch.zeros_like(v)
    expected = []
    for t, x in enumerate(x_seq):
        spike, v = functional.gated_lif_step(
            x,
            v,
            spike,
            alpha,
            beta,
            gamma,
            tau,
            v_threshold,
            linear_decay,
            v_subreset,
            module.conduct[t].view(1, -1, 1, 1).sigmoid(),
            module.surrogate_function,
        )
        expected.append(spike)

    actual = module(x_seq)

    _assert_close(actual, torch.stack(expected))
    _assert_close(module.u, v)
    _assert_close(module.v, v)


def _stbif_reference(x, q, acc_q, threshold, pos_max, neg_min):
    normalized = x / threshold
    q_next = q + normalized
    positive = (q_next >= 1) & (acc_q < pos_max)
    negative = (q_next < 0) & (acc_q > neg_min)
    current = positive.to(x.dtype) - negative.to(x.dtype)
    acc_q_next = acc_q + current
    q_next = torch.where(positive, q_next - 1, q_next)
    q_next = torch.where(negative, q_next + 1, q_next)
    return current * threshold, q_next, acc_q_next, current


def test_stbif_step_matches_reference_sequence():
    x_seq = torch.randn(4, 2, 3, dtype=torch.float64)
    threshold = torch.tensor(0.25)
    pos_max = torch.tensor(1.0)
    neg_min = torch.tensor(-2.0)
    q = torch.full_like(x_seq[0], 0.5)
    acc_q = torch.zeros_like(q)

    q_ref = q
    acc_q_ref = acc_q
    for x in x_seq:
        step = functional.stbif_step(x, q_ref, acc_q_ref, threshold, pos_max, neg_min)
        reference = _stbif_reference(x, q_ref, acc_q_ref, threshold, pos_max, neg_min)
        for actual, expected in zip(step, reference):
            _assert_close(actual, expected)
        q_ref, acc_q_ref = step[1:3]


def test_stbif_module_derives_is_work():
    module = neuron.STBIFNeuron(0.25, level=8, sym=True)
    x = torch.zeros(2, 3)

    module(x)
    assert not module.is_work

    module(torch.full_like(x, 0.25))
    assert module.is_work

    module.reset()
    module.step_mode = "m"
    module(torch.stack([x, x]))
    assert not module.is_work

    module(torch.stack([x, torch.full_like(x, 0.25)]))
    assert module.is_work


def test_multi_step_names_identify_independent_sequence_paths():
    backend_suffixes = ("_cupy", "_triton")
    for name in functional_neuron.__all__:
        if "_multi_step" in name:
            assert name.endswith(backend_suffixes), name


def test_functional_neuron_exports():
    for name in functional_neuron.__all__:
        assert getattr(functional, name) is getattr(functional_neuron, name)


def test_functional_neuron_public_api_documentation():
    for name in functional_neuron.__all__:
        function = getattr(functional_neuron, name)
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ), name
        doc = inspect.getdoc(function)
        assert f".. _{name}-cn:" in doc, name
        assert f".. _{name}-en:" in doc, name
