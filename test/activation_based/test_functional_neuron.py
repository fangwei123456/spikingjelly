import pytest
import torch

from spikingjelly.activation_based import functional, lava_exchange, neuron, surrogate


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


@pytest.mark.parametrize(
    ("device", "dtype", "rtol", "atol"),
    [
        ("cpu", torch.float64, 1e-10, 1e-12),
        pytest.param(
            "cuda",
            torch.float32,
            1e-6,
            1e-6,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_complementary_lif_matches_paper_dynamics_and_gradient(
    device, dtype, rtol, atol
):
    x_seq = torch.tensor(
        [[0.8], [0.4], [0.9], [0.4], [1.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    module = neuron.ComplementaryLIFNode(step_mode="m", store_state_seqs=True)

    spike_seq = module(x_seq)

    expected_spike_seq = x_seq.new_tensor([[0.0], [0.0], [1.0], [0.0], [1.0]])
    expected_v_seq = x_seq.new_tensor(
        [
            [0.8],
            [0.8],
            [-0.43105857863000474],
            [0.18447071068499765],
            [-0.6988051099221311],
        ]
    )
    expected_m_seq = x_seq.new_tensor(
        [[0.0], [0.0], [1.0], [0.5230425052426416], [1.331208504044421]]
    )
    torch.testing.assert_close(spike_seq, expected_spike_seq, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        module.state_seqs[0], expected_v_seq, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        module.state_seqs[1], expected_m_seq, rtol=rtol, atol=atol
    )

    weights = x_seq.new_tensor([1, 2, 3, 4, 5]).unsqueeze(1)
    (spike_seq * weights).sum().backward()
    expected_grad = x_seq.new_tensor(
        [
            [0.5587231024047484],
            [1.3784269134271934],
            [1.8404118601606416],
            [2.5],
            [5.0],
        ]
    )
    torch.testing.assert_close(x_seq.grad, expected_grad, rtol=rtol, atol=atol)
    assert list(module.parameters()) == []


def test_complementary_lif_single_multi_step_and_functional_state():
    scalar_state = neuron.ComplementaryLIFNode()
    for m in (1, torch.tensor(1.0)):
        _, states = scalar_state.functional_forward((torch.zeros(2),), (0.0, m))
        torch.testing.assert_close(states[1], torch.full((2,), 0.5))

    x_seq = torch.tensor(
        [[0.3, 0.8], [0.9, 0.4], [0.7, 1.0], [0.6, 0.2]],
        dtype=torch.float64,
    )
    multi_step = neuron.ComplementaryLIFNode(step_mode="m", store_state_seqs=True)
    original_states = (multi_step.v, multi_step.m)
    functional_outputs, functional_states = multi_step.functional_forward(
        (x_seq,), original_states
    )

    assert (multi_step.v, multi_step.m) == original_states
    assert multi_step.state_seqs is None

    multi_output = multi_step(x_seq)
    single_step = neuron.ComplementaryLIFNode()
    single_outputs = []
    single_v_steps = []
    single_m_steps = []
    for x in x_seq:
        single_outputs.append(single_step(x))
        single_v_steps.append(single_step.v)
        single_m_steps.append(single_step.m)

    torch.testing.assert_close(multi_output, torch.stack(single_outputs))
    torch.testing.assert_close(multi_output, functional_outputs[0])
    torch.testing.assert_close(multi_step.state_seqs[0], torch.stack(single_v_steps))
    torch.testing.assert_close(multi_step.state_seqs[1], torch.stack(single_m_steps))
    torch.testing.assert_close(multi_step.v, functional_states[0])
    torch.testing.assert_close(multi_step.m, functional_states[1])


@pytest.mark.parametrize("training", [True, False])
def test_clif_step_matches_module_and_gradients(training):
    x_module = torch.tensor([0.4, 1.1], requires_grad=True)
    v_module = torch.tensor([0.2, 0.6], requires_grad=True)
    m_module = torch.tensor([0.1, 0.3], requires_grad=True)
    module = neuron.ComplementaryLIFNode(v_threshold=0.8).train(training)
    module.v = v_module
    module.m = m_module

    spike_module = module(x_module)
    (spike_module + module.v + module.m).sum().backward()

    x = x_module.detach().clone().requires_grad_()
    v = v_module.detach().clone().requires_grad_()
    m = m_module.detach().clone().requires_grad_()
    spike_function = (
        module.surrogate_function
        if training or not module.surrogate_function.spiking
        else surrogate.heaviside
    )
    spike, v_next, m_next = functional.clif_step(
        x, v, m, module.tau, module.v_threshold, spike_function
    )
    (spike + v_next + m_next).sum().backward()

    torch.testing.assert_close(spike, spike_module)
    torch.testing.assert_close(v_next, module.v)
    torch.testing.assert_close(m_next, module.m)
    torch.testing.assert_close(x.grad, x_module.grad)
    torch.testing.assert_close(v.grad, v_module.grad)
    torch.testing.assert_close(m.grad, m_module.grad)


def test_complementary_lif_state_storage_reset_and_backend_contract():
    module = neuron.ComplementaryLIFNode(step_mode="m", store_state_seqs=True)
    x_seq = torch.randn(3, 2, 4)

    training_output = module(x_seq)
    assert len(module.state_seqs) == 2
    assert all(state_seq.shape == x_seq.shape for state_seq in module.state_seqs)

    module.reset()
    assert module.v == 0.0
    assert module.m == 0.0
    assert module.state_seqs is None

    module.eval()
    evaluation_output = module(x_seq)
    torch.testing.assert_close(training_output, evaluation_output)
    module.store_state_seqs = False
    module.store_v_seq = True
    assert module.state_seqs is None

    x_seq = torch.randn(2, 3, dtype=torch.float64)
    module(x_seq)
    assert module.v_seq is None
    assert module.v.shape == x_seq.shape[1:]
    assert module.m.shape == x_seq.shape[1:]
    assert module.v.dtype == x_seq.dtype
    assert module.m.dtype == x_seq.dtype


def test_complementary_lif_single_step_ignores_store_v_seq():
    module = neuron.ComplementaryLIFNode()
    module.store_v_seq = True
    x = torch.rand(2, 3)

    module(x)
    module(x)
    assert module.v_seq is None
    assert module.v.shape == x.shape

    functional.reset_net(module)
    assert module.v_seq is None
    assert module.v == 0.0


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


def test_store_v_seq_does_not_change_functional_state_layout():
    node = neuron.IFNode()
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


@pytest.mark.parametrize("node_type", [neuron.IFNode, neuron.LIFNode])
def test_store_v_seq_accumulates_across_single_steps_until_reset(node_type):
    x_seq = torch.rand(4, 2, 3) + 0.5
    x_single = x_seq.clone().requires_grad_()
    x_multi = x_seq.clone().requires_grad_()
    node_s = node_type(store_v_seq=True)
    node_m = node_type(store_v_seq=True, step_mode="m")

    spike_seq = torch.stack([node_s(x) for x in x_single])
    expected_spike_seq = node_m(x_multi)

    torch.testing.assert_close(spike_seq, expected_spike_seq)
    torch.testing.assert_close(node_s.v_seq, node_m.v_seq)
    torch.testing.assert_close(node_s.v, node_m.v)
    node_s.v_seq.sum().backward()
    node_m.v_seq.sum().backward()
    torch.testing.assert_close(x_single.grad, x_multi.grad)

    node_s.reset()
    assert node_s.v_seq is None

    node_s(x_seq[0])
    node_s.store_v_seq = False
    node_s(x_seq[1])
    assert node_s.v_seq is None
    node_s.store_v_seq = True
    node_s(x_seq[2])
    torch.testing.assert_close(node_s.v_seq, node_s.v.unsqueeze(0))

    node_s(torch.rand(5, 3))
    assert node_s.v_seq.shape == (1, 5, 3)
    torch.testing.assert_close(node_s.v_seq, node_s.v.unsqueeze(0))

    node_off = node_type()
    for x in x_seq:
        node_off(x)
    assert node_off.v_seq is None


def test_single_step_v_seq_owns_its_storage():
    node = neuron.IFNode(store_v_seq=True)
    with torch.no_grad():
        node(torch.rand(2, 3))
        recorded = node.v_seq
        node.v.fill_(-1.0)
    assert node.v_seq.data_ptr() != node.v.data_ptr()
    assert not torch.equal(recorded[0], node.v)


def test_detach_covers_single_step_v_seq():
    node = neuron.LIFNode(store_v_seq=True)
    x = torch.rand(2, 2, 3, requires_grad=True)
    node(x[0])
    assert node.v_seq.requires_grad
    functional.detach_net(node)
    assert not node.v.requires_grad
    assert not node.v_seq.requires_grad
    node(x[1])
    assert node.v_seq.shape == (2, 2, 3)
    node.v_seq.sum().backward()
    assert torch.all(x.grad[0] == 0)
    assert torch.any(x.grad[1] != 0)


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


@pytest.mark.parametrize("v_reset", [0.0, None])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_raf_step_matches_module(v_reset, detach_reset):
    x = torch.randn(2, 3)
    u = torch.randn(2, 3) * 0.3
    v = torch.randn(2, 3) * 0.3
    function_surrogate = _surrogate()
    module = neuron.RAFNode(
        b=-0.2,
        omega=1.3,
        dt=0.5,
        v_threshold=0.5,
        v_reset=v_reset,
        surrogate_function=_surrogate(),
        detach_reset=detach_reset,
    )
    module.u = u.clone()
    module.v = v.clone()

    spike, u_next, v_next = functional.raf_step(
        x,
        u,
        v,
        -0.2,
        1.3,
        0.5,
        0.5,
        v_reset,
        function_surrogate,
        detach_reset,
    )
    module_spike = module(x)

    _assert_close(spike, module_spike)
    _assert_close(u_next, module.u)
    _assert_close(v_next, module.v)


def test_raf_reset_only_touches_v():
    # a spike must hard/soft-reset v but leave u exactly as the charge step
    # produced it -- that is what gives post-inhibitory rebound.
    for v_reset in (0.0, None):
        module = neuron.RAFNode(
            b=-0.2, omega=1.0, dt=1.0, v_threshold=0.3, v_reset=v_reset
        )
        module.u = torch.tensor([2.0])
        module.v = torch.tensor([2.0])

        reference = neuron.RAFNode(
            b=-0.2, omega=1.0, dt=1.0, v_threshold=1e9, v_reset=v_reset
        )
        reference.u = torch.tensor([2.0])
        reference.v = torch.tensor([2.0])

        spike = module(torch.tensor([0.0]))
        reference(torch.tensor([0.0]))

        assert spike.item() == 1.0
        _assert_close(module.u, reference.u)
        if v_reset is None:
            _assert_close(module.v, reference.v - module.v_threshold)
        else:
            _assert_close(module.v, torch.full_like(reference.v, v_reset))


def test_raf_single_step_multi_step_equivalence():
    torch.manual_seed(0)
    x_seq = torch.randn(6, 2, 3)

    single_step = neuron.RAFNode(b=-0.15, omega=0.8, dt=0.5, v_threshold=0.4)
    single_spikes, single_u_steps, single_v_steps = [], [], []
    for x in x_seq:
        single_spikes.append(single_step(x))
        single_u_steps.append(single_step.u.clone())
        single_v_steps.append(single_step.v.clone())

    multi_step = neuron.RAFNode(
        b=-0.15, omega=0.8, dt=0.5, v_threshold=0.4, step_mode="m", store_v_seq=True
    )
    multi_spikes = multi_step(x_seq)

    _assert_close(multi_spikes, torch.stack(single_spikes))
    _assert_close(multi_step.v_seq, torch.stack(single_v_steps))
    _assert_close(multi_step.v, single_step.v)
    _assert_close(multi_step.u, single_step.u)


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


def test_cuba_store_v_seq_accumulates_across_single_steps_until_reset():
    x_seq = torch.rand(4, 2, 3) + 0.5
    node_s = lava_exchange.CubaLIFNode(
        current_decay=0.25, voltage_decay=0.5, store_v_seq=True
    )
    node_m = lava_exchange.CubaLIFNode(
        current_decay=0.25, voltage_decay=0.5, step_mode="m", store_v_seq=True
    )

    spike_seq = torch.stack([node_s(x) for x in x_seq])
    expected_spike_seq = node_m(x_seq)

    torch.testing.assert_close(spike_seq, expected_spike_seq)
    torch.testing.assert_close(node_s.v_seq, node_m.v_seq)
    torch.testing.assert_close(node_s.voltage_state, node_m.voltage_state)
    assert node_s.i_seq is None

    node_s(torch.rand(5, 3))
    assert node_s.v_seq.shape == (1, 5, 3)
    torch.testing.assert_close(node_s.v_seq, node_s.voltage_state.unsqueeze(0))

    x = torch.rand(5, 3, requires_grad=True)
    node_s(x)
    assert node_s.v_seq.requires_grad
    node_s.detach()
    assert not node_s.v_seq.requires_grad

    node_s.reset()
    assert node_s.v_seq is None

    # The inherited ``v`` memory is vestigial for CubaLIFNode; even if a caller
    # assigns a tensor to it, each step must record exactly one entry.
    node_s(x_seq[0])
    node_s.v = torch.zeros_like(node_s.voltage_state)
    node_s(x_seq[1])
    assert node_s.v_seq.shape == (2, 2, 3)
    torch.testing.assert_close(node_s.v_seq[-1], node_s.voltage_state)
    node_s.reset()

    node_off = lava_exchange.CubaLIFNode(current_decay=0.25, voltage_decay=0.5)
    for x in x_seq:
        node_off(x)
    assert node_off.v_seq is None


def test_flexsn_sequence_cache_is_not_a_functional_state():
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    def core(x, state):
        state = state + x
        return state, state

    module = FlexSN(
        core,
        num_states=1,
        step_mode="m",
        store_state_seqs=True,
    )
    x_seq = torch.randn(3, 2, 4)

    outputs, updated_states = module.functional_forward(
        (x_seq,), module.states, static_inputs=()
    )

    assert module.states == (None,)
    assert module.state_seqs is None
    torch.testing.assert_close(module(x_seq), outputs[0])
    assert len(module.state_seqs) == 1
    torch.testing.assert_close(module.states[0], updated_states[0])


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_sliding_psn_multi_step_gemm_follows_parameter_dtype(dtype):
    # Regression: gen_gemm_weight() built the T x T weight matrix with the
    # parameter's device but a hardcoded float32 dtype, so a bf16/fp16
    # SlidingPSN crashed in the gemm multi-step path with
    # "mat1 and mat2 must have the same dtype".
    x_seq = torch.randn(5, 2, 3, dtype=dtype)
    module = neuron.SlidingPSN(k=3, surrogate_function=_surrogate()).to(dtype)
    assert module.gen_gemm_weight(x_seq.shape[0]).dtype == dtype

    module.step_mode = "m"
    multi_step = module(x_seq)
    assert multi_step.dtype == dtype

    if dtype == torch.float32:
        # The gemm and single-step paths accumulate in different orders, so the
        # emitted spikes only agree in float32; in half precision a membrane
        # potential near the threshold can flip one spike between the paths.
        module.step_mode = "s"
        module.reset()
        single_step = torch.stack([module(x) for x in x_seq])
        torch.testing.assert_close(single_step, multi_step)


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


def test_masked_psn_overflow_keeps_queue_side_effect():
    module = neuron.MaskedPSN(k=2, T=2, lambda_init=1.0)
    x0, x1, x2 = torch.randn(3, 2, 4)
    module(x0)
    module(x1)

    with pytest.raises(OverflowError):
        module(x2)

    assert module.time_step == 2
    torch.testing.assert_close(module.queue[0], x1.flatten())
    torch.testing.assert_close(module.queue[1], x2.flatten())

    explicit_queue = [x0.flatten(), x1.flatten()]
    with pytest.raises(OverflowError):
        module.single_step_functional_forward((x2,), (2, explicit_queue))
    torch.testing.assert_close(explicit_queue[0], x0.flatten())
    torch.testing.assert_close(explicit_queue[1], x1.flatten())


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


@pytest.mark.parametrize(
    "node_type", [neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode]
)
def test_multistep_trajectory_uses_sequence_operator_and_preserves_gradients(
    node_type, monkeypatch
):
    import copy

    torch.manual_seed(73)
    node = node_type(step_mode="m", store_v_seq=True)
    reference = copy.deepcopy(node)
    reference.step_mode = "s"
    x = torch.rand(4, 2, 3, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    v = torch.rand(2, 3, requires_grad=True)
    vr = v.detach().clone().requires_grad_()
    node.v, reference.v = v, vr
    spikes = torch.stack([reference(step) for step in xr])

    name = {
        neuron.IFNode: "if_multi_step",
        neuron.LIFNode: "lif_multi_step",
        neuron.ParametricLIFNode: "plif_multi_step",
    }[node_type]
    entry = getattr(functional, name)
    shapes = []

    def observe(sequence, *args, **kwargs):
        shapes.append(sequence.shape)
        return entry(sequence, *args, **kwargs)

    monkeypatch.setattr(functional, name, observe)
    actual = node(x)
    assert shapes == [x.shape]
    for got, want in (
        (actual, spikes),
        (node.v, reference.v),
        (node.v_seq, reference.v_seq),
    ):
        torch.testing.assert_close(got, want)
    inputs = (x, v, node.w) if node_type is neuron.ParametricLIFNode else (x, v)
    refs = (xr, vr, reference.w) if node_type is neuron.ParametricLIFNode else (xr, vr)
    got = torch.autograd.grad(actual.sum() + node.v.sum() + node.v_seq.sum(), inputs)
    want = torch.autograd.grad(
        spikes.sum() + reference.v.sum() + reference.v_seq.sum(), refs
    )
    for g, r in zip(got, want, strict=True):
        torch.testing.assert_close(g, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "expand"])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("tau", [2.0, 3.7])
def test_clif_triton_matches_torch_states_and_all_gradients(
    dtype, layout, alpha, tau, monkeypatch
):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    torch.manual_seed(261014)
    shape = (1, 31) if layout == "expand" else (7, 31)
    inputs = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    actual_inputs = [x.clone().requires_grad_() for x in inputs]
    reference_inputs = [x.clone().requires_grad_() for x in inputs]

    def view(x):
        if layout == "transpose":
            return x.t()
        if layout == "expand":
            return x.expand(7, -1)
        return x

    spike_function = surrogate.Rect(alpha=alpha)
    actual = tuple(map(view, actual_inputs))
    reference = tuple(map(view, reference_inputs))
    actual_outputs, reference_outputs = [], []
    for _ in range(6):
        actual = functional.clif_step(*actual, tau, 1.0, spike_function)
        reference = clif_reference.step(*reference, tau, 1.0, spike_function)
        actual_outputs.extend(actual)
        reference_outputs.extend(reference)
    weights = [torch.randn_like(x) for x in actual_outputs]
    ag = torch.autograd.grad(
        sum((x * w).sum() for x, w in zip(actual_outputs, weights)), actual_inputs
    )
    rg = torch.autograd.grad(
        sum((x * w).sum() for x, w in zip(reference_outputs, weights)), reference_inputs
    )
    rtol, atol = (
        (1e-6, 1e-6)
        if dtype == torch.float32
        else (1e-3, 1e-5)
        if dtype == torch.float16
        else (1.6e-2, 1e-5)
    )
    for index, (a, r) in enumerate(zip(actual_outputs, reference_outputs)):
        torch.testing.assert_close(a, r, rtol=rtol, atol=atol)
        if index % 3 == 0:
            torch.testing.assert_close(a, r, rtol=0, atol=0)
    for a, r in zip(ag, rg):
        torch.testing.assert_close(a, r, rtol=rtol, atol=atol)
    assert (
        functional.neuron_implementation("clif", torch.device("cuda"))["implementation"]
        == "triton"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("create_graph", [False, True])
def test_clif_fused_higher_order_matches_standard_surrogates(
    monkeypatch,
    create_graph,
):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    for spike in (surrogate.Rect(), surrogate.ATan(), surrogate.Sigmoid()):
        inputs = [torch.tensor([0.5, 1.5, -0.2], device="cuda", requires_grad=True)]
        inputs += [torch.zeros_like(inputs[0], requires_grad=True) for _ in range(2)]
        reference = [x.detach().clone().requires_grad_() for x in inputs]
        actual = functional.clif_step(*inputs, 3.7, 1.0, spike)
        expected = clif_reference.step(*reference, 3.7, 1.0, spike)
        ga = torch.autograd.grad(
            sum(x.square().sum() for x in actual), inputs, create_graph=create_graph
        )
        gr = torch.autograd.grad(
            sum(x.square().sum() for x in expected),
            reference,
            create_graph=create_graph,
        )
        for a, r in zip(actual + ga, expected + gr):
            torch.testing.assert_close(a, r)
        if create_graph:
            gga = torch.autograd.grad(sum(x.sum() for x in ga), inputs)
            ggr = torch.autograd.grad(sum(x.sum() for x in gr), reference)
            for a, r in zip(gga, ggr):
                torch.testing.assert_close(a, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_clif_triton_module_reset_state_and_graph(monkeypatch):
    from spikingjelly._ops.clif import _selection

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    node = neuron.ComplementaryLIFNode(step_mode="m", store_state_seqs=True).cuda()
    x = torch.rand(6, 2, 31, device="cuda", requires_grad=True)
    first = node(x)
    first.sum().backward()
    assert node.state_seqs[0].shape == x.shape
    functional.reset_net(node)
    assert node.v == node.m == 0.0 and node.state_seqs is None
    torch.testing.assert_close(node(x), first)
    functional.reset_net(node)
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                node(x)
                functional.reset_net(node)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = node(x)
            functional.reset_net(node)
        graph.replay()
        torch.testing.assert_close(captured, first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_clif_triton_empty_outputs_and_unsupported_profile(monkeypatch):
    from spikingjelly._ops.clif import _selection

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    inputs = [torch.empty(0, device="cuda", requires_grad=True) for _ in range(3)]
    outputs = functional.clif_step(*inputs, 2.0, 1.0, surrogate.Rect())
    gradients = torch.autograd.grad(sum(x.sum() for x in outputs), inputs)
    assert all(x.numel() == 0 for x in outputs + gradients)
    with pytest.raises(RuntimeError, match="matching CUDA"):
        functional.clif_step(*(x.double() for x in inputs), 2.0, 1.0, surrogate.Rect())


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_clif_compile_preserves_reference_contract(device_type, monkeypatch):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "auto")
    inputs = [
        torch.randn(7, 31, device=device_type, requires_grad=True) for _ in range(3)
    ]
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    spike = surrogate.Rect()

    def step(x, v, m):
        return functional.clif_step(x, v, m, 2.0, 1.0, spike)

    actual = torch.compile(step, backend="aot_eager", fullgraph=True)(*inputs)
    expected = clif_reference.step(*reference, 2.0, 1.0, spike)
    ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
    gr = torch.autograd.grad(sum(x.sum() for x in expected), reference)
    for a, r in zip(actual + ga, expected + gr):
        torch.testing.assert_close(a, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clif_auto_uses_fused_training_and_inference(monkeypatch):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "auto")
    monkeypatch.setattr(_selection, "_selections", {})
    for training in (False, True):
        with torch.set_grad_enabled(training):
            inputs = [
                torch.randn(2, 31, device="cuda", requires_grad=training)
                for _ in range(3)
            ]
            refs = [x.detach().clone().requires_grad_(training) for x in inputs]
            actual = functional.clif_step(*inputs, 2.0, 1.0, surrogate.Rect())
            expected = clif_reference.step(*refs, 2.0, 1.0, surrogate.Rect())
            assert (
                functional.neuron_implementation("clif", inputs[0].device)[
                    "implementation"
                ]
                == "triton"
            )
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e)
            if training:
                ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
                gr = torch.autograd.grad(sum(x.sum() for x in expected), refs)
                for a, e in zip(ga, gr):
                    torch.testing.assert_close(a, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clif_custom_surrogate_keeps_reference_or_rejects_explicit_fusion(monkeypatch):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    inputs = [torch.randn(2, 31, device="cuda", requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    monkeypatch.setattr(_selection, "_requested", "auto")
    actual = functional.clif_step(*inputs, 2.0, 1.0, torch.sigmoid)
    expected = clif_reference.step(*refs, 2.0, 1.0, torch.sigmoid)
    ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs, create_graph=True)
    gr = torch.autograd.grad(sum(x.sum() for x in expected), refs, create_graph=True)
    for a, e in zip(actual + ga, expected + gr):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    monkeypatch.setattr(_selection, "_requested", "triton")
    with pytest.raises(RuntimeError, match="supported surrogate"):
        functional.clif_step(*inputs, 2.0, 1.0, torch.sigmoid)


@pytest.mark.parametrize("requested", ["auto", "torch", "triton", "cuda"])
def test_clif_diagnostics_reject_explicit_unsupported_profile(requested, monkeypatch):
    from spikingjelly._ops.clif import _selection

    monkeypatch.setattr(_selection, "_requested", requested)
    monkeypatch.setattr(torch.version, "hip", "test")
    execution = "eager"
    if requested in ("auto", "torch"):
        assert (
            functional.neuron_implementation(
                "clif", torch.device("cuda:0"), execution=execution
            )["implementation"]
            == "torch-reference"
        )
    else:
        with pytest.raises(RuntimeError, match="cannot run this execution profile"):
            functional.neuron_implementation(
                "clif", torch.device("cuda:0"), execution=execution
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "surrogate_class",
    [
        surrogate.Sigmoid,
        surrogate.ATan,
        surrogate.PiecewiseQuadratic,
        surrogate.PiecewiseExp,
        surrogate.SoftSign,
        surrogate.SuperSpike,
        surrogate.Erf,
    ],
)
def test_clif_fused_configurable_surrogate(dtype, surrogate_class, monkeypatch):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    torch.manual_seed(31)
    inputs = [
        torch.randn(3, 47, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    spike = surrogate_class(alpha=1.3)
    actual = functional.clif_step(*inputs, 3.7, 1.0, spike)
    expected = clif_reference.step(*refs, 3.7, 1.0, spike)
    weights = [torch.randn_like(x) for x in actual]
    ga = torch.autograd.grad(actual, inputs, weights)
    gr = torch.autograd.grad(expected, refs, weights)
    rtol, atol = (
        (1e-6, 1e-6)
        if dtype == torch.float32
        else (1e-3, 1e-5)
        if dtype == torch.float16
        else (1.6e-2, 1e-5)
    )
    for a, r in zip(actual + ga, expected + gr):
        torch.testing.assert_close(a, r, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clif_fused_rect_endpoints_and_heaviside_gradients(monkeypatch):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    for spike in (surrogate.Rect(), surrogate.heaviside):
        inputs = [torch.tensor([0.5, 1.5, 1.0], device="cuda", requires_grad=True)]
        inputs += [torch.zeros_like(inputs[0], requires_grad=True) for _ in range(2)]
        refs = [x.detach().clone().requires_grad_() for x in inputs]
        actual = functional.clif_step(*inputs, 2.0, 1.0, spike)
        expected = clif_reference.step(*refs, 2.0, 1.0, spike)
        assert actual[0].requires_grad == expected[0].requires_grad
        ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
        gr = torch.autograd.grad(sum(x.sum() for x in expected), refs)
        for a, r in zip(actual + ga, expected + gr):
            torch.testing.assert_close(a, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("implementation", ["triton", "torch"])
def test_clif_registered_opcheck_and_inductor(monkeypatch, implementation):
    from spikingjelly._ops.clif import _forward, _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", implementation)
    monkeypatch.setattr(_selection, "_selections", {})
    monkeypatch.setattr(_selection, "_compiled_selections", {})
    inputs = [torch.randn(3, 47, device="cuda", requires_grad=True) for _ in range(3)]
    x, v, m = inputs
    torch.library.opcheck(_forward, (x, v, v, m, 2.0, 1.0, 1.0, 7))
    spike = surrogate.Rect()

    def step(x, v, m):
        return functional.clif_step(x, v, m, 2.0, 1.0, spike)

    actual = torch.compile(step, fullgraph=True)(*inputs)
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    expected = clif_reference.step(*refs, 2.0, 1.0, spike)
    ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
    gr = torch.autograd.grad(sum(x.sum() for x in expected), refs)
    for a, r in zip(actual + ga, expected + gr):
        torch.testing.assert_close(a, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("create_graph", [False, True])
def test_clif_spike_only_loss_preserves_unused_complementary_state(
    monkeypatch, create_graph
):
    from spikingjelly._ops.clif import _selection, reference as clif_reference

    monkeypatch.setattr(_selection, "_requested", "triton")
    inputs = [torch.randn(2, 31, device="cuda", requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    spike = surrogate.Sigmoid()
    actual = functional.clif_step(*inputs, 2.0, 1.0, spike)[0]
    expected = clif_reference.step(*refs, 2.0, 1.0, spike)[0]
    ga = torch.autograd.grad(
        actual.sum(), inputs, allow_unused=True, create_graph=create_graph
    )
    gr = torch.autograd.grad(
        expected.sum(), refs, allow_unused=True, create_graph=create_graph
    )
    assert ga[2] is gr[2] is None
    for a, r in zip(ga[:2], gr[:2]):
        torch.testing.assert_close(a, r)
