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


def _stbif_reference(x, q, acc_q, threshold, pos_max, neg_min):
    normalized = x / threshold
    q_next = q + normalized
    positive = (q_next >= 1) & (acc_q < pos_max)
    negative = (q_next < 0) & (acc_q > neg_min)
    current = positive.to(x.dtype) - negative.to(x.dtype)
    acc_q_next = acc_q + current
    q_next = torch.where(positive, q_next - 1, q_next)
    q_next = torch.where(negative, q_next + 1, q_next)
    is_work = bool((normalized != 0).any() | (current != 0).any())
    return current * threshold, q_next, acc_q_next, current, is_work


def test_stbif_step_and_scan_match_reference():
    x_seq = torch.randn(4, 2, 3)
    threshold = torch.tensor(0.25)
    pos_max = torch.tensor(1.0)
    neg_min = torch.tensor(-2.0)
    q = torch.full_like(x_seq[0], 0.5)
    acc_q = torch.zeros_like(q)

    scan = functional.stbif_scan_torch(x_seq, q, acc_q, threshold, pos_max, neg_min)
    expected_spikes = []
    q_ref = q
    acc_q_ref = acc_q
    current_ref = None
    is_work_ref = False
    for x in x_seq:
        step = functional.stbif_step(x, q_ref, acc_q_ref, threshold, pos_max, neg_min)
        reference = _stbif_reference(x, q_ref, acc_q_ref, threshold, pos_max, neg_min)
        for actual, expected in zip(step[:4], reference[:4]):
            _assert_close(actual, expected)
        expected_spikes.append(step[0])
        q_ref, acc_q_ref, current_ref = step[1:4]
        is_work_ref = is_work_ref or step[4]

    _assert_close(scan[0], torch.stack(expected_spikes))
    _assert_close(scan[1], q_ref)
    _assert_close(scan[2], acc_q_ref)
    _assert_close(scan[3], current_ref)
    assert scan[4] is is_work_ref


def test_scan_names_identify_independent_sequence_paths():
    backend_suffixes = ("_cupy", "_triton", "_inductor", "_torch")
    for name in functional_neuron.__all__:
        if "_scan" in name and name != "gated_lif_scan":
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
