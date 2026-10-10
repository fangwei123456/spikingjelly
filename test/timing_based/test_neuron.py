import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from spikingjelly.timing_based.neuron import Tempotron


def _ode_reference(times, indices, weights, duration, tau, tau_s, threshold):
    """Integrate physical current/voltage; do not use event extrema formulas."""
    delay = tau * tau_s * math.log(tau / tau_s) / (tau - tau_s)
    delta = 1 / tau_s - 1 / tau
    normalizer = math.exp(-delay / tau) * (-math.expm1(-delta * delay)) / delta
    events = sorted(
        (float(t), int(i)) for t, i in zip(times, indices) if 0 <= t <= duration
    )
    grouped = {}
    for t, i in events:
        grouped[t] = grouped.get(t, 0.0) + float(weights[i]) / normalizer
    voltage_max, peak_time, spike_time = 0.0, 0.0, math.inf
    state = np.zeros(2)
    start = 0.0
    fired = False

    def dynamics(t, y):
        return [-y[0] / tau + y[1], -y[1] / tau_s]

    def crossing(t, y):
        return y[0] - threshold

    crossing.direction = 1
    crossing.terminal = True
    boundaries = sorted(set([0.0, duration, *grouped]))
    for end in boundaries[1:]:
        if not fired:
            state[1] += grouped.get(start, 0.0)
        while start < end:
            solution = solve_ivp(
                dynamics,
                (start, end),
                state,
                dense_output=True,
                events=None if fired else crossing,
                rtol=3e-13,
                atol=3e-14,
            )
            stop = solution.t[-1]
            candidates = [start, stop]
            left_slope = dynamics(start, solution.sol(start))[0]
            right_slope = dynamics(stop, solution.sol(stop))[0]
            if left_slope * right_slope < 0:
                candidates.append(
                    brentq(
                        lambda t: dynamics(t, solution.sol(t))[0],
                        start,
                        stop,
                        xtol=1e-13,
                    )
                )
            for t in sorted(candidates):
                v = float(solution.sol(t)[0])
                if v > voltage_max:
                    voltage_max, peak_time = v, t
            state = solution.y[:, -1].copy()
            start = stop
            if not fired and solution.t_events[0].size:
                spike_time = start
                fired = True
            elif start < end:
                raise AssertionError("Reference integration did not reach its endpoint")
        start = end
    return voltage_max, peak_time, spike_time


def test_single_event_peak_and_first_spike():
    neuron = Tempotron(1, 1, duration=5.0, tau=2.0, tau_s=0.5, v_threshold=0.6).double()
    with torch.no_grad():
        neuron.weight.fill_(1.0)
    times = torch.tensor([0.3], dtype=torch.float64)
    indices = torch.tensor([0])
    offsets = torch.tensor([0, 1])
    actual = neuron(times, indices, offsets)
    expected = _ode_reference([0.3], [0], [1.0], 5.0, 2.0, 0.5, 0.6)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(
            value, value.new_tensor([[reference]]), rtol=1e-9, atol=1e-10
        )
    assert actual[2].item() < actual[1].item()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multievent_shunting_matches_ode(device, dtype):
    times = torch.tensor(
        [1.1, 0.2, 0.5, 4.0, 0.3, 0.8, -1.0], dtype=dtype, device=device
    )
    indices = torch.tensor([1, 0, 0, 2, 0, 1, 2], device=device)
    offsets = torch.tensor([0, 4, 7, 7], device=device)
    neuron = Tempotron(3, 2, 5.0, tau=2.0, tau_s=0.5, v_threshold=0.6).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        neuron.weight.copy_(
            torch.tensor(
                [[0.9, 0.8, -5.0], [0.08, 0.1, 0.15]], device=device, dtype=dtype
            )
        )
    actual = neuron(times, indices, offsets)
    for b, (start, end) in enumerate(zip(offsets.tolist()[:-1], offsets.tolist()[1:])):
        for o in range(2):
            expected = _ode_reference(
                times[start:end].tolist(),
                indices[start:end].tolist(),
                neuron.weight[o].tolist(),
                5.0,
                2.0,
                0.5,
                0.6,
            )
            for value, reference in zip(actual, expected):
                tolerance = (
                    dict(rtol=1e-5, atol=1e-6)
                    if dtype == torch.float32
                    else dict(rtol=1e-9, atol=1e-10)
                )
                torch.testing.assert_close(
                    value[b, o], value.new_tensor(reference), **tolerance
                )


def test_joint_gradcheck_shared_connections(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor(
        [0.9, 0.2, 4.0, 0.7], dtype=torch.float64, device=device, requires_grad=True
    )
    indices = torch.tensor([1, 0, 1, 0], device=device)
    offsets = torch.tensor([0, 4], device=device)
    weight = torch.tensor(
        [[0.8, 0.7], [0.1, 0.2]], dtype=torch.float64, device=device, requires_grad=True
    )

    def evaluate(t, w):
        maximum, peak, spike = tempotron(t, indices, offsets, w, 5.0, 2.0, 0.5, 0.6)
        return maximum, peak, spike[:, :1]

    assert torch.autograd.gradcheck(
        evaluate, (times, weight), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_event_boundary_peak_gradcheck(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor(
        [0.2, 0.7], dtype=torch.float64, device=device, requires_grad=True
    )
    indices = torch.tensor([0, 1], device=device)
    offsets = torch.tensor([0, 2], device=device)
    weight = torch.tensor(
        [[1.0, -10.0]], dtype=torch.float64, device=device, requires_grad=True
    )

    def evaluate(t, w):
        maximum, peak, _ = tempotron(t, indices, offsets, w, 3.0, 2.0, 0.5, 3.0)
        return maximum, peak

    assert torch.autograd.gradcheck(
        evaluate, (times, weight), eps=1e-6, atol=1e-6, rtol=1e-4
    )
    assert evaluate(times, weight)[1].item() == pytest.approx(0.7)


@pytest.mark.parametrize(
    "tau,tau_s", [(2.0, 0.5), (0.5, 2.0), (1.00000001, 1.0), (1.000000000001, 1.0)]
)
def test_unit_peak_and_grazing(device, tau, tau_s):
    neuron = Tempotron(1, 1, 5.0, tau=tau, tau_s=tau_s).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.fill_(1.0)
    maximum, peak, spike = neuron(
        torch.tensor([0.3], dtype=torch.float64, device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    slow, fast = max(tau, tau_s), min(tau, tau_s)
    expected_delay = fast * math.log1p((slow - fast) / fast) / ((slow - fast) / slow)
    assert maximum.item() == pytest.approx(1.0, abs=2e-15)
    assert peak.item() == pytest.approx(0.3 + expected_delay, abs=1e-12)
    assert torch.isinf(spike).all()


def test_same_time_inputs_are_shunted(device):
    neuron = Tempotron(2, 1, 5.0, tau=2.0, tau_s=0.5, v_threshold=0.6).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.copy_(
            torch.tensor([[1.0, 10.0]], dtype=torch.float64, device=device)
        )
    first = neuron(
        torch.tensor([0.3], dtype=torch.float64, device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    times = torch.tensor(
        [0.3, first[2].item(), 3.0],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    result = neuron(
        times,
        torch.tensor([0, 1, 1], device=device),
        torch.tensor([0, 3], device=device),
    )
    torch.testing.assert_close(result[0], first[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result[2], first[2], rtol=1e-12, atol=1e-12)
    sum(v.sum() for v in result).backward()
    assert torch.equal(times.grad[1:], torch.zeros_like(times.grad[1:]))
    assert neuron.weight.grad[0, 1].item() == 0


def test_empty_trials_and_outside_events(device):
    neuron = Tempotron(1, 1, 2.0).to(device=device, dtype=torch.float64)
    times = torch.tensor(
        [-1.0, 3.0], dtype=torch.float64, device=device, requires_grad=True
    )
    maximum, peak, spike = neuron(
        times,
        torch.tensor([0, 0], device=device),
        torch.tensor([0, 0, 2, 2], device=device),
    )
    assert maximum.shape == (3, 1)
    assert torch.equal(maximum, torch.zeros_like(maximum))
    assert torch.equal(peak, torch.zeros_like(peak))
    assert torch.isinf(spike).all()
    (maximum.sum() + peak.sum() + spike.sum()).backward()
    assert torch.equal(times.grad, torch.zeros_like(times))
    assert torch.equal(neuron.weight.grad, torch.zeros_like(neuron.weight))


def test_window_endpoint_peak_and_crossing(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
    indices, offsets = (
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    weight = torch.tensor(
        [[1.0]], dtype=torch.float64, device=device, requires_grad=True
    )
    value = tempotron(times, indices, offsets, weight, 0.4, 2.0, 0.5, 3.0)
    assert value[1].item() == 0.4
    value[1].sum().backward()
    assert torch.equal(times.grad, torch.zeros_like(times))
    assert torch.equal(weight.grad, torch.zeros_like(weight))
    threshold = value[0].item()
    endpoint = tempotron(times, indices, offsets, weight, 0.4, 2.0, 0.5, threshold)
    assert endpoint[2].item() == 0.4


def test_rejects_second_order_backward():
    neuron = Tempotron(1, 1, 5.0, tau=2.0, tau_s=0.5, v_threshold=0.6).double()
    with torch.no_grad():
        neuron.weight.fill_(1.0)
    spike = neuron(
        torch.tensor([0.3], dtype=torch.float64),
        torch.tensor([0]),
        torch.tensor([0, 1]),
    )[2]
    first = torch.autograd.grad(spike.sum(), neuron.weight, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(first.sum(), neuron.weight)


@pytest.mark.parametrize(
    "bad,exception",
    [
        ("offset_start", ValueError),
        ("offset_end", ValueError),
        ("offset_order", ValueError),
        ("index", ValueError),
        ("time_nan", ValueError),
        ("weight_inf", ValueError),
        ("shape", ValueError),
        ("index_dtype", TypeError),
        ("dtype", TypeError),
    ],
)
def test_invalid_event_contract(bad, exception):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor([0.3], dtype=torch.float64)
    indices, offsets = torch.tensor([0]), torch.tensor([0, 1])
    weight = torch.ones(1, 1, dtype=torch.float64)
    if bad == "offset_start":
        offsets = torch.tensor([1, 1])
    elif bad == "offset_end":
        offsets = torch.tensor([0, 2])
    elif bad == "offset_order":
        offsets = torch.tensor([0, 1, 0, 1])
    elif bad == "index":
        indices = torch.tensor([-1])
    elif bad == "time_nan":
        times.fill_(torch.nan)
    elif bad == "weight_inf":
        weight.fill_(torch.inf)
    elif bad == "shape":
        times = times.unsqueeze(0)
    elif bad == "index_dtype":
        indices = indices.int()
    elif bad == "dtype":
        times, weight = times.half(), weight.half()
    with pytest.raises(exception):
        tempotron(times, indices, offsets, weight, 5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0.0},
        {"tau": 0.0},
        {"tau": 1.0, "tau_s": 1.0},
        {"tau": float("nan")},
        {"v_threshold": -1.0},
    ],
)
def test_invalid_model_parameters(kwargs):
    params = {"duration": 5.0, **kwargs}
    with pytest.raises(ValueError):
        Tempotron(1, 1, **params)


def test_noncontiguous_inputs_and_weights(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor([0.3, 99.0, 0.8, 99.0], dtype=torch.float64, device=device)[
        ::2
    ]
    indices = torch.tensor([0, 99, 1, 99], device=device)[::2]
    offsets = torch.tensor([0, 99, 2, 99], device=device)[::2]
    weight = torch.tensor(
        [[0.9, 0.1], [0.7, 0.2]], dtype=torch.float64, device=device
    ).t()
    actual = tempotron(times, indices, offsets, weight, 5.0, 2.0, 0.5, 0.6)
    contiguous = tempotron(
        times.contiguous(),
        indices.contiguous(),
        offsets.contiguous(),
        weight.contiguous(),
        5.0,
        2.0,
        0.5,
        0.6,
    )
    for a, b in zip(actual, contiguous):
        torch.testing.assert_close(a, b)


def test_simultaneous_inputs_gradcheck(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor(
        [0.4, 0.4], dtype=torch.float64, device=device, requires_grad=True
    )
    indices = torch.tensor([0, 1], device=device)
    offsets = torch.tensor([0, 2], device=device)
    weight = torch.tensor(
        [[0.4, 0.7]], dtype=torch.float64, device=device, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda t, w: tempotron(t, indices, offsets, w, 5.0, 2.0, 0.5, 0.6),
        (times, weight),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_platform_selects_earliest_peak_and_empty_batch(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.tensor(
        [0.0, 0.5], dtype=torch.float64, device=device, requires_grad=True
    )
    weight = torch.zeros(1, 1, dtype=torch.float64, device=device, requires_grad=True)
    result = tempotron(
        times,
        torch.tensor([0, 0], device=device),
        torch.tensor([0, 2], device=device),
        weight,
        2.0,
    )
    assert result[0].item() == result[1].item() == 0
    (result[0] + result[1]).sum().backward()
    assert torch.equal(times.grad, torch.zeros_like(times))
    assert torch.equal(weight.grad, torch.zeros_like(weight))
    empty = tempotron(
        times[:0],
        torch.empty(0, dtype=torch.int64, device=device),
        torch.tensor([0], device=device),
        weight,
        2.0,
    )
    assert all(x.shape == (0, 1) for x in empty)


@pytest.mark.parametrize("threshold", [1.0, 0.6])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_grazing_and_adjacent_weights_are_distinct(device, threshold, direction):
    weight = torch.tensor(threshold, dtype=torch.float64, device=device)
    if direction:
        weight = torch.nextafter(
            weight, weight.new_tensor(float("inf") if direction > 0 else -float("inf"))
        )
    neuron = Tempotron(1, 1, 5.0, tau=2.0, tau_s=0.5, v_threshold=threshold).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.fill_(weight)
    _, peak, spike = neuron(
        torch.tensor([0.3], dtype=torch.float64, device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    if direction > 0:
        assert torch.isfinite(spike).all() and spike.item() < peak.item()
    else:
        assert torch.isinf(spike).all()


def test_threshold_above_window_maximum_does_not_fire():
    neuron = Tempotron(1, 1, 0.4, tau=2.0, tau_s=0.5, v_threshold=3.0).double()
    with torch.no_grad():
        neuron.weight.fill_(1.0)
    args = (
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([0]),
        torch.tensor([0, 1]),
    )
    maximum = neuron(*args)[0]
    neuron.v_threshold = torch.nextafter(
        maximum, torch.full_like(maximum, torch.inf)
    ).item()
    assert torch.isinf(neuron(*args)[2]).all()


def test_seeded_signed_multievent_patterns_match_ode():
    from spikingjelly.timing_based.functional import tempotron

    generator = torch.Generator().manual_seed(20261007)
    for _ in range(8):
        times = torch.rand(8, generator=generator, dtype=torch.float64) * 4.5
        indices = torch.randint(3, (8,), generator=generator)
        weight = torch.randn(3, 3, generator=generator, dtype=torch.float64) * 0.7
        actual = tempotron(
            times, indices, torch.tensor([0, 8]), weight, 5.0, 2.0, 0.5, 0.6
        )
        for output in range(3):
            expected = _ode_reference(
                times.tolist(),
                indices.tolist(),
                weight[output].tolist(),
                5.0,
                2.0,
                0.5,
                0.6,
            )
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(
                    value[0, output], value.new_tensor(reference), rtol=1e-9, atol=1e-10
                )


@pytest.mark.parametrize("start", [0.0, 0.3])
def test_input_at_grazing_peak_can_create_a_later_upcrossing(device, start):
    neuron = Tempotron(2, 1, 5.0, tau=2.0, tau_s=0.5).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.copy_(
            torch.tensor([[1.0, 0.5]], device=device, dtype=torch.float64)
        )
    initial = neuron(
        torch.tensor([start], device=device, dtype=torch.float64),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    assert torch.isinf(initial[2]).all()
    arrival = initial[1].item()
    maximum, peak, spike = neuron(
        torch.tensor([start, arrival], device=device, dtype=torch.float64),
        torch.tensor([0, 1], device=device),
        torch.tensor([0, 2], device=device),
    )
    assert maximum.item() > 1
    assert torch.isfinite(spike).all() and arrival <= spike.item() < peak.item()


def test_ignored_event_at_peak_has_zero_time_gradient(device):
    neuron = Tempotron(2, 1, 5.0, tau=2.0, tau_s=0.5, v_threshold=0.6).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.copy_(
            torch.tensor([[1.0, 10.0]], device=device, dtype=torch.float64)
        )
    baseline = neuron(
        torch.tensor([0.0], device=device, dtype=torch.float64),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
    )
    times = torch.tensor(
        [0.0, baseline[1].item(), 3.0],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    result = neuron(
        times,
        torch.tensor([0, 1, 1], device=device),
        torch.tensor([0, 3], device=device),
    )
    for actual, expected in zip(result, baseline):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    sum(value.sum() for value in result).backward()
    assert torch.equal(times.grad[1:], torch.zeros_like(times.grad[1:]))
    assert neuron.weight.grad[0, 1].item() == 0


@pytest.mark.parametrize("tau,tau_s", [(2.0, 0.5), (1.000000000001, 1.0)])
def test_multiblock_crossing_and_gradients(device, tau, tau_s):
    from spikingjelly.timing_based.functional import tempotron

    base = torch.arange(260, dtype=torch.float64, device=device) * 0.1
    indices = torch.ones(260, dtype=torch.int64, device=device)
    indices[127:129] = 0
    offsets = torch.tensor([0, 260], device=device)
    arrivals = base[127:129].clone().requires_grad_()
    weight = torch.tensor(
        [[0.8, -0.0001]], dtype=torch.float64, device=device, requires_grad=True
    )

    def evaluate(t, w):
        times = torch.cat((base[:127], t, base[129:]))
        return tempotron(times, indices, offsets, w, 30.0, tau, tau_s, 0.6)

    result = evaluate(arrivals, weight)
    expected = _ode_reference(
        base.tolist(), indices.tolist(), weight[0].tolist(), 30.0, tau, tau_s, 0.6
    )
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(
            actual, actual.new_tensor([[reference]]), rtol=1e-9, atol=1e-10
        )
    assert result[2].item() > base[128].item()
    assert torch.autograd.gradcheck(
        evaluate, (arrivals, weight), eps=1e-6, atol=1e-6, rtol=1e-4, nondet_tol=1e-12
    )


def test_zero_current_groups_preserve_grazing_and_weight_gradients(device):
    from spikingjelly.timing_based.functional import tempotron

    times = torch.arange(260, dtype=torch.float64, device=device) * 0.01
    indices = torch.ones(260, dtype=torch.int64, device=device)
    indices[0] = 0
    offsets = torch.tensor([0, 260], device=device)
    weight = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float64, device=device, requires_grad=True
    )
    result = tempotron(times, indices, offsets, weight, 5.0, 1.000000000001, 1.0)
    assert result[0].item() == 1.0
    assert torch.isinf(result[2]).all()
    result[0].sum().backward()
    delay = (result[1].detach() - times[1:]).clamp_min(0)
    tau = 1.000000000001
    delta = (tau - 1.0) / tau
    h = torch.exp(-delay / tau) * (-torch.expm1(-delta * delay)) / delta
    peak = (
        torch.log1p(torch.tensor(delta * tau, device=device, dtype=torch.float64))
        / delta
    )
    normalizer = torch.exp(-peak / tau) * (-torch.expm1(-delta * peak)) / delta
    torch.testing.assert_close(weight.grad[0, 1], (h / normalizer).sum())


def test_shunted_overflowing_groups_do_not_contaminate_accepted_prefix(device):
    from spikingjelly.timing_based.functional import tempotron

    weight = torch.tensor([[1.0, 1e308]], dtype=torch.float64, device=device)
    baseline = tempotron(
        torch.tensor([0.0], dtype=torch.float64, device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
        weight,
        5.0,
        2.0,
        0.5,
        0.6,
    )
    actual = tempotron(
        torch.tensor([0.0, 4.0, 4.0], dtype=torch.float64, device=device),
        torch.tensor([0, 1, 1], device=device),
        torch.tensor([0, 3], device=device),
        weight,
        5.0,
        2.0,
        0.5,
        0.6,
    )
    for result, expected in zip(actual, baseline):
        torch.testing.assert_close(result, expected)
    with pytest.raises(RuntimeError, match="not finite"):
        tempotron(
            torch.tensor([0.0, 0.0], dtype=torch.float64, device=device),
            torch.tensor([1, 1], device=device),
            torch.tensor([0, 2], device=device),
            weight,
            5.0,
            2.0,
            0.5,
            0.6,
        )
