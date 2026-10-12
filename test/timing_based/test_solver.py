import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from spikingjelly.timing_based import functional, solver
from spikingjelly.timing_based.examples.cubic_srm_mnist import _CubicNet


def _linear(time, piece, start, slope):
    return slope * (time - start) - 1, slope.expand_as(time)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_linear_implicit_gradients_and_silence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    knots = torch.tensor([[0.0, 2.0], [0.0, 2.0]], dtype=torch.float64, device=device)
    start = torch.tensor(
        [[0.1], [0.2]], dtype=torch.float64, device=device, requires_grad=True
    )
    slope = torch.tensor(
        [[2.0], [0.1]], dtype=torch.float64, device=device, requires_grad=True
    )
    root = solver.first_spike_time(_linear, knots, (start, slope))
    torch.testing.assert_close(
        root, torch.tensor([0.6, torch.inf], dtype=torch.float64, device=device)
    )
    # Even an upstream gradient on inf must contribute zero on the silent branch.
    root.sum().backward()
    torch.testing.assert_close(
        start.grad, torch.tensor([[1.0], [0.0]], dtype=torch.float64, device=device)
    )
    torch.testing.assert_close(
        slope.grad, torch.tensor([[-0.25], [0.0]], dtype=torch.float64, device=device)
    )


def test_nonconstant_output_history_keeps_time_chain_rule():
    def response(t, piece, gain, history):
        history_response = torch.exp(history - t)
        return torch.exp(t) - gain - history_response, torch.exp(t) + history_response

    gain = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    history = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    knots = torch.tensor([0.0, 2.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda g, h: solver.first_spike_time(response, knots, (g, h)),
        (gain, history),
        atol=1e-6,
        rtol=1e-4,
    )
    root = solver.first_spike_time(response, knots, (gain, history))
    root.backward()
    assert gain.grad.item() > 0 and history.grad.item() > 0


def test_hidden_peak_tangent_endpoint_and_padding():
    def bump(t, piece, amplitude):
        return amplitude - (t - 1).square(), -2 * (t - 1)

    knots = torch.tensor([[0.0, 1.0, 2.0, 2.0]], dtype=torch.float64)
    flags = torch.tensor([[[False, True], [True, False], [False, False]]])
    amplitude = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(bump, knots, (amplitude,), flags)
    assert root.item() == pytest.approx(0.5, abs=1e-12)
    root.backward()
    assert amplitude.grad.item() == pytest.approx(-1.0, abs=1e-12)
    assert torch.isinf(
        solver.first_spike_time(bump, knots, (amplitude * 0,), flags)
    ).all()
    endpoint = solver.first_spike_time(
        _linear,
        torch.tensor([0.0, 0.5], dtype=torch.float64),
        (
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        ),
    )
    assert endpoint.item() == 0.5


def test_first_spike_second_backward_is_rejected():
    p = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        _linear, torch.tensor([0.0, 1.0], dtype=torch.float64), (p * 0, p)
    )
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(root, p, create_graph=True)


def test_invalid_partition_and_nonfinite_response():
    p = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="nondecreasing"):
        solver.first_spike_time(_linear, torch.tensor([1.0, 0.0]), (p, p))
    with pytest.raises(RuntimeError, match="non-finite"):
        solver.first_spike_time(
            lambda t, i: (t * torch.nan, t * 0), torch.tensor([0.0, 1.0]), ()
        )


def test_double_exponential_adapter_matches_tempotron_time_and_jacobian():
    slow, fast = 1.5, 0.375
    delta = 1 / fast - 1 / slow
    peak = np.log(slow / fast) / delta
    normalizer = np.exp(-peak / slow) * (-np.expm1(-delta * peak))

    def response(t, piece, arrival, weight):
        u = t - arrival
        decay, difference = torch.exp(-u / slow), -torch.expm1(-delta * u)
        h = decay * difference / normalizer
        dh = decay * (delta * torch.exp(-delta * u) - difference / slow) / normalizer
        return weight * h - 1, weight * dh

    t = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[1.8]], dtype=torch.float64, requires_grad=True)
    knots = torch.tensor([[0.2, 0.2 + peak, 4.0]], dtype=torch.float64)
    generic = solver.first_spike_time(
        response, knots, (t, w), torch.tensor([[[False, True], [True, False]]])
    )
    specific = functional.tempotron(
        t, torch.tensor([0]), torch.tensor([0, 1]), w, 4.0, slow, fast
    )[2]
    torch.testing.assert_close(generic, specific.flatten(), atol=1e-10, rtol=1e-9)
    generic_grads = torch.autograd.grad(generic.sum(), (t, w))
    specific_grads = torch.autograd.grad(specific.sum(), (t, w))
    for actual, expected in zip(generic_grads, specific_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


def test_knot_has_independent_one_sided_stationary_flags():
    def response(t, piece, parameter):
        left = -(t - 1).square()
        right = parameter * (t - 1)
        return torch.where(piece == 0, left, right), torch.where(
            piece == 0, -2 * (t - 1), parameter.expand_as(t)
        )

    p = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        response,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        (p,),
        torch.zeros(2, 2, dtype=torch.bool),
    )
    assert root.item() == 1.0
    with pytest.raises(RuntimeError, match="not transverse"):
        root.backward()


def test_cubic_simultaneous_events_keep_individual_time_gradients():
    times = torch.tensor([0.2, 0.2], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[0.7, 1.3]], dtype=torch.float64, requires_grad=True)
    output = functional.cubic_srm(
        times, torch.tensor([0, 1]), torch.tensor([0, 2]), weight, 3.0
    )
    output[0].sum().backward()
    torch.testing.assert_close(
        times.grad, torch.tensor([0.35, 0.65], dtype=torch.float64)
    )
    torch.testing.assert_close(weight.grad[:, 0], weight.grad[:, 1])


def _cubic_reference(times, indices, offsets, weights, duration, support):
    result, connections, boundaries = [], [], [0]
    for b in range(len(offsets) - 1):
        events = [
            (float(times[e]), int(indices[e]))
            for e in range(offsets[b], offsets[b + 1])
            if 0 <= times[e] <= duration
        ]
        knots = sorted(
            {
                0.0,
                duration,
                *(t for t, _ in events),
                *(t + support for t, _ in events if t + support < duration),
            }
        )
        for o, weights_row in enumerate(weights):

            def residual(t):
                return (
                    sum(
                        float(weights_row[i])
                        * 27
                        / 4
                        * ((t - a) / support)
                        * (1 - (t - a) / support) ** 2
                        for a, i in events
                        if 0 < t - a < support
                    )
                    - 1
                )

            found = False
            for lo, hi in zip(knots, knots[1:]):
                # Independent local polynomial construction, no prefix coefficient scan.
                p = np.polynomial.Polynomial([-1.0])
                for a, i in events:
                    if a <= lo < a + support:
                        z = np.polynomial.Polynomial([(lo - a) / support, 1 / support])
                        p += float(weights_row[i]) * 27 / 4 * z * (1 - z) ** 2
                critical = [
                    float(r.real) + lo
                    for r in p.deriv().roots()
                    if abs(r.imag) < 1e-12 and 0 < r.real < hi - lo
                ]
                points = [lo, *sorted(critical), hi]
                for left, right in zip(points, points[1:]):
                    value_left, value_right = residual(left), residual(right)
                    if value_left < 0 and value_right > 1e-12:
                        result.append(brentq(residual, left, right, xtol=1e-14))
                        connections.append(o)
                        found = True
                        break
                if found:
                    break
        boundaries.append(len(result))
    return result, connections, boundaries


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cubic_against_independent_polynomial_brent_reference(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = np.random.default_rng(13)
    for _ in range(12):
        t = torch.tensor(generator.uniform(-0.1, 2.5, 12), device=device, dtype=dtype)
        t[2:4] = 0.35
        indices = torch.tensor(generator.integers(0, 3, 12), device=device)
        w = torch.tensor(generator.normal(0.5, 0.9, (3, 3)), device=device, dtype=dtype)
        offsets = torch.tensor([0, 6, 6, 12], device=device)
        expected = _cubic_reference(
            t.cpu().numpy(),
            indices.cpu().numpy(),
            offsets.cpu().numpy(),
            w.cpu().numpy(),
            2.4,
            0.8,
        )
        actual = functional.cubic_srm(t, indices, offsets, w, 2.4, 0.8)
        torch.testing.assert_close(
            actual[0],
            torch.tensor(expected[0], device=device, dtype=dtype),
            rtol=1e-5 if dtype == torch.float32 else 1e-9,
            atol=1e-6 if dtype == torch.float32 else 1e-10,
        )
        assert actual[1].tolist() == expected[1]
        assert actual[2].tolist() == expected[2]


def test_cubic_shared_events_gradcheck_and_two_layers():
    t = torch.tensor([0.13, 0.37, 0.18], dtype=torch.float64, requires_grad=True)
    indices, offsets = torch.tensor([0, 1, 0]), torch.tensor([0, 3])
    w1 = torch.tensor([[0.8, 1.0], [0.9, 0.7]], dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor([[1.1, 1.3]], dtype=torch.float64, requires_grad=True)

    def network(times, first, second):
        events = functional.cubic_srm(times, indices, offsets, first, 3.0)
        return functional.cubic_srm(*events, second, 3.0)[0]

    assert torch.autograd.gradcheck(
        network, (t, w1, w2), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_cubic_tangent_late_peak_empty_outside_and_signed_inputs():
    t = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    i, off = torch.tensor([0, 1]), torch.tensor([0, 2, 2])
    w = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    events = functional.cubic_srm(t, i, off, w, 4.0, support=1.0)
    assert events[0].item() > 2.0  # First PSP only grazes; the later PSP fires.
    assert events[2].tolist() == [0, 1, 1]
    events[0].sum().backward()
    torch.testing.assert_close(
        t.grad, torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-10, rtol=1e-9
    )
    empty = functional.cubic_srm(t[:0], i[:0], torch.tensor([0, 0]), w, 4.0)
    assert empty[0].numel() == 0 and empty[2].tolist() == [0, 0]
    outside = functional.cubic_srm(t + 8, i, off, w, 4.0)
    assert outside[0].numel() == 0
    cancelling = functional.cubic_srm(
        t * 0, i, off, w * torch.tensor([[1.0, -0.5]]), 4.0
    )
    assert cancelling[0].numel() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cubic_demo_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(9)
    model = _CubicNet(4).to(device)
    logits = model(torch.rand(2, 1, 28, 28, device=device))
    torch.nn.functional.cross_entropy(
        logits, torch.tensor([1, 4], device=device)
    ).backward()
    for p in model.parameters():
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("shift", [0.0, 99.0, 999.0, 1e6])
def test_cubic_translation_does_not_turn_tangent_into_spike(shift):
    times = torch.tensor([shift], dtype=torch.float64, requires_grad=True)
    indices, offsets = torch.tensor([0]), torch.tensor([0, 1])
    tangent = functional.cubic_srm(
        times,
        indices,
        offsets,
        torch.ones(1, 1, dtype=torch.float64),
        shift + 2,
        support=1,
    )
    assert tangent[0].numel() == 0
    weights = torch.tensor([[1.2]], dtype=torch.float64, requires_grad=True)
    shifted = functional.cubic_srm(
        times, indices, offsets, weights, shift + 2, support=1
    )[0]
    baseline = functional.cubic_srm(
        times * 0, indices, offsets, weights, 2.0, support=1
    )[0]
    torch.testing.assert_close(shifted - shift, baseline, atol=1e-9, rtol=1e-9)
    shifted.sum().backward()
    torch.testing.assert_close(times.grad, torch.ones_like(times), atol=1e-9, rtol=1e-9)


def test_horizontal_upcrossing_emits_but_rejects_singular_gradient():
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        lambda t, i, p: ((t - p).pow(3), 3 * (t - p).square()),
        torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float64),
        (p,),
        torch.zeros(3, 2, dtype=torch.bool),
    )
    assert root.item() == 1.0
    with pytest.raises(RuntimeError, match="not transverse"):
        root.backward()


def test_small_representable_root_has_relative_time_and_gradient_accuracy():
    weight = torch.tensor([1e38], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        lambda t, i, w: (t * w - 1, w.expand_as(t)),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        (weight,),
    )
    torch.testing.assert_close(root, weight[0].reciprocal(), atol=0, rtol=1e-14)
    root.backward()
    torch.testing.assert_close(
        weight.grad, -weight.detach().pow(-2), atol=0, rtol=1e-14
    )


def test_expired_cubic_support_has_exact_zero_state_in_long_empty_tail():
    for n, duration in ((40, 1e8), (1000, 1e6)):
        times = torch.arange(n, dtype=torch.float64) * 1.2 + 0.123
        output = functional.cubic_srm(
            times,
            torch.zeros(n, dtype=torch.int64),
            torch.tensor([0, n]),
            torch.tensor([[0.9]], dtype=torch.float64),
            duration,
            support=1.0,
        )
        assert output[0].numel() == 0


def test_separated_supports_restart_gradients_without_expired_history():
    times = torch.tensor([0.1, 5.2], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[0.9, 1.3]], dtype=torch.float64, requires_grad=True)
    indices, offsets = torch.tensor([0, 1]), torch.tensor([0, 2])
    assert torch.autograd.gradcheck(
        lambda t, w: functional.cubic_srm(t, indices, offsets, w, 7.0, support=1.0)[0],
        (times, weight),
        atol=1e-6,
        rtol=1e-4,
    )
    functional.cubic_srm(times, indices, offsets, weight, 7.0, support=1.0)[
        0
    ].sum().backward()
    torch.testing.assert_close(
        times.grad[:1], torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        weight.grad[:, :1], torch.zeros(1, 1, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_general_solver_reports_cast_gradient_overflow():
    weight = torch.tensor([1e-38], requires_grad=True)
    root = solver.first_spike_time(
        lambda t, i, w: (t * w - 1, w.expand_as(t)),
        torch.tensor([0.0, 1.1e38], dtype=torch.float64),
        (weight,),
    )
    with pytest.raises(RuntimeError, match="overflow"):
        root.backward()


def test_horizontal_upcrossing_at_window_endpoint_uses_certified_type():
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        lambda t, i, p: ((t - p).pow(3), 3 * (t - p).square()),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        (p,),
        extrema=torch.tensor([[False, False]]),
    )
    assert root.item() == 1.0
    with pytest.raises(RuntimeError, match="not transverse"):
        root.backward()


def test_cubic_knot_crossing_uses_earlier_branch_with_zero_slope():
    times = torch.tensor([0.0, 1 / 3], dtype=torch.float64, requires_grad=True)
    output = functional.cubic_srm(
        times,
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.ones(1, 2, dtype=torch.float64),
        2.0,
        support=1.0,
    )[0]
    torch.testing.assert_close(
        output, torch.tensor([1 / 3], dtype=torch.float64), atol=1e-14, rtol=1e-14
    )
    with pytest.raises(RuntimeError, match="not transverse"):
        output.sum().backward()


@pytest.mark.parametrize("support", [1.0, 4.0])
def test_cubic_unrepresentable_relative_support_or_stationary_time_raises(support):
    with pytest.raises(RuntimeError, match="representable"):
        functional.cubic_srm(
            torch.tensor([2e16], dtype=torch.float64),
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.tensor([[2.0]], dtype=torch.float64),
            2e16 + 8,
            support=support,
        )


def test_composite_parameters_cannot_silently_return_partial_second_derivative():
    weight = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    root = solver.first_spike_time(
        lambda t, i, p: (t * p - 1, p.expand_as(t)),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        (weight.square(),),
    )
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(root, weight, create_graph=True)


def test_inactive_rows_do_not_evaluate_nonfinite_dummy_midpoints():
    weight = torch.tensor([[0.0], [2.0]], dtype=torch.float64, requires_grad=True)
    roots = solver.first_spike_time(
        _linear,
        torch.tensor([[-1e308, 1e308], [0.0, 1.0]], dtype=torch.float64),
        (torch.zeros_like(weight), weight),
    )
    torch.testing.assert_close(
        roots, torch.tensor([torch.inf, 0.5], dtype=torch.float64)
    )
    roots.sum().backward()
    torch.testing.assert_close(
        weight.grad, torch.tensor([[0.0], [-0.25]], dtype=torch.float64)
    )
