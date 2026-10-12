import math

import numpy as np
import pytest
import torch

from spikingjelly.timing_based import functional, neuron
from spikingjelly.timing_based.examples.srm_mnist import _SRMNet


def _reference(times, indices, offsets, weights, duration, threshold):
    output_times, output_indices, output_offsets = [], [], [0]
    for batch in range(len(offsets) - 1):
        events = sorted(
            (float(times[e]), int(indices[e]))
            for e in range(offsets[batch], offsets[batch + 1])
            if 0 <= times[e] <= duration
        )
        for output, row in enumerate(weights):
            voltage, current, now = 0.0, 0.0, 0.0
            for arrival, connection in events + [(duration, None)]:
                while current > 0 and voltage + current * (arrival - now) >= threshold:
                    now += (threshold - voltage) / current
                    output_times.append(now)
                    output_indices.append(output)
                    voltage = 0.0
                voltage += current * (arrival - now)
                now = arrival
                if connection is not None:
                    current += float(row[connection])
        output_offsets.append(len(output_times))
    return output_times, output_indices, output_offsets


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_against_independent_sequential_reset_reference(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = np.random.default_rng(4)
    for _ in range(12):
        times = generator.uniform(-0.2, 3.4, size=16)
        times[2:4] = 0.4
        indices = generator.integers(0, 3, size=16)
        offsets = [0, 7, 7, 16]
        weights = generator.normal(0.5, 1.0, size=(3, 3))
        t = torch.tensor(times, device=device, dtype=dtype)
        w = torch.tensor(weights, device=device, dtype=dtype)
        expected = _reference(t.cpu().numpy(), indices, offsets, w.cpu().numpy(), 3, 1)
        actual = functional.srm(
            t,
            torch.tensor(indices, device=device),
            torch.tensor(offsets, device=device),
            w,
            3,
        )
        torch.testing.assert_close(
            actual[0],
            torch.tensor(expected[0], dtype=dtype, device=device),
            rtol=1e-5 if dtype == torch.float32 else 1e-9,
            atol=1e-6 if dtype == torch.float32 else 1e-10,
        )
        assert actual[1].tolist() == expected[1]
        assert actual[2].tolist() == expected[2]


def test_multiple_spikes_endpoint_and_output_before_same_time_input():
    times, _, offsets = functional.srm(
        torch.tensor([0.0, 0.5], dtype=torch.float64),
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([[2.0, -2.0]], dtype=torch.float64),
        2.0,
    )
    assert times.tolist() == [0.5]
    assert offsets.tolist() == [0, 1]
    times, _, _ = functional.srm(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        torch.tensor([[2.0]], dtype=torch.float64),
        2.0,
    )
    assert times.tolist() == [0.5, 1.0, 1.5, 2.0]


def test_shared_weight_time_gradcheck_with_unordered_events():
    times = torch.tensor([1.1, 0.13, 0.51, 0.8, 0.2], dtype=torch.float64)
    times.requires_grad_()
    weight = torch.tensor([[1.4, 0.9], [0.7, 1.3]], dtype=torch.float64)
    weight.requires_grad_()
    indices, offsets = torch.tensor([0, 0, 1, 0, 1]), torch.tensor([0, 3, 5])
    assert torch.autograd.gradcheck(
        lambda t, w: functional.srm(t, indices, offsets, w, 2.3)[0],
        (times, weight),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_empty_strided_and_outside_events_have_zero_gradient():
    times = torch.tensor([-1.0, 9.0, 0.25, 9.0, 3.0, 9.0], dtype=torch.float64)[::2]
    times.requires_grad_()
    weight = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    result = functional.srm(
        times, torch.zeros(3, dtype=torch.int64), torch.tensor([0, 3, 3]), weight, 2
    )
    result[0].sum().backward()
    assert times.grad.tolist() == [0.0, 3.0, 0.0]
    assert result[2].tolist() == [0, 3, 3]
    empty = functional.srm(
        times[:0], torch.empty(0, dtype=torch.int64), torch.tensor([0, 0, 0]), weight, 2
    )
    assert empty[0].numel() == 0 and empty[2].tolist() == [0, 0, 0]
    zero_batch = functional.srm(
        times[:0], torch.empty(0, dtype=torch.int64), torch.tensor([0]), weight, 2
    )
    assert zero_batch[2].tolist() == [0]


def test_capacity_raises_instead_of_truncating():
    model = neuron.SRM(1, 1, 2, max_spikes=2).double()
    with torch.no_grad():
        model.weight.fill_(2)
    with pytest.raises(RuntimeError, match="max_spikes"):
        model(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0]),
            torch.tensor([0, 1]),
        )


@pytest.mark.parametrize(
    "times,indices,offsets,weight,error",
    [
        ([math.nan], [0], [0, 1], [[1.0]], ValueError),
        ([0.0], [1], [0, 1], [[1.0]], ValueError),
        ([0.0], [0], [1, 1], [[1.0]], ValueError),
        ([0.0], [0], [0, 1, 0], [[1.0]], ValueError),
        ([0.0], [0], [0, 1], [[math.inf]], ValueError),
    ],
)
def test_invalid_event_inputs(times, indices, offsets, weight, error):
    with pytest.raises(error):
        functional.srm(
            torch.tensor(times),
            torch.tensor(indices),
            torch.tensor(offsets),
            torch.tensor(weight),
            2,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_two_layer_event_chain_gradcheck(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    times = torch.tensor(
        [0.13, 0.47], dtype=torch.float64, device=device, requires_grad=True
    )
    first = torch.tensor(
        [[1.4, 0.9], [0.7, 1.3]], dtype=torch.float64, device=device, requires_grad=True
    )
    second = torch.tensor(
        [[0.71, 0.83]], dtype=torch.float64, device=device, requires_grad=True
    )

    def network(t, w1, w2):
        events = functional.srm(
            t,
            torch.tensor([0, 1], device=device),
            torch.tensor([0, 2], device=device),
            w1,
            2.3,
        )
        return functional.srm(*events, w2, 2.3)[0]

    assert torch.autograd.gradcheck(
        network, (times, first, second), atol=1e-6, rtol=1e-4
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_demo_uses_both_trainable_srm_layers(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(3)
    model = _SRMNet(hidden_features=8).to(device)
    logits = model(torch.rand(2, 1, 28, 28, device=device))
    torch.nn.functional.cross_entropy(
        logits, torch.tensor([1, 4], device=device)
    ).backward()
    assert logits.shape == (2, 10)
    for layer in (model.hidden, model.output):
        assert isinstance(layer, neuron.SRM)
        gradient = layer.parametrizations.weight.original.grad
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_unsupported_time_dtype(dtype):
    with pytest.raises(TypeError):
        functional.srm(
            torch.zeros(1, dtype=dtype),
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.ones(1, 1, dtype=dtype),
            2,
        )


def test_root_smaller_than_fp64_time_resolution_raises():
    with pytest.raises(RuntimeError, match="representable"):
        functional.srm(
            torch.tensor([2e16], dtype=torch.float64),
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.ones(1, 1, dtype=torch.float64),
            2e16 + 8,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_finite_spike_with_overflowing_weight_gradient_raises(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    weight = torch.tensor([[1e-38]], device=device, requires_grad=True)
    times = functional.srm(
        torch.tensor([0.0], device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
        weight,
        1.1e38,
    )[0]
    assert torch.isfinite(times).all()
    with pytest.raises(RuntimeError, match="gradients overflow"):
        times.sum().backward()
