import pytest
import torch
from torch import nn

from spikingjelly.timing_based.encoding import GaussianTuning
from spikingjelly.timing_based.neuron import Tempotron


def test_continuous_events_keep_receptive_fields_and_index_order():
    encoder = GaussianTuning(1, 4, torch.tensor([0.0]), torch.tensor([1.0]))
    assert isinstance(encoder, nn.Module)
    torch.testing.assert_close(encoder.mu, torch.tensor([[-0.25, 0.25, 0.75, 1.25]]))
    torch.testing.assert_close(encoder.sigma2, torch.full((1, 4), 1 / 9))
    times, indices, offsets = encoder(torch.tensor([[[0.1, 0.5, 0.9]]]), duration=100.0)
    assert offsets.tolist() == [0, 12]
    assert indices.tolist() == list(range(12))
    expected = []
    for x in (0.1, 0.5, 0.9):
        for center in (-0.25, 0.25, 0.75, 1.25):
            expected.append(
                100 * (1 - torch.exp(torch.tensor(-((x - center) ** 2) / (2 / 9))))
            )
    torch.testing.assert_close(times, torch.stack(expected), atol=1e-5, rtol=1e-6)
    assert bool((times != times.round()).any())
    assert set(encoder.state_dict()) == {"mu", "sigma2"}
    assert not list(encoder.parameters())


def test_encoder_empty_events_and_multiple_batches():
    encoder = GaussianTuning(
        2, 5, torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0])
    ).double()
    x = torch.full((3, 2, 4), 1e6, dtype=torch.float64, requires_grad=True)
    times, indices, offsets = encoder(x)
    assert times.shape == indices.shape == (0,)
    assert offsets.tolist() == [0, 0, 0, 0]
    times.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_encoder_input_gradcheck():
    encoder = GaussianTuning(
        2, 4, torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0])
    ).double()
    x = torch.tensor(
        [[[0.2, 0.7], [-0.3, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda values: encoder(values, 5.0)[0], (x,))
    times, indices, offsets = encoder(x, 5.0)
    assert indices.tolist() == list(range(16))
    assert offsets.tolist() == [0, 16]
    assert times.dtype == torch.float64


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encoder_neuron_user_loss_training(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1)).to(
        device=device, dtype=torch.float64
    )
    neuron = Tempotron(8, 2, 10.0, tau=2.0, tau_s=0.5, v_threshold=3.0).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        neuron.weight.copy_(
            torch.tensor([[0.2] * 8, [0.1] * 8], device=device, dtype=torch.float64)
        )
    x = torch.tensor(
        [[[0.2, 0.7]], [[0.4, 0.8]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    before = neuron.weight.detach().clone()
    optimizer = torch.optim.SGD(neuron.parameters(), lr=0.01)
    maximum, peak, spike = neuron(*encoder(x, 10.0))
    loss = torch.nn.functional.cross_entropy(
        maximum, torch.tensor([0, 1], device=device)
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
    assert torch.isfinite(neuron.weight.grad).all()
    assert not torch.equal(neuron.weight, before)
    assert maximum.shape == peak.shape == spike.shape == (2, 2)


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_encoder_invalid_duration(duration):
    encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError):
        encoder(torch.zeros(1, 1, 1), duration)


def test_encoder_rejects_invalid_shape_dtype_and_bounds():
    with pytest.raises(ValueError):
        GaussianTuning(1, 2, torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError):
        GaussianTuning(1, 4, torch.ones(1), torch.zeros(1))
    with pytest.raises(ValueError):
        GaussianTuning(1, 4, torch.zeros(2), torch.ones(2))
    encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError):
        encoder(torch.zeros(1, 2, 1))
    with pytest.raises(TypeError):
        encoder(torch.zeros(1, 1, 1, dtype=torch.float64))
    with pytest.raises(ValueError):
        encoder(torch.full((1, 1, 1), torch.nan))


@pytest.mark.parametrize("dtype,value", [(torch.float32, 3e38), (torch.float64, 1e308)])
def test_far_finite_inputs_have_zero_finite_gradient(dtype, value):
    encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1)).to(dtype=dtype)
    x = torch.full((1, 1, 1), value, dtype=dtype, requires_grad=True)
    times, _, offsets = encoder(x)
    assert times.numel() == 0 and offsets.tolist() == [0, 0]
    times.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.equal(x.grad, torch.zeros_like(x))


def test_dtype_conversion_cannot_silently_destroy_receptive_fields():
    encoder = GaussianTuning(
        1,
        4,
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([1e-160], dtype=torch.float64),
    ).float()
    with pytest.raises(ValueError, match="Receptive"):
        encoder(torch.zeros(1, 1, 1))
