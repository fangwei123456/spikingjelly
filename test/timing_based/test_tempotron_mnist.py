import pytest
import torch

from spikingjelly.timing_based.examples.tempotron_mnist import _classification_loss
from spikingjelly.timing_based.neuron import Tempotron


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_recovers_late_evidence_after_shared_premature_spikes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = Tempotron(3, 2, 5.0, tau=2.0, tau_s=0.5).to(
        device=device, dtype=torch.float64
    )
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[1.2, 1.0, -1.0], [1.2, -1.0, 1.0]], device=device)
        )
    times = torch.tensor([0.0, 3.0, 0.0, 3.0], dtype=torch.float64, device=device)
    indices = torch.tensor([0, 1, 0, 2], device=device)
    offsets = torch.tensor([0, 2, 4], device=device)
    labels = torch.tensor([0, 1], device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(20):
        maximum, _, spike = model(times, indices, offsets)
        loss = _classification_loss(maximum, spike, labels, model.v_threshold)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    maximum, _, spike = model(times, indices, offsets)
    assert torch.equal(maximum.argmax(1), labels)
    assert torch.equal(
        torch.isfinite(spike), torch.eye(2, dtype=torch.bool, device=device)
    )
    assert model.weight[:, 0].max().item() < model.v_threshold
