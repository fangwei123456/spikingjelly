import math

import pytest
import torch

from spikingjelly.activation_based.functional import spike_similar_loss


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
@pytest.mark.parametrize("active", [False, True])
def test_linear_similarity_handles_silent_and_saturated_spikes(
    dtype, loss_type, active
):
    """Ordinary binary activity must give a finite loss and trainable gradients."""
    spikes = torch.full((2, 7), float(active), dtype=dtype, requires_grad=True)
    original = spikes.detach().clone()
    loss = spike_similar_loss(spikes, torch.eye(2), "linear", loss_type)
    loss.backward()

    assert loss.dtype == dtype
    assert loss.isfinite()
    assert spikes.grad.isfinite().all()
    assert torch.equal(spikes.detach(), original)
    if not active:
        expected = 50 if loss_type == "bce" else 0.5
        torch.testing.assert_close(loss, loss.new_tensor(expected))


def reference_loss(values, loss_type):
    """Scalar high-precision definition, independent of the tensor implementation."""
    losses = []
    for i, x in enumerate(values):
        for j, y in enumerate(values):
            dot = math.fsum(a * b for a, b in zip(x, y))
            norm_x = math.sqrt(math.fsum(a * a for a in x))
            norm_y = math.sqrt(math.fsum(b * b for b in y))
            probability = min(1.0, max(-1.0, dot / (norm_x * norm_y + 1e-8)))
            target = float(i == j)
            if loss_type == "mse":
                losses.append((probability - target) ** 2)
            elif loss_type == "l1":
                losses.append(abs(probability - target))
            else:
                positive = max(math.log(probability), -100) if probability else -100
                negative = (
                    max(math.log1p(-probability), -100) if probability < 1 else -100
                )
                losses.append(-target * positive - (1 - target) * negative)
    return math.fsum(losses) / len(losses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
def test_linear_similarity_matches_scalar_reference(dtype, loss_type):
    values = [[0.2, 0.7, 0.4], [0.9, 0.3, 0.1], [0.1, 0.2, 0.9]]
    spikes = torch.tensor(values, dtype=dtype, requires_grad=True)
    expected = reference_loss(spikes.detach().double().tolist(), loss_type)
    loss = spike_similar_loss(spikes, torch.eye(3), "linear", loss_type)
    torch.testing.assert_close(loss, loss.new_tensor(expected))


def test_linear_similarity_bce_preserves_gradient_near_one():
    """BCE must retain the small cosine gap hidden by float32 rounding."""
    spikes = torch.ones((2, 7), dtype=torch.float32, requires_grad=True)
    labels = torch.eye(2)
    loss = spike_similar_loss(spikes, labels, "linear", "bce")
    loss.backward()

    reference_spikes = spikes.detach().double().requires_grad_()
    reference_lengths = reference_spikes.norm(p=2, dim=1, keepdim=True)
    reference_similarity = (reference_spikes @ reference_spikes.t()) / (
        reference_lengths @ reference_lengths.t() + 1e-8
    )
    reference_loss = torch.nn.functional.binary_cross_entropy(
        reference_similarity, labels.double()
    )
    reference_loss.backward()

    torch.testing.assert_close(loss.double(), reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        spikes.grad.double(), reference_spikes.grad, rtol=1e-6, atol=1e-7
    )


def test_autocast_availability_is_checked_before_enabled_state(monkeypatch):
    """Unsupported devices must not be queried for their autocast state."""
    from spikingjelly.activation_based.functional.loss import (
        _is_autocast_enabled,
    )

    monkeypatch.setattr(torch.amp, "is_autocast_available", lambda _: False)
    monkeypatch.setattr(
        torch,
        "is_autocast_enabled",
        lambda *_: pytest.fail("unsupported device should not query autocast state"),
    )
    assert not _is_autocast_enabled("privateuseone")


@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
def test_linear_similarity_preserves_cpu_autocast_loss_dtype(loss_type):
    spikes = torch.tensor([[0.2, 0.7, 0.4], [0.9, 0.3, 0.1]])
    expected = reference_loss(spikes.double().tolist(), loss_type)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = spike_similar_loss(spikes, torch.eye(2), "linear", loss_type)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, loss.new_tensor(expected))


@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
def test_linear_similarity_does_not_reinterpret_signed_cosines(loss_type):
    spikes = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    if loss_type == "bce":
        with pytest.raises(RuntimeError, match="between 0 and 1"):
            spike_similar_loss(spikes, torch.eye(2), "linear", loss_type)
    else:
        loss = spike_similar_loss(spikes, torch.eye(2), "linear", loss_type)
        torch.testing.assert_close(loss, loss.new_tensor(0.5))


def test_linear_similarity_keeps_meta_shape_propagation():
    """An existing meta-device path must not require an autocast dispatch key."""
    loss = spike_similar_loss(
        torch.empty(2, 7, device="meta"), torch.empty(2, 2, device="meta")
    )
    assert loss.device.type == "meta"
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_linear_similarity_remains_fullgraph_traceable(autocast_enabled):
    """Precision management must remain capturable without a graph break."""
    compiled = torch.compile(spike_similar_loss, backend="eager", fullgraph=True)
    spikes = torch.tensor([[0.2, 0.7, 0.4], [0.9, 0.3, 0.1]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        loss = compiled(spikes, torch.eye(2))
    loss.backward()
    assert loss.isfinite()
    assert spikes.grad.isfinite().all()
