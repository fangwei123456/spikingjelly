import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.functional.loss import (
    kernel_dot_product,
    spike_similar_loss,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("label_dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
def test_spike_similarity_loss_dtype_and_gradient(dtype, label_dtype, loss_type):
    """Match pairwise loss values and spike gradients across label dtypes."""
    spikes = torch.tensor(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 1]]],
        dtype=dtype,
        requires_grad=True,
    )
    labels = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=label_dtype)
    original_labels = labels.clone()
    flat = spikes.flatten(1)
    similarity = torch.stack(
        [torch.stack([(x * y).sum().sigmoid() for y in flat]) for x in flat]
    )
    target = (labels.bool()[:, None] & labels.bool()[None, :]).any(-1).to(dtype)
    reference_loss = {
        "mse": F.mse_loss,
        "l1": F.l1_loss,
        "bce": F.binary_cross_entropy,
    }[loss_type]
    expected = reference_loss(similarity, target)
    expected_grad = torch.autograd.grad(expected, spikes, retain_graph=True)[0]

    actual = spike_similar_loss(spikes, labels, "sigmoid", loss_type, 1.0)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(spikes.grad, expected_grad)
    torch.testing.assert_close(labels, original_labels)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_kernel_roundoff_matches_pairwise_distance(
    dtype: torch.dtype,
) -> None:
    """A Gaussian self-kernel is one, even after distance roundoff."""
    x = torch.rand((2, 32), generator=torch.Generator().manual_seed(0)).to(dtype)
    x.requires_grad_()
    reference_x = x.detach().double().requires_grad_()
    difference = reference_x[:, None] - reference_x[None]
    expected = (-difference.square().sum(-1) / 2).exp()

    actual = kernel_dot_product(x, x, "gaussian", 1.0)

    assert (actual <= 1).all()
    torch.testing.assert_close(actual.diag(), torch.ones(2, dtype=dtype))
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)
    expected_grad = torch.autograd.grad(expected.sum(), reference_x)[0]
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(
        actual_grad.double(), expected_grad, rtol=1e-5, atol=1e-7
    )


def test_gaussian_spike_similarity_bce_roundoff() -> None:
    """Check a finite BCE and gradient for continuous spike representations."""
    spikes = torch.rand((2, 4, 8), generator=torch.Generator().manual_seed(0))
    spikes.requires_grad_()
    reference_spikes = spikes.detach().double().requires_grad_()
    probability = (
        -(reference_spikes[0] - reference_spikes[1]).square().sum() / 2
    ).exp()
    # Each self-pair has loss zero. Two different-label pairs share this loss.
    expected = -torch.log1p(-probability) / 2
    expected_grad = torch.autograd.grad(expected, reference_spikes)[0]

    actual = spike_similar_loss(spikes, torch.eye(2), "gaussian", "bce", 1.0)
    actual.backward()

    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        spikes.grad.double(), expected_grad, rtol=1e-5, atol=1e-7
    )
