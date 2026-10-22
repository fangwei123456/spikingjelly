import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.functional.loss import spike_similar_loss


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
