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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_spike_similarity_bce_roundoff(dtype: torch.dtype) -> None:
    """Identical binary spike trains have a valid unit cosine similarity."""
    spikes = torch.ones((2, 1, 7), dtype=dtype, requires_grad=True)
    labels = torch.ones((2, 1), dtype=torch.bool)

    loss = spike_similar_loss(spikes, labels, "linear", "bce")

    # Seven common events give 7 / (7 + 1e-8), which rounds to 1 here.
    expected_probability = torch.tensor(7 / (7 + 1e-8), dtype=dtype)
    torch.testing.assert_close(loss, -expected_probability.log())
    assert loss.dtype == dtype
    loss.backward()
    torch.testing.assert_close(spikes.grad, torch.zeros_like(spikes))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("loss_type", ["mse", "l1", "bce"])
def test_linear_spike_similarity_matches_pairwise_reference(
    dtype: torch.dtype, loss_type: str
) -> None:
    """Check values and gradients against double-precision event overlap."""
    data = [[[1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 0, 0]], [[0, 0, 0, 0, 1, 1, 1]]]
    spikes = torch.tensor(data, dtype=dtype, requires_grad=True)
    reference_spikes = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, 0], [1, 1], [0, 1]])
    target = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float64)
    vectors = reference_spikes.flatten(1)
    probabilities = torch.stack(
        [
            torch.stack(
                [
                    (x * y).sum()
                    / ((x.square().sum() * y.square().sum()).sqrt() + 1e-8)
                    for y in vectors
                ]
            )
            for x in vectors
        ]
    )
    if loss_type == "mse":
        expected = (probabilities - target).square().mean()
    elif loss_type == "l1":
        expected = (probabilities - target).abs().mean()
    else:
        expected = -(
            target * probabilities.log() + (1 - target) * torch.log1p(-probabilities)
        ).mean()
    expected_grad = torch.autograd.grad(expected, reference_spikes)[0]

    actual = spike_similar_loss(spikes, labels, "linear", loss_type)
    actual.backward()

    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        spikes.grad.double(), expected_grad, rtol=1e-5, atol=1e-7
    )
    assert actual.dtype == dtype
