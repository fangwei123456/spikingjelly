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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "features", "autocast_enabled"),
    [
        (torch.float16, 32768, False),
        (torch.bfloat16, 2048, False),
        (torch.float16, 2048, True),
        (torch.bfloat16, 2048, True),
        (torch.float32, 2048, True),
        (torch.float64, 2048, True),
    ],
)
def test_gaussian_binary_spike_distance_and_gradients(
    dtype, features, autocast_enabled, device
):
    x = torch.ones(2, features, dtype=dtype, device=device, requires_grad=True)
    y = torch.ones(2, features, dtype=dtype, device=device)
    y[1, -1] = 0
    y.requires_grad_()
    reference_x = x.detach().double().requires_grad_()
    reference_y = y.detach().double().requires_grad_()
    reference = torch.exp(
        -((reference_x[:, None] - reference_y[None, :]) ** 2).sum(-1) / 2
    )
    expected_gradients = torch.autograd.grad(
        reference.sum(), (reference_x, reference_y)
    )
    with torch.autocast(
        device,
        dtype=torch.bfloat16 if device == "cpu" else torch.float16,
        enabled=autocast_enabled,
    ):
        actual = kernel_dot_product(x, y, "gaussian", 1.0)
    expected_dtype = (
        torch.float32 if autocast_enabled and dtype != torch.float64 else dtype
    )
    assert actual.dtype == expected_dtype
    torch.testing.assert_close(actual, reference.detach().to(actual.dtype))
    for gradient, expected in zip(
        torch.autograd.grad(actual.sum(), (x, y)), expected_gradients
    ):
        torch.testing.assert_close(gradient, expected.to(dtype))


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_gaussian_precision_remains_fullgraph_traceable(autocast_enabled):
    dtype = torch.float32 if autocast_enabled else torch.float16
    x = torch.ones(2, 2048, dtype=dtype)
    y = x.clone()
    y[1, -1] = 0
    compiled = torch.compile(kernel_dot_product, backend="eager", fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        expected = kernel_dot_product(x, y, "gaussian", 1.0)
        actual = compiled(x, y, "gaussian", 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gaussian_precision_keeps_meta_shape_propagation(dtype):
    x = torch.empty(2, 2048, device="meta", dtype=torch.float16)
    result = kernel_dot_product(x, x.to(dtype), "gaussian", 1.0)
    assert result.shape == (2, 2)
    assert result.dtype == dtype
    assert result.device.type == "meta"


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_gaussian_keeps_incompatible_matmul_dtype_errors(autocast_enabled):
    left_dtype = torch.float64 if autocast_enabled else torch.float16
    x = torch.ones(2, 3, dtype=left_dtype)
    y = torch.ones(2, 3, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        with pytest.raises(RuntimeError):
            kernel_dot_product(x, y, "gaussian", 1.0)


def test_gaussian_integer_inputs_keep_floating_output():
    x = torch.eye(2, dtype=torch.int64)
    actual = kernel_dot_product(x, x, "gaussian", 1.0)
    expected = torch.exp(-((x.float()[:, None] - x.float()[None, :]) ** 2).sum(-1) / 2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_gaussian_spike_similarity_loss_learns_under_cpu_autocast():
    spikes = torch.ones(2, 2048)
    spikes[1, -1] = 0
    spikes.requires_grad_()
    labels = torch.eye(2)
    reference_spikes = spikes.detach().double().requires_grad_()
    reference_kernel = torch.exp(
        -((reference_spikes[:, None] - reference_spikes[None, :]) ** 2).sum(-1) / 2
    )
    expected = (reference_kernel - labels.double()).square().mean()
    expected_gradient = torch.autograd.grad(expected, reference_spikes)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = spike_similar_loss(spikes, labels, "gaussian", "mse", 1.0)
    torch.testing.assert_close(actual, expected.detach().float())
    torch.testing.assert_close(
        torch.autograd.grad(actual, spikes)[0], expected_gradient.float()
    )
