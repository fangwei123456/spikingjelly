import pytest
import torch
from spikingjelly.activation_based.functional import (
    kernel_dot_product,
    spike_similar_loss,
)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("features", [2048, 32768])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_gaussian_binary_spike_distance_and_gradients(
    dtype, features, autocast_enabled
):
    x = torch.ones(2, features, dtype=dtype, requires_grad=True)
    y = torch.ones(2, features, dtype=dtype)
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
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        actual = kernel_dot_product(x, y, "gaussian", 1.0)
    torch.testing.assert_close(actual, reference.detach().to(actual.dtype))
    for gradient, expected in zip(
        torch.autograd.grad(actual.sum(), (x, y)), expected_gradients
    ):
        torch.testing.assert_close(gradient, expected.to(dtype))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_gaussian_precision_remains_fullgraph_traceable(dtype, autocast_enabled):
    x = torch.ones(2, 2048, dtype=dtype)
    y = x.clone()
    y[1, -1] = 0
    compiled = torch.compile(kernel_dot_product, backend="eager", fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        actual = compiled(x, y, "gaussian", 1.0)
    expected = torch.exp(
        -((x.double()[:, None] - y.double()[None, :]) ** 2).sum(-1) / 2
    )
    torch.testing.assert_close(actual, expected.to(actual.dtype))


def test_gaussian_precision_keeps_meta_shape_propagation():
    x = torch.empty(2, 2048, device="meta", dtype=torch.float16)
    result = kernel_dot_product(x, x, "gaussian", 1.0)
    assert result.shape == (2, 2)
    assert result.dtype == x.dtype
    assert result.device.type == "meta"


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_gaussian_keeps_incompatible_matmul_dtype_errors(autocast_enabled):
    """Widening must not accept mixed dtypes rejected by native matrix products."""
    left_dtype = torch.float64 if autocast_enabled else torch.float16
    x = torch.ones(2, 3, dtype=left_dtype)
    y = torch.ones(2, 3, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        with pytest.raises(RuntimeError):
            kernel_dot_product(x, y, "gaussian", 1.0)


def test_gaussian_integer_inputs_keep_floating_output():
    """The native integer-input Gaussian path must not narrow its result to int."""
    x = torch.eye(2, dtype=torch.int64)
    actual = kernel_dot_product(x, x, "gaussian", 1.0)
    expected = torch.exp(-((x.float()[:, None] - x.float()[None, :]) ** 2).sum(-1) / 2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_gaussian_spike_similarity_loss_learns_under_cpu_autocast():
    """A single differing spike must retain its loss and gradient during AMP."""
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
