import pytest
import torch

from spikingjelly.timing_based import functional, neuron


def _dense_local_weights(weight, input_size, stride, padding):
    channels, height, width = weight.shape[1], *input_size
    kernel = weight.shape[-1]
    output_height = (height + 2 * padding - kernel) // stride + 1
    output_width = (width + 2 * padding - kernel) // stride + 1
    rows = []
    for output in range(weight.shape[0]):
        for y in range(output_height):
            for x in range(output_width):
                row = []
                for channel in range(channels):
                    for iy in range(height):
                        for ix in range(width):
                            ky = iy - (y * stride - padding)
                            kx = ix - (x * stride - padding)
                            row.append(
                                weight[output, channel, ky, kx]
                                if 0 <= ky < kernel and 0 <= kx < kernel
                                else weight.new_zeros(())
                            )
                rows.append(torch.stack(row))
    return torch.stack(rows)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (1, 1), (3, 1)])
def test_conv_shared_gradients_match_dense_connectivity(device, stride, padding):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    times = torch.tensor(
        [0.3, 0.1, 0.2, 0.6, 0.9, 0.4, -0.1, 5.0],
        device=device,
        requires_grad=True,
    )
    indices = torch.tensor([0, 2, 2, 5, 17, 20, 3, 22], device=device)
    offsets = torch.tensor([0, 4, 4, 8], device=device)
    weight = torch.linspace(0.7, 1.3, 16, device=device).reshape(2, 2, 2, 2)
    weight.requires_grad_()
    actual = functional.cubic_conv2d(
        times,
        indices,
        offsets,
        weight,
        (3, 4),
        3.0,
        support=1.0,
        stride=stride,
        padding=padding,
    )
    expected = functional.cubic_srm(
        times,
        indices,
        offsets,
        _dense_local_weights(weight, (3, 4), stride, padding),
        3.0,
        support=1.0,
    )
    assert actual[0].numel() > 0
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    actual_grad = torch.autograd.grad(
        actual[0].sum(), (times, weight), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected[0].sum(), (times, weight))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    assert actual_grad[0][-2:].count_nonzero() == 0


@pytest.mark.filterwarnings("ignore:Input #.*not a double precision:UserWarning")
def test_conv_overlapping_time_and_shared_weight_gradcheck():
    times = torch.linspace(0.1, 0.7, 9, requires_grad=True)
    weight = torch.tensor([[[[0.8, 0.9], [1.0, 1.1]]]], requires_grad=True)
    indices, offsets = torch.arange(9), torch.tensor([0, 9])
    assert torch.autograd.gradcheck(
        lambda t, w: functional.cubic_conv2d(t, indices, offsets, w, (3, 3), 3.0)[0],
        (times, weight),
        eps=1e-3,
        atol=2e-3,
        rtol=1e-2,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conv_empty_trials_and_module_chain(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    conv = neuron.CubicConv2d(1, 2, 2, 3.0, stride=2, padding=1).to(device)
    with torch.no_grad():
        conv.weight.fill_(1.2)
    times = torch.tensor([0.2, 0.4], device=device, requires_grad=True)
    indices, offsets = (
        torch.tensor([0, 3], device=device),
        torch.tensor([0, 0, 2], device=device),
    )
    events = conv(times, indices, offsets, (2, 2))
    assert events[2][1] == 0 and events[0].numel() > 0
    head = neuron.CubicSRM(8, 1, 3.0).to(device)
    with torch.no_grad():
        head.weight.fill_(1.2)
    output = head(*events)
    output[0].sum().backward()
    assert torch.isfinite(times.grad).all() and torch.isfinite(conv.weight.grad).all()
    for offsets in (
        torch.tensor([0], device=device),
        torch.tensor([0, 0], device=device),
    ):
        empty = conv(times[:0], indices[:0], offsets, (2, 2))
        assert empty[0].numel() == 0 and torch.equal(empty[2], offsets)


@pytest.mark.parametrize("bad", ["shape", "stride", "padding", "index"])
def test_conv_rejects_invalid_geometry_and_indices(bad):
    weight = torch.ones(1, 1, 2, 2)
    kwargs = dict(input_size=(3, 3), stride=1, padding=0)
    index = torch.tensor([0])
    if bad == "shape":
        kwargs["input_size"] = (0, 3)
    elif bad == "stride":
        kwargs["stride"] = 0
    elif bad == "padding":
        kwargs["padding"] = -1
    else:
        index[0] = 9
    with pytest.raises(ValueError):
        functional.cubic_conv2d(
            torch.tensor([0.2]),
            index,
            torch.tensor([0, 1]),
            weight,
            duration=3.0,
            **kwargs,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conv_mnist_loss_backpropagates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from spikingjelly.timing_based.examples.cubic_conv_mnist import _CubicConvNet

    torch.manual_seed(0)
    model = _CubicConvNet(2).to(device)
    logits = model(torch.rand(2, 1, 28, 28, device=device))
    torch.nn.functional.cross_entropy(
        logits, torch.tensor([1, 2], device=device)
    ).backward()
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
