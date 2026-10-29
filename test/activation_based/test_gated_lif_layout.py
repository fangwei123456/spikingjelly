import copy

import pytest
import torch
from torch import nn

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import GatedLIFNode


@pytest.mark.parametrize("spatial_shape", [(), (5,), (2, 3)])
@pytest.mark.parametrize("channel_wise", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gated_lif_layout_matches_image_sequence_values_and_gradients(
    spatial_shape, channel_wise, dtype
):
    node = GatedLIFNode(T=3, inplane=4 if channel_wise else None).to(dtype)
    image_node = copy.deepcopy(node)
    x = torch.randn(
        3, 2, 4, *spatial_shape, dtype=dtype, generator=torch.Generator().manual_seed(0)
    ).requires_grad_()
    # The established 2D-image path is the reference for the same independent neurons.
    image_x = x.detach().reshape(3, 2, 4, -1, 1).clone().requires_grad_()
    original = x.detach().clone()

    result = node(x)
    reference = image_node(image_x)

    assert result.shape == x.shape
    assert node.v.shape == node.u.shape == x.shape[1:]
    torch.testing.assert_close(result, reference.reshape_as(x))
    torch.testing.assert_close(node.v, image_node.v.reshape_as(node.v))
    torch.testing.assert_close(node.u, image_node.u.reshape_as(node.u))
    torch.testing.assert_close(x, original)

    (result.square().mean() + node.v.square().mean()).backward()
    (reference.square().mean() + image_node.v.square().mean()).backward()
    torch.testing.assert_close(x.grad, image_x.grad.reshape_as(x))
    for parameter, image_parameter in zip(node.parameters(), image_node.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, image_parameter.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_lif_between_linear_layers_trains_without_extra_axes(dtype):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(6, 4), GatedLIFNode(T=3, inplane=4), nn.Linear(4, 2)
    ).to(dtype)
    with torch.no_grad():
        model[0].weight.fill_(0.4)
        model[0].bias.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.rand(3, 2, 6, dtype=dtype)
    target = torch.full((3, 2, 2), 0.25, dtype=dtype)
    initial_head = model[-1].weight.detach().clone()

    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        assert output.shape == target.shape
        assert output.dtype == dtype
        loss = (output - target).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        functional.reset_net(model)

    assert not torch.equal(model[-1].weight, initial_head)
