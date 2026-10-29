import copy
import math

import pytest
import torch

from spikingjelly.activation_based.lava_exchange import CubaLIFNode, LeakyIntegratorStep


@pytest.mark.parametrize("x_grad", [False, True])
@pytest.mark.parametrize("state_grad", [False, True])
@pytest.mark.parametrize("decay_grad", [False, True])
def test_leaky_integrator_saves_state_for_each_differentiable_input(
    x_grad, state_grad, decay_grad
):
    x = torch.tensor([0.25, -0.5], requires_grad=x_grad)
    state = torch.tensor([2.0, -1.0], requires_grad=state_grad)
    decay = torch.tensor(1024.0, requires_grad=decay_grad)
    result = LeakyIntegratorStep.apply(x, decay, state, 4096)

    torch.testing.assert_close(result, torch.tensor([1.75, -1.25]), rtol=0, atol=0)
    assert result.requires_grad == (x_grad or state_grad or decay_grad)
    if not result.requires_grad:
        return
    result.backward(torch.tensor([3.0, -2.0]))
    if x_grad:
        torch.testing.assert_close(x.grad, torch.tensor([3.0, -2.0]), rtol=0, atol=0)
    if state_grad:
        torch.testing.assert_close(
            state.grad, torch.tensor([2.25, -1.5]), rtol=0, atol=0
        )
    if decay_grad:
        # The established surrogate derivative sums -state*upstream/4096.
        torch.testing.assert_close(
            decay.grad, torch.tensor(-8.0 / 4096), rtol=0, atol=0
        )


@pytest.mark.parametrize("x_grad", [False, True])
def test_broadcast_decay_gradient_is_reduced_per_parameter(x_grad):
    x = torch.tensor([[0.25, -0.5], [0.5, 0.25]], requires_grad=x_grad)
    state = torch.tensor([[2.0, -1.0], [1.0, -2.0]])
    decay = torch.tensor([1024.0, 2048.0], requires_grad=True)
    result = LeakyIntegratorStep.apply(x, decay, state, 4096)

    torch.testing.assert_close(
        result, torch.tensor([[1.75, -1.0], [1.25, -0.75]]), rtol=0, atol=0
    )
    result.backward(torch.tensor([[3.0, -2.0], [1.0, 4.0]]))
    torch.testing.assert_close(
        decay.grad, torch.tensor([-7.0, 6.0]) / 4096, rtol=0, atol=0
    )


@pytest.mark.parametrize("step_mode", ["s", "m"])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("tensor_decay", [False, True])
def test_cuba_decay_gradients_do_not_require_input_gradients(
    step_mode, detach_reset, tensor_decay
):
    current_decay = torch.tensor([0.125, 0.25, 0.375]) if tensor_decay else 0.25
    voltage_decay = torch.tensor([0.25, 0.5, 0.75]) if tensor_decay else 0.5
    node = CubaLIFNode(
        current_decay=current_decay,
        voltage_decay=voltage_decay,
        requires_grad=True,
        step_mode=step_mode,
        detach_reset=detach_reset,
    )
    node.current_state = torch.full((2, 3), 0.75)
    node.voltage_state = torch.full((2, 3), 0.25)
    reference = copy.deepcopy(node)
    shape = (3, 2, 3) if step_mode == "m" else (2, 3)
    x = torch.linspace(0.3, 0.7, math.prod(shape)).reshape(shape)
    differentiable_x = x.clone().requires_grad_()

    result = node(x)
    expected = reference(differentiable_x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        node.current_state, reference.current_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        node.voltage_state, reference.voltage_state, rtol=0, atol=0
    )

    for module, spikes in ((node, result), (reference, expected)):
        (
            spikes.sum()
            + module.current_state.square().mean()
            + module.voltage_state.square().mean()
        ).backward()

    assert x.grad is None
    for name in ("current_decay", "voltage_decay"):
        gradient = getattr(node, name).grad
        assert gradient is not None
        assert gradient.shape == ((3,) if tensor_decay else ())
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(
            gradient, getattr(reference, name).grad, rtol=0, atol=0
        )
