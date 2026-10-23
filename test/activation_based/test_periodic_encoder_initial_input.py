import pytest
import torch

from spikingjelly.activation_based import base, functional
from spikingjelly.activation_based.encoding import PeriodicEncoder


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3,), (3, 2)])
def test_constructor_sequence_is_used_for_single_step_calls(dtype, shape):
    spike = torch.tensor([[0, 1], [1, 0], [1, 1]], dtype=dtype)
    if len(shape) == 1:
        spike = spike[:, 0]
    encoder = PeriodicEncoder(spike)

    actual = torch.stack([encoder() for _ in range(7)])

    torch.testing.assert_close(actual, spike[[0, 1, 2, 0, 1, 2, 0]])
    assert actual.dtype == dtype


def test_explicit_first_input_keeps_precedence_over_constructor_sequence():
    initial = torch.zeros((3, 2))
    replacement = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    encoder = PeriodicEncoder(initial)

    actual = torch.stack([encoder(replacement), encoder(), encoder()])

    torch.testing.assert_close(actual, replacement[[0, 1, 0]])
    encoder.reset()
    torch.testing.assert_close(encoder(replacement), replacement[0])


def test_reset_restores_constructor_sequence_as_the_default():
    initial = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    encoder = PeriodicEncoder(initial)
    encoder(torch.zeros((2, 2)))
    encoder()

    encoder.reset()
    actual = torch.stack([encoder(), encoder(), encoder(), encoder()])

    torch.testing.assert_close(actual, initial[[0, 1, 2, 0]])


def test_default_sequence_follows_module_dtype_conversion_after_reset():
    initial = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    encoder = PeriodicEncoder(initial).double()
    first = encoder()
    encoder.reset()

    torch.testing.assert_close(first, initial[0].double())
    torch.testing.assert_close(encoder(), initial[0].double())


def test_constructor_sequence_preserves_autograd_through_multiple_cycles():
    spike = torch.rand((3, 2), requires_grad=True)
    encoder = PeriodicEncoder(spike)
    coefficients = torch.tensor([2.0, 3.0])
    loss = sum((encoder() * coefficients).sum() for _ in range(7))

    loss.backward()

    expected = torch.tensor([3.0, 2.0, 2.0])[:, None] * coefficients
    torch.testing.assert_close(spike.grad, expected)


def test_constructor_sequence_does_not_add_state_dict_keys():
    encoder = PeriodicEncoder(torch.ones((3, 2)))
    assert not encoder.state_dict()


@pytest.mark.parametrize("kind", ["leaf", "nonleaf", "view"])
@pytest.mark.parametrize("detach_net", [False, True])
def test_detaching_state_preserves_constructor_autograd(kind, detach_net):
    leaf = torch.arange(6.0, requires_grad=True)
    if kind == "leaf":
        initial = leaf
    elif kind == "nonleaf":
        initial = leaf * 2
    else:
        initial = leaf.view(3, 2)
    encoder = PeriodicEncoder(initial)
    encoder()

    if detach_net:
        functional.detach_net(encoder)
    else:
        encoder.detach()

    assert initial.requires_grad
    assert not encoder.spike.requires_grad
    encoder.reset()
    sum(encoder().sum() for _ in range(initial.shape[0])).backward()
    torch.testing.assert_close(
        leaf.grad, torch.full_like(leaf, 2.0 if kind == "nonleaf" else 1.0)
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_functional_forward_matches_default_and_explicit_cycles(wrapped, explicit):
    initial = torch.arange(6.0).view(3, 2)
    replacement = torch.arange(4.0).view(2, 2) + 10
    encoder = PeriodicEncoder(initial)
    forward = (
        base.to_functional_forward(encoder) if wrapped else encoder.functional_forward
    )
    sequence = replacement if explicit else initial
    states = (None, 0)
    actual = []

    for t in range(7):
        inputs = (replacement,) if explicit and t == 0 else ()
        outputs, states = forward(inputs, states)
        actual.append(outputs[0])

    torch.testing.assert_close(
        torch.stack(actual), sequence[torch.arange(7) % len(sequence)]
    )
    assert encoder.spike is None
    assert encoder.t == 0
