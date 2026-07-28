import inspect
import math

import pytest
import torch

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.functional import layer as functional_layer


def test_delay_step_matches_sequence_delay_and_gradients():
    x_seq = torch.randn(5, 2, 3, requires_grad=True)
    queue = ()
    outputs = []
    for x in x_seq:
        y, queue = functional.delay_step(x, queue, delay_steps=2)
        outputs.append(y)
    y_seq = torch.stack(outputs)
    y_seq.sum().backward()

    x_ref = x_seq.detach().clone().requires_grad_()
    functional.delay(x_ref, 2).sum().backward()

    assert torch.allclose(y_seq, functional.delay(x_seq.detach(), 2))
    assert torch.allclose(x_seq.grad, x_ref.grad)
    assert torch.equal(queue[0], x_seq[-2])
    assert torch.equal(queue[1], x_seq[-1])


def test_delay_step_does_not_mutate_queue():
    x = torch.randn(2, 3)
    queue = (torch.randn(2, 3), torch.randn(2, 3))
    snapshot = tuple(item.clone() for item in queue)

    y, queue_next = functional.delay_step(x, queue, delay_steps=2)

    assert y is queue[0]
    assert queue_next[0] is queue[1]
    assert queue_next[1] is x
    for actual, expected in zip(queue, snapshot):
        assert torch.equal(actual, expected)


def test_delay_step_rejects_negative_delay():
    with pytest.raises(ValueError, match="non-negative integer"):
        functional.delay_step(torch.randn(2, 3), (), -1)


def test_delay_module_keeps_public_queue_list():
    x0 = torch.randn(2, 3)
    x1 = torch.randn(2, 3)
    module = layer.Delay(delay_steps=1)
    queue = module.queue

    y0 = module(x0)
    y1 = module(x1)

    assert module.queue is queue
    assert torch.equal(y0, torch.zeros_like(x0))
    assert y1 is x0
    assert module.queue == [x1]


@pytest.mark.parametrize("learnable", [False, True])
def test_synapse_filter_step_matches_module_and_gradients(learnable):
    x = torch.rand(2, 3)
    state = torch.rand(2, 3)
    tau = 3.5

    module = layer.SynapseFilter(tau=tau, learnable=learnable)
    x_module = x.clone().requires_grad_()
    state_module = state.clone().requires_grad_()
    module.out_i = state_module
    module(x_module).sum().backward()

    x_functional = x.clone().requires_grad_()
    state_functional = state.clone().requires_grad_()
    if learnable:
        w = torch.tensor(-math.log(tau - 1.0), requires_grad=True)
        reciprocal_tau = w.sigmoid()
    else:
        w = None
        reciprocal_tau = 1.0 / tau
    output = functional.synapse_filter_step(
        x_functional, state_functional, reciprocal_tau
    )
    output.sum().backward()

    assert torch.allclose(output, module.out_i)
    assert torch.allclose(x_functional.grad, x_module.grad)
    assert torch.allclose(state_functional.grad, state_module.grad)
    if learnable:
        assert torch.allclose(w.grad, module.w.grad)


def test_synapse_filter_step_does_not_mutate_state():
    x = torch.rand(2, 3)
    state = torch.rand(2, 3)
    x_before = x.clone()
    state_before = state.clone()

    functional.synapse_filter_step(x, state, 0.25)

    assert torch.equal(x, x_before)
    assert torch.equal(state, state_before)


def test_functional_layer_exports():
    for name in functional_layer.__all__:
        assert getattr(functional, name) is getattr(functional_layer, name)


def test_functional_layer_public_api_documentation():
    for name in functional_layer.__all__:
        function = getattr(functional_layer, name)
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
        doc = inspect.getdoc(function)
        assert f".. _{name}-cn:" in doc
        assert f".. _{name}-en:" in doc
