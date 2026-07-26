import inspect
import math

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.functional import (
    forward,
    layer as functional_layer,
    loss,
    misc,
    net_config,
    neuron as functional_neuron,
    online_learning,
)


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_delay_single_step_matches_sequence_delay_and_gradients():
    torch.manual_seed(10)
    x_seq = torch.randn(5, 2, 3, requires_grad=True)
    delay_steps = 2
    queue = ()
    outputs = []
    for t in range(x_seq.shape[0]):
        y, queue = functional.delay_single_step(x_seq[t], queue, delay_steps)
        outputs.append(y)
    y_seq = torch.stack(outputs)
    loss = y_seq.sum()
    loss.backward()

    x_ref = x_seq.detach().clone().requires_grad_()
    y_ref = functional.delay(x_ref, delay_steps)
    y_ref.sum().backward()

    _assert_close(y_seq, y_ref)
    _assert_close(x_seq.grad, x_ref.grad)
    assert len(queue) == delay_steps
    _assert_close(queue[0], x_seq[-2])
    _assert_close(queue[1], x_seq[-1])


def test_delay_single_step_does_not_mutate_queue_or_tensors():
    x = torch.randn(2, 3)
    old_0 = torch.randn(2, 3)
    old_1 = torch.randn(2, 3)
    queue = (old_0, old_1)
    old_0_before = old_0.clone()
    old_1_before = old_1.clone()

    y, queue_next = functional.delay_single_step(x, queue, delay_steps=2)

    assert y is old_0
    assert queue == (old_0, old_1)
    assert queue_next == (old_1, x)
    assert torch.equal(old_0, old_0_before)
    assert torch.equal(old_1, old_1_before)


def test_delay_single_step_zero_delay_returns_input_and_empty_queue():
    x = torch.randn(2, 3)

    y, queue_next = functional.delay_single_step(x, (), delay_steps=0)

    assert y is x
    assert queue_next == ()


def test_delay_single_step_rejects_invalid_delay_steps():
    with pytest.raises(ValueError, match="non-negative integer"):
        functional.delay_single_step(torch.randn(2, 3), (), -1)


def test_delay_module_uses_functional_state_but_keeps_public_queue_list():
    x0 = torch.randn(2, 3)
    x1 = torch.randn(2, 3)
    module = layer.Delay(delay_steps=1)
    queue = module.queue

    y0 = module(x0)
    y1 = module(x1)

    assert module.queue is queue
    assert isinstance(module.queue, list)
    _assert_close(y0, torch.zeros_like(x0))
    assert y1 is x0
    assert module.queue == [x1]


def test_delay_multi_step_does_not_consume_single_step_queue():
    module = layer.Delay(delay_steps=1, step_mode="s")
    queued = torch.randn(2, 3)
    module.queue = [queued]
    functional.set_step_mode(module, "m")
    x_seq = torch.randn(4, 2, 3)

    y_seq = module(x_seq)

    _assert_close(y_seq, functional.delay(x_seq, 1))
    assert module.queue == [queued]


def test_element_wise_recurrent_single_step_matches_module_and_gradients():
    torch.manual_seed(11)
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(2, 3, requires_grad=True)

    module_sub = nn.Linear(3, 3)
    func_sub = nn.Linear(3, 3)
    func_sub.load_state_dict(module_sub.state_dict())

    def element_wise_function(y_state, x_input):
        return y_state * 0.25 + x_input

    module = layer.ElementWiseRecurrentContainer(
        module_sub, element_wise_function=element_wise_function
    )
    x_module = x.detach().clone().requires_grad_()
    y_module = y.detach().clone().requires_grad_()
    module.y = y_module
    out_module = module(x_module)
    out_module.sum().backward()

    x_func = x.detach().clone().requires_grad_()
    y_func = y.detach().clone().requires_grad_()
    out_func = functional.element_wise_recurrent_single_step(
        x_func, y_func, element_wise_function, func_sub
    )
    out_func.sum().backward()

    _assert_close(out_func, out_module)
    _assert_close(module.y, out_module)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(y_func.grad, y_module.grad)
    _assert_close(func_sub.weight.grad, module_sub.weight.grad)
    _assert_close(func_sub.bias.grad, module_sub.bias.grad)


def test_element_wise_recurrent_module_materializes_none_state_before_functional_update():
    x = torch.randn(2, 3)
    sub_module = nn.Identity()
    module = layer.ElementWiseRecurrentContainer(
        sub_module, element_wise_function=lambda y, x: y + x
    )

    out = module(x)

    _assert_close(out, x)
    _assert_close(module.y, x)


def test_element_wise_recurrent_single_step_does_not_mutate_input_or_state():
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    x_before = x.clone()
    y_before = y.clone()

    functional.element_wise_recurrent_single_step(
        x, y, lambda y_state, x_input: y_state + x_input, nn.Identity()
    )

    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)


@pytest.mark.parametrize("bias", [False, True])
def test_linear_recurrent_single_step_matches_module_and_gradients(bias):
    torch.manual_seed(12)
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)

    module_sub = nn.Linear(3, 2)
    func_sub = nn.Linear(3, 2)
    func_sub.load_state_dict(module_sub.state_dict())
    module = layer.LinearRecurrentContainer(module_sub, 3, 2, bias=bias)

    x_module = x.detach().clone().requires_grad_()
    y_module = y.detach().clone().requires_grad_()
    module.y = y_module
    out_module = module(x_module)
    out_module.sum().backward()

    x_func = x.detach().clone().requires_grad_()
    y_func = y.detach().clone().requires_grad_()
    recurrent_func = nn.Linear(5, 3, bias=bias)
    recurrent_func.load_state_dict(module.rc.state_dict())
    out_func = functional.linear_recurrent_single_step(
        x_func, y_func, recurrent_func, func_sub
    )
    out_func.sum().backward()

    _assert_close(out_func, out_module)
    _assert_close(module.y, out_module)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(y_func.grad, y_module.grad)
    _assert_close(recurrent_func.weight.grad, module.rc.weight.grad)
    if bias:
        _assert_close(recurrent_func.bias.grad, module.rc.bias.grad)
    _assert_close(func_sub.weight.grad, module_sub.weight.grad)
    _assert_close(func_sub.bias.grad, module_sub.bias.grad)


def test_linear_recurrent_module_materializes_none_state_before_functional_update():
    x = torch.randn(2, 4, 3)
    module = layer.LinearRecurrentContainer(nn.Identity(), 3, 3, bias=False)
    module.rc.weight.data.copy_(torch.eye(3, 6))

    out = module(x)

    _assert_close(out, x)
    assert module.y.shape == x.shape
    _assert_close(module.y, x)


def test_linear_recurrent_single_step_does_not_mutate_input_or_state():
    x = torch.randn(2, 3)
    y = torch.randn(2, 2)
    x_before = x.clone()
    y_before = y.clone()

    functional.linear_recurrent_single_step(
        x,
        y,
        nn.Linear(5, 3),
        nn.Identity(),
    )

    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)


def test_linear_recurrent_module_preserves_recurrent_module_hooks():
    module = layer.LinearRecurrentContainer(nn.Identity(), 3, 3)
    hook_inputs = []
    module.rc.register_forward_hook(
        lambda _module, args, _out: hook_inputs.append(args[0])
    )

    x = torch.randn(2, 3)
    module(x)

    assert len(hook_inputs) == 1
    assert hook_inputs[0].shape == (2, 6)


@pytest.mark.parametrize("shared_across_channels", [False, True])
def test_neunorm_single_step_matches_formula_and_gradients(shared_across_channels):
    torch.manual_seed(13)
    in_spikes = torch.randn(2, 3, 4, 5)
    x = torch.randn(2, 1, 4, 5)
    if shared_across_channels:
        w = torch.randn(1, 4, 5)
    else:
        w = torch.randn(3, 4, 5)
    k0 = 0.8
    k1 = (1.0 - k0) / 3**2

    in_ref = in_spikes.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    x_next_ref = k0 * x_ref + k1 * in_ref.sum(dim=1, keepdim=True)
    out_ref = in_ref - w_ref * x_next_ref
    (out_ref.sum() + x_next_ref.sum()).backward()

    in_func = in_spikes.detach().clone().requires_grad_()
    x_func = x.detach().clone().requires_grad_()
    w_func = w.detach().clone().requires_grad_()
    out_func, x_next_func = functional.neunorm_single_step(
        in_func, x_func, w_func, k0, k1
    )
    (out_func.sum() + x_next_func.sum()).backward()

    _assert_close(out_func, out_ref)
    _assert_close(x_next_func, x_next_ref)
    _assert_close(in_func.grad, in_ref.grad)
    _assert_close(x_func.grad, x_ref.grad)
    _assert_close(w_func.grad, w_ref.grad)


def test_neunorm_module_materializes_scalar_state_before_functional_update():
    torch.manual_seed(14)
    in_spikes = torch.randn(2, 3, 4, 5)
    module = layer.NeuNorm(3, 4, 5, k=0.75, shared_across_channels=False)
    module.x = 0.25
    x_initial = torch.full_like(in_spikes.sum(dim=1, keepdim=True), 0.25)

    out = module(in_spikes)
    expected_out, expected_x = functional.neunorm_single_step(
        in_spikes, x_initial, module.w, module.k0, module.k1
    )

    _assert_close(out, expected_out)
    _assert_close(module.x, expected_x)


def test_neunorm_single_step_does_not_mutate_input_or_state():
    in_spikes = torch.randn(2, 3, 4, 5)
    x = torch.randn(2, 1, 4, 5)
    w = torch.randn(3, 4, 5)
    in_before = in_spikes.clone()
    x_before = x.clone()
    w_before = w.clone()

    functional.neunorm_single_step(in_spikes, x, w, 0.9, (1.0 - 0.9) / 3**2)

    assert torch.equal(in_spikes, in_before)
    assert torch.equal(x, x_before)
    assert torch.equal(w, w_before)


class _ScaleModule(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.track_running_stats = True

    def forward(self, x):
        return x * self.scale


def test_batch_norm_through_time_single_step_selects_next_module_and_time():
    x = torch.randn(2, 3)
    bn_modules = nn.ModuleList([_ScaleModule(2.0), _ScaleModule(3.0)])

    out0, t0 = functional.batch_norm_through_time_single_step(x, -1, bn_modules)
    out1, t1 = functional.batch_norm_through_time_single_step(x, t0, bn_modules)

    _assert_close(out0, x * 2.0)
    assert t0 == 0
    _assert_close(out1, x * 3.0)
    assert t1 == 1


def test_batch_norm_through_time_single_step_preserves_existing_overflow_error():
    with pytest.raises(IndexError):
        functional.batch_norm_through_time_single_step(
            torch.randn(2, 3), 1, nn.ModuleList([nn.Identity(), nn.Identity()])
        )


def test_batch_norm_through_time_single_step_restores_track_running_stats_on_error():
    class RaisingBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.track_running_stats = True

        def forward(self, x):
            raise RuntimeError("boom")

    bn = RaisingBN()

    with pytest.raises(RuntimeError, match="boom"):
        functional.batch_norm_through_time_single_step(
            torch.randn(2, 3),
            -1,
            nn.ModuleList([bn]),
            disable_running_stats=True,
        )

    assert bn.track_running_stats is True


def test_batch_norm_through_time_module_updates_time_state():
    x = torch.randn(2, 3)
    module = layer.BatchNormThroughTime1d(T=2, num_features=3).eval()
    expected = module.bn_list[0](x)

    out = module(x)

    _assert_close(out, expected)
    assert module.t == 0


def test_batch_norm_through_time_module_advances_time_before_bn_error():
    class RaisingBN(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    module = layer.BatchNormThroughTime1d(T=1, num_features=3)
    module.bn_list[0] = RaisingBN()

    with pytest.raises(RuntimeError, match="boom"):
        module(torch.randn(2, 3))

    assert module.t == 0


@pytest.mark.parametrize("learnable", [False, True])
def test_synapse_filter_single_step_matches_module_and_gradients(learnable):
    torch.manual_seed(9)
    x = torch.rand(2, 3, requires_grad=True)
    out_i = torch.rand(2, 3, requires_grad=True)
    tau = 3.5

    module = layer.SynapseFilter(tau=tau, learnable=learnable)
    x_module = x.detach().clone().requires_grad_()
    out_i_module = out_i.detach().clone().requires_grad_()
    module.out_i = out_i_module
    y_module = module(x_module)
    loss_module = y_module.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    out_i_func = out_i.detach().clone().requires_grad_()
    if learnable:
        w_func = torch.tensor(-math.log(tau - 1.0), requires_grad=True)
        reciprocal_tau = w_func.sigmoid()
    else:
        w_func = None
        reciprocal_tau = 1.0 / tau
    y_func = functional.synapse_filter_single_step(x_func, out_i_func, reciprocal_tau)
    loss_func = y_func.sum()
    loss_func.backward()

    _assert_close(y_func, y_module)
    _assert_close(module.out_i, y_module)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(out_i_func.grad, out_i_module.grad)
    if learnable:
        _assert_close(w_func.grad, module.w.grad)


def test_synapse_filter_module_materializes_scalar_state_before_functional_update():
    x = torch.ones(2, 3)
    module = layer.SynapseFilter(tau=2.0)
    module.out_i = 0.25

    y = module(x)
    expected_initial = torch.full_like(x, 0.25)
    expected = functional.synapse_filter_single_step(x, expected_initial, 0.5)

    _assert_close(y, expected)
    _assert_close(module.out_i, expected)


def test_synapse_filter_single_step_does_not_mutate_input_or_state():
    x = torch.rand(2, 3)
    out_i = torch.rand(2, 3)
    x_before = x.clone()
    out_i_before = out_i.clone()

    functional.synapse_filter_single_step(x, out_i, 0.25)

    assert torch.equal(x, x_before)
    assert torch.equal(out_i, out_i_before)


def test_functional_layer_top_level_exports():
    for name in functional_layer.__all__:
        assert getattr(functional, name) is getattr(functional_layer, name)


def test_functional_layer_exports_do_not_collide_with_existing_functional_modules():
    modules = (
        forward,
        functional_layer,
        loss,
        misc,
        net_config,
        functional_neuron,
        online_learning,
    )
    seen = {}
    for module in modules:
        for name in module.__all__:
            previous = seen.setdefault(name, module.__name__)
            assert previous == module.__name__, (
                f"`{name}` exported by both {previous} and {module.__name__}"
            )


def test_functional_layer_public_functions_have_annotations_and_bilingual_docstrings():
    for name in functional_layer.__all__:
        fn = getattr(functional_layer, name)
        signature = inspect.signature(fn)
        assert signature.return_annotation is not inspect.Signature.empty, name
        for parameter in signature.parameters.values():
            assert parameter.annotation is not inspect.Parameter.empty, (
                name,
                parameter.name,
            )

        doc = inspect.getdoc(fn)
        assert doc is not None, name
        assert f".. _{name}-cn:" in doc
        assert f".. _{name}-en:" in doc
        assert ":param" in doc
        assert ":return:" in doc
        assert ":rtype:" in doc
