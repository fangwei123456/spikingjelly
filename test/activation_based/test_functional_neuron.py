import math
import inspect

import pytest
import torch

from spikingjelly.activation_based import functional, lava_exchange, neuron, surrogate
from spikingjelly.activation_based.functional import (
    forward,
    loss,
    misc,
    net_config,
    online_learning,
)
from spikingjelly.activation_based.functional import neuron as functional_neuron
from spikingjelly.activation_based.model import spike_dhs
from spikingjelly.activation_based.neuron import psn as neuron_psn
from spikingjelly.activation_based.neuron import inductor_cache


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def _make_surrogate():
    return surrogate.Sigmoid(alpha=4.0)


def test_gated_lif_materializes_scalar_voltage_without_losing_value():
    torch.manual_seed(0)
    x_seq = torch.randn(3, 2, 2, 2, 2)
    scalar_state = neuron.GatedLIFNode(T=3, inplane=2)
    tensor_state = neuron.GatedLIFNode(T=3, inplane=2)
    tensor_state.load_state_dict(scalar_state.state_dict())
    scalar_state.v = 2.0
    tensor_state.v = torch.full_like(x_seq[0], 2.0)

    spike_scalar = scalar_state(x_seq)
    spike_tensor = tensor_state(x_seq)

    _assert_close(spike_scalar, spike_tensor)
    _assert_close(scalar_state.v, tensor_state.v)


@pytest.fixture
def identity_inductor_compile(monkeypatch):
    inductor_cache.clear()
    compile_calls = []

    def compile_identity(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", compile_identity)
    yield compile_calls
    inductor_cache.clear()


@pytest.mark.parametrize("v_reset", [0.0, None])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_if_single_step_matches_module_torch_training(v_reset, detach_reset):
    torch.manual_seed(0)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    module = neuron.IFNode(
        v_threshold=0.7,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
        backend="torch",
    ).train()
    x_module = x.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    module.v = v_module
    spike_module = module(x_module)
    loss_module = spike_module.sum() + module.v.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    spike_func, v_next = functional.if_single_step(
        x_func,
        v_func,
        v_threshold=0.7,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
    )
    loss_func = spike_func.sum() + v_next.sum()
    loss_func.backward()

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)


@pytest.mark.parametrize("v_reset", [0.0, None])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_lif_single_step_matches_module_torch_training(
    v_reset, decay_input, detach_reset
):
    torch.manual_seed(1)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    module = neuron.LIFNode(
        tau=2.5,
        decay_input=decay_input,
        v_threshold=0.6,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
        backend="torch",
    ).train()
    x_module = x.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    module.v = v_module
    spike_module = module(x_module)
    loss_module = spike_module.sum() + module.v.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    spike_func, v_next = functional.lif_single_step(
        x_func,
        v_func,
        tau=2.5,
        decay_input=decay_input,
        v_threshold=0.6,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
    )
    loss_func = spike_func.sum() + v_next.sum()
    loss_func.backward()

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)


@pytest.mark.parametrize("v_reset", [0.0, None])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_plif_single_step_matches_module_torch_training(
    v_reset, decay_input, detach_reset
):
    torch.manual_seed(2)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)
    w_value = torch.tensor(-math.log(2.5 - 1.0))

    module = neuron.ParametricLIFNode(
        init_tau=2.0,
        decay_input=decay_input,
        v_threshold=0.6,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
        backend="torch",
    ).train()
    module.w.data.copy_(w_value)
    x_module = x.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    module.v = v_module
    spike_module = module(x_module)
    loss_module = spike_module.sum() + module.v.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    w_func = w_value.detach().clone().requires_grad_()
    spike_func, v_next = functional.plif_single_step(
        x_func,
        v_func,
        w_func,
        decay_input=decay_input,
        v_threshold=0.6,
        v_reset=v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=detach_reset,
    )
    loss_func = spike_func.sum() + v_next.sum()
    loss_func.backward()

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)
    _assert_close(w_func.grad, module.w.grad)


def test_qif_and_eif_charge_match_module_and_gradients():
    torch.manual_seed(3)
    cases = (
        (
            neuron.QIFNode(
                tau=2.5,
                v_c=0.3,
                a0=0.8,
                v_rest=0.0,
                backend="torch",
            ),
            lambda x, v: functional.qif_charge(
                x,
                v,
                tau=2.5,
                a0=0.8,
                v_rest=0.0,
                v_c=0.3,
            ),
        ),
        (
            neuron.EIFNode(
                tau=2.5,
                delta_T=1.2,
                theta_rh=0.4,
                v_rest=0.0,
                backend="torch",
            ),
            lambda x, v: functional.eif_charge(
                x,
                v,
                tau=2.5,
                v_rest=0.0,
                delta_t=1.2,
                theta_rh=0.4,
            ),
        ),
    )

    for module, charge in cases:
        x = torch.randn(2, 3, requires_grad=True)
        v = torch.randn(2, 3, requires_grad=True)

        x_module = x.detach().clone().requires_grad_()
        v_module = v.detach().clone().requires_grad_()
        module.v = v_module
        module.neuronal_charge(x_module)
        loss_module = module.v.sum()
        loss_module.backward()

        x_func = x.detach().clone().requires_grad_()
        v_func = v.detach().clone().requires_grad_()
        v_next = charge(x_func, v_func)
        loss_func = v_next.sum()
        loss_func.backward()

        _assert_close(v_next, module.v)
        _assert_close(x_func.grad, x_module.grad)
        _assert_close(v_func.grad, v_module.grad)


def test_qif_and_eif_multi_step_cupy_normalize_kernel_output(monkeypatch):
    from spikingjelly.activation_based import cuda_kernel

    x_seq = torch.randn(3, 2, 4)
    v = torch.randn_like(x_seq[0])
    spike_flat = torch.randn(3, 8)
    v_flat = torch.randn(3, 8)
    calls = []

    def fake_kernel(*args):
        calls.append(args)
        return spike_flat, v_flat

    monkeypatch.setattr(cuda_kernel, "multistep_qif_ptt", fake_kernel)
    spike, v_next, v_seq = functional.qif_multi_step_cupy(
        x_seq,
        v,
        tau=2.0,
        v_threshold=1.0,
        v_reset=0.0,
        v_rest=0.0,
        v_c=0.8,
        a0=1.0,
        detach_reset=False,
        surrogate_function=_make_surrogate(),
        store_v_seq=True,
    )
    assert calls[-1][0].shape == (3, 8)
    assert calls[-1][1].shape == (8,)
    _assert_close(spike, spike_flat.reshape_as(x_seq))
    _assert_close(v_next, v_flat[-1].reshape_as(v))
    _assert_close(v_seq, v_flat.reshape_as(x_seq))

    monkeypatch.setattr(cuda_kernel, "multistep_eif_ptt", fake_kernel)
    spike, v_next, v_seq = functional.eif_multi_step_cupy(
        x_seq,
        v,
        tau=2.0,
        v_threshold=1.0,
        v_reset=None,
        v_rest=0.0,
        theta_rh=0.8,
        delta_t=1.0,
        detach_reset=True,
        surrogate_function=_make_surrogate(),
        store_v_seq=False,
    )
    assert calls[-1][0].shape == (3, 8)
    assert calls[-1][1].shape == (8,)
    _assert_close(spike, spike_flat.reshape_as(x_seq))
    _assert_close(v_next, v_flat[-1].reshape_as(v))
    assert v_seq is None


def test_if_and_lif_single_step_cupy_normalize_kernel_io(monkeypatch):
    from spikingjelly.activation_based.cuda_kernel.auto_cuda import ss_neuron_kernel

    x = torch.randn(2, 3, 4)
    v = torch.randn_like(x)
    spike_flat = torch.randn(x.numel())
    v_flat = torch.randn(x.numel())
    calls = []
    forward_kernel = object()
    backward_kernel = object()

    def fake_kernel(*args):
        calls.append(args)
        return spike_flat, v_flat

    monkeypatch.setattr(ss_neuron_kernel, "ss_if_step", fake_kernel)
    spike, v_next = functional.if_single_step_cupy(
        x,
        v,
        v_threshold=0.7,
        v_reset=None,
        forward_kernel=forward_kernel,
        backward_kernel=backward_kernel,
    )
    assert calls[-1][0].shape == (x.numel(),)
    assert calls[-1][1].shape == (v.numel(),)
    assert calls[-1][-2:] == (forward_kernel, backward_kernel)
    _assert_close(spike, spike_flat.reshape_as(x))
    _assert_close(v_next, v_flat.reshape_as(v))

    monkeypatch.setattr(ss_neuron_kernel, "ss_lif_step", fake_kernel)
    spike, v_next = functional.lif_single_step_cupy(
        x,
        v,
        tau=2.0,
        v_threshold=0.7,
        v_reset=0.0,
        forward_kernel=forward_kernel,
        backward_kernel=backward_kernel,
    )
    assert calls[-1][0].shape == (x.numel(),)
    assert calls[-1][1].shape == (v.numel(),)
    assert calls[-1][4] == 0.5
    assert calls[-1][-2:] == (forward_kernel, backward_kernel)
    _assert_close(spike, spike_flat.reshape_as(x))
    _assert_close(v_next, v_flat.reshape_as(v))


@pytest.mark.parametrize(
    ("kernel_name", "functional_call"),
    [
        (
            "multistep_if",
            lambda x_seq, v: functional.if_multi_step_cupy(
                x_seq,
                v,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
            ),
        ),
        (
            "multistep_lif",
            lambda x_seq, v: functional.lif_multi_step_cupy(
                x_seq,
                v,
                tau=2.0,
                decay_input=True,
                v_threshold=0.7,
                v_reset=0.0,
                surrogate_function=_make_surrogate(),
            ),
        ),
        (
            "multistep_plif",
            lambda x_seq, v: functional.plif_multi_step_cupy(
                x_seq,
                v,
                torch.tensor(0.0),
                decay_input=False,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
            ),
        ),
    ],
)
def test_if_lif_plif_multi_step_cupy_normalize_shape_and_release_sequence(
    monkeypatch, kernel_name, functional_call
):
    from spikingjelly.activation_based.cuda_kernel.auto_cuda import neuron_kernel

    x_seq = torch.randn(3, 2, 4, 5)
    v = torch.randn_like(x_seq[0])
    spike_flat = torch.randn(3, v.numel())
    v_flat = torch.randn(3, v.numel())
    calls = []

    def fake_kernel(*args):
        calls.append(args)
        return spike_flat, v_flat

    monkeypatch.setattr(neuron_kernel, kernel_name, fake_kernel)
    spike, v_next, v_seq = functional_call(x_seq, v)

    assert calls[-1][0].shape == (x_seq.shape[0], v.numel())
    assert calls[-1][1].shape == (v.numel(),)
    _assert_close(spike, spike_flat.reshape_as(x_seq))
    _assert_close(v_next, v_flat[-1].reshape_as(v))
    assert v_next.untyped_storage().data_ptr() != v_flat.untyped_storage().data_ptr()
    assert v_seq is None


def test_adaptive_current_reset_and_izhikevich_charge_match_module_helpers():
    torch.manual_seed(4)
    v = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(2, 3, requires_grad=True)
    spike = torch.rand(2, 3).requires_grad_()

    w_expected = neuron.AdaptBaseNode.jit_neuronal_adaptation(
        w,
        tau_w=2.5,
        a=0.4,
        v_rest=-0.1,
        v=v,
    )
    w_actual = functional.adaptive_current_update(
        w,
        v,
        tau_w=2.5,
        a=0.4,
        v_rest=-0.1,
    )
    _assert_close(w_actual, w_expected)

    v_hard, w_hard = functional.adaptive_reset(
        v,
        w,
        spike,
        v_threshold=0.7,
        v_reset=-0.2,
        b=0.3,
        detach_reset=True,
    )
    v_hard_expected, w_hard_expected = neuron.AdaptBaseNode.apply_hard_reset(
        v,
        w,
        spike.detach(),
        v_reset=-0.2,
        b=0.3,
        spike=spike,
    )
    _assert_close(v_hard, v_hard_expected)
    _assert_close(w_hard, w_hard_expected)

    v_soft, w_soft = functional.adaptive_reset(
        v,
        w,
        spike,
        v_threshold=0.7,
        v_reset=None,
        b=0.3,
        detach_reset=False,
    )
    v_soft_expected, w_soft_expected = neuron.AdaptBaseNode.apply_soft_reset(
        v,
        w,
        spike,
        v_threshold=0.7,
        b=0.3,
        spike=spike,
    )
    _assert_close(v_soft, v_soft_expected)
    _assert_close(w_soft, w_soft_expected)

    x = torch.randn(2, 3, requires_grad=True)
    module = neuron.IzhikevichNode(
        tau=2.5,
        v_c=0.2,
        a0=0.9,
        v_rest=-0.1,
        w_rest=0.0,
        backend="torch",
    )

    x_module = x.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    w_module = w.detach().clone().requires_grad_()
    module.v = v_module
    module.w = w_module
    module.neuronal_charge(x_module)
    loss_module = module.v.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    w_func = w.detach().clone().requires_grad_()
    v_next = functional.izhikevich_charge(
        x_func,
        v_func,
        w_func,
        tau=2.5,
        a0=0.9,
        v_rest=-0.1,
        v_c=0.2,
    )
    loss_func = v_next.sum()
    loss_func.backward()

    _assert_close(v_next, module.v)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)
    _assert_close(w_func.grad, w_module.grad)


def test_izhikevich_multi_step_cupy_normalizes_kernel_output(monkeypatch):
    from spikingjelly.activation_based import cuda_kernel

    x_seq = torch.randn(3, 2, 4)
    v = torch.randn_like(x_seq[0])
    w = torch.randn_like(x_seq[0])
    spike_flat = torch.randn(3, 8)
    v_flat = torch.randn(3, 8)
    w_flat = torch.randn(3, 8)
    calls = []

    def fake_kernel(*args):
        calls.append(args)
        return spike_flat, v_flat, w_flat

    monkeypatch.setattr(cuda_kernel, "multistep_izhikevich_ptt", fake_kernel)
    spike, v_next, w_next, v_seq, w_seq = functional.izhikevich_multi_step_cupy(
        x_seq,
        v,
        w,
        tau=2.0,
        v_threshold=1.0,
        v_reset=0.0,
        v_rest=0.0,
        a=0.0,
        b=0.0,
        tau_w=2.0,
        v_c=0.8,
        a0=1.0,
        detach_reset=False,
        surrogate_function=_make_surrogate(),
        store_state_seq=True,
    )

    assert calls[-1][0].shape == (3, 8)
    assert calls[-1][1].shape == (8,)
    assert calls[-1][2].shape == (8,)
    _assert_close(spike, spike_flat.reshape_as(x_seq))
    _assert_close(v_next, v_flat[-1].reshape_as(v))
    _assert_close(w_next, w_flat[-1].reshape_as(w))
    _assert_close(v_seq, v_flat.reshape_as(x_seq))
    _assert_close(w_seq, w_flat.reshape_as(x_seq))


def test_klif_cuba_lif_and_liaf_helpers_match_modules():
    torch.manual_seed(5)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    for decay_input in (False, True):
        for scale_reset in (False, True):
            for v_reset in (None, -0.2):
                module = neuron.KLIFNode(
                    scale_reset=scale_reset,
                    tau=2.5,
                    decay_input=decay_input,
                    v_threshold=0.7,
                    v_reset=v_reset,
                    backend="torch",
                )
                module.k.data.copy_(torch.tensor(1.3))
                x_module = x.detach().clone().requires_grad_()
                v_module = v.detach().clone().requires_grad_()
                module.v = v_module
                module.neuronal_charge(x_module)
                spike = module.neuronal_fire()
                module.neuronal_reset(spike)
                loss_module = spike.sum() + module.v.sum()
                loss_module.backward()

                x_func = x.detach().clone().requires_grad_()
                v_func = v.detach().clone().requires_grad_()
                k_func = module.k.detach().clone().requires_grad_()
                v_charged = functional.klif_charge(
                    x_func,
                    v_func,
                    k_func,
                    tau=2.5,
                    decay_input=decay_input,
                    v_reset=v_reset,
                )
                spike_func = module.surrogate_function(v_charged - 0.7)
                v_next = functional.klif_reset(
                    v_charged,
                    spike_func,
                    k_func,
                    v_threshold=0.7,
                    v_reset=v_reset,
                    scale_reset=scale_reset,
                    detach_reset=False,
                )
                loss_func = spike_func.sum() + v_next.sum()
                loss_func.backward()

                _assert_close(spike_func, spike)
                _assert_close(v_next, module.v)
                _assert_close(x_func.grad, x_module.grad)
                _assert_close(v_func.grad, v_module.grad)
                _assert_close(k_func.grad, module.k.grad)

    c = torch.randn(2, 3, requires_grad=True)
    module = neuron.CUBALIFNode(c_decay=0.4, v_decay=0.6)
    x_module = x.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    c_module = c.detach().clone().requires_grad_()
    module.v = v_module
    module.c = c_module
    module.neuronal_charge(x_module)
    loss_module = module.c.sum() + module.v.sum()
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    c_func = c.detach().clone().requires_grad_()
    c_next, v_next = functional.cuba_lif_charge(
        x_func,
        c_func,
        v_func,
        c_decay=0.4,
        v_decay=0.6,
    )
    loss_func = c_next.sum() + v_next.sum()
    loss_func.backward()

    _assert_close(c_next, module.c)
    _assert_close(v_next, module.v)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)
    _assert_close(c_func.grad, c_module.grad)

    for threshold_related in (False, True):
        module = neuron.LIAFNode(
            act=torch.tanh,
            threshold_related=threshold_related,
            tau=2.5,
            decay_input=True,
            v_threshold=0.7,
            v_reset=0.0,
            backend="torch",
        )
        x_module = x.detach().clone().requires_grad_()
        v_module = v.detach().clone().requires_grad_()
        module.v = v_module
        module.neuronal_charge(x_module)
        y_module = (
            module.act(module.v - module.v_threshold)
            if threshold_related
            else module.act(module.v)
        )
        y_module.sum().backward()

        x_func = x.detach().clone().requires_grad_()
        v_func = v.detach().clone().requires_grad_()
        v_charged = functional.lif_charge(
            x_func,
            v_func,
            tau=2.5,
            decay_input=True,
            v_reset=0.0,
        )
        y_func = functional.liaf_output(
            v_charged,
            v_threshold=0.7,
            act=torch.tanh,
            threshold_related=threshold_related,
        )
        y_func.sum().backward()

        _assert_close(y_func, y_module)
        _assert_close(x_func.grad, x_module.grad)
        _assert_close(v_func.grad, v_module.grad)


def test_mpbn_fire_matches_module_residual_paths():
    torch.manual_seed(6)

    v = torch.randn(2, 3)
    module = neuron.MPBNLIFNode(tau=2.5, out_features=3, mpbn=False, backend="torch")
    module.v = v.detach().clone()
    module.vth = torch.tensor([0.1, 0.2, 0.3])
    module.gamma = torch.tensor([1.1, 0.9, 1.2])
    module.mu = torch.tensor([0.0, -0.1, 0.2])
    module.beta = torch.tensor([0.3, -0.2, 0.1])
    module.sigma2 = torch.tensor([1.0, 1.2, 0.8])
    module.eps = 1e-5
    module.normalize_residual = True

    spike_module = module.neuronal_fire()

    v_func = v.detach().clone()
    spike_func, v_next = functional.mpbn_fire(
        v_func,
        module.vth,
        module.surrogate_function,
        normalize_residual=True,
        gamma=module.gamma,
        mu=module.mu,
        beta=module.beta,
        sigma2=module.sigma2,
        eps=module.eps,
    )

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)

    v4 = torch.randn(2, 3, 4, 4, requires_grad=True)
    module4 = neuron.MPBNLIFNode(tau=2.5, out_channels=3, mpbn=False, backend="torch")
    v4_module = v4.detach().clone().requires_grad_()
    module4.v = v4_module
    module4.vth = torch.tensor([0.1, 0.2, 0.3])
    spike_module4 = module4.neuronal_fire()
    loss_module4 = spike_module4.sum() + module4.v.sum()
    loss_module4.backward()

    v4_func = v4.detach().clone().requires_grad_()
    spike_func4, v4_next = functional.mpbn_fire(
        v4_func,
        module4.vth,
        module4.surrogate_function,
    )
    loss_func4 = spike_func4.sum() + v4_next.sum()
    loss_func4.backward()

    _assert_close(spike_func4, spike_module4)
    _assert_close(v4_next, module4.v)
    _assert_close(v4_func.grad, v4_module.grad)


def test_online_lif_charge_and_ottt_trace_update_match_nodes():
    torch.manual_seed(7)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    for cls in (neuron.OTTTLIFNode, neuron.SLTTLIFNode):
        for decay_input in (False, True):
            for v_reset in (None, 0.0, -0.2):
                module = cls(
                    tau=2.5,
                    decay_input=decay_input,
                    v_reset=v_reset,
                    backend="torch",
                ).train()
                x_module = x.detach().clone().requires_grad_()
                v_module = v.detach().clone().requires_grad_()
                module.v = v_module
                module.neuronal_charge(x_module)
                loss_module = module.v.sum()
                loss_module.backward()

                x_func = x.detach().clone().requires_grad_()
                v_func = v.detach().clone().requires_grad_()
                v_next = functional.online_lif_charge(
                    x_func,
                    v_func,
                    tau=2.5,
                    decay_input=decay_input,
                    v_reset=v_reset,
                )
                loss_func = v_next.sum()
                loss_func.backward()

                _assert_close(v_next, module.v)
                _assert_close(x_func.grad, x_module.grad)
                assert v_func.grad is None
                assert v_module.grad is None

    spike = torch.rand(2, 3, requires_grad=True)
    trace = torch.rand(2, 3, requires_grad=True)
    expected = neuron.OTTTLIFNode.track_trace(spike, trace, tau=2.5)
    actual = functional.ottt_trace_update(spike, trace, tau=2.5)
    _assert_close(actual, expected)
    assert actual.grad_fn is None
    assert not actual.requires_grad


def test_lava_cuba_lif_helpers_match_module():
    torch.manual_seed(8)
    x = torch.rand(2, 3, requires_grad=True)
    current = torch.rand(2, 3, requires_grad=True)
    voltage = torch.rand(2, 3, requires_grad=True)

    module = lava_exchange.CubaLIFNode(
        current_decay=0.25,
        voltage_decay=0.5,
        requires_grad=True,
        detach_reset=True,
    )
    current_module = current.detach().clone().requires_grad_()
    voltage_module = voltage.detach().clone().requires_grad_()
    module.current_state = current_module
    module.voltage_state = voltage_module
    x_module = x.detach().clone().requires_grad_()
    spike_module = module.single_step_forward(x_module)
    loss_module = (
        spike_module.sum() + module.current_state.sum() + module.voltage_state.sum()
    )
    loss_module.backward()

    current_decay_func = module.current_decay.detach().clone().requires_grad_()
    voltage_decay_func = module.voltage_decay.detach().clone().requires_grad_()
    x_func = x.detach().clone().requires_grad_()
    current_func = current.detach().clone().requires_grad_()
    voltage_func = voltage.detach().clone().requires_grad_()
    spike_func, current_next, voltage_next = functional.lava_cuba_lif_single_step(
        x_func,
        current_func,
        voltage_func,
        current_decay_func,
        voltage_decay_func,
        s_scale=module.s_scale,
        v_threshold=module.v_threshold,
        v_threshold_eps=module.v_threshold_eps,
        v_reset=module.v_reset,
        surrogate_function=module.surrogate_function,
        detach_reset=module.detach_reset,
    )
    loss_func = spike_func.sum() + current_next.sum() + voltage_next.sum()
    loss_func.backward()

    _assert_close(spike_func, spike_module)
    _assert_close(current_next, module.current_state)
    _assert_close(voltage_next, module.voltage_state)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(current_func.grad, current_module.grad)
    _assert_close(voltage_func.grad, voltage_module.grad)
    _assert_close(current_decay_func.grad, module.current_decay.grad)
    _assert_close(voltage_decay_func.grad, module.voltage_decay.grad)

    x_seq = torch.rand(4, 2, 3)
    module_seq = lava_exchange.CubaLIFNode(
        current_decay=0.25,
        voltage_decay=0.5,
        store_i_seq=True,
        store_v_seq=True,
    )
    module_seq.current_state = current.detach().clone()
    module_seq.voltage_state = voltage.detach().clone()
    spike_seq_module = module_seq.multi_step_forward(x_seq)

    spike_seq_func, current_last, voltage_last, current_seq, voltage_seq = (
        functional.lava_cuba_lif_multi_step(
            x_seq,
            current,
            voltage,
            module_seq.current_decay,
            module_seq.voltage_decay,
            s_scale=module_seq.s_scale,
            v_threshold=module_seq.v_threshold,
            v_threshold_eps=module_seq.v_threshold_eps,
            v_reset=module_seq.v_reset,
            surrogate_function=module_seq.surrogate_function,
            detach_reset=module_seq.detach_reset,
            store_i_seq=True,
            store_v_seq=True,
        )
    )

    _assert_close(spike_seq_func, spike_seq_module)
    _assert_close(current_last, module_seq.current_state)
    _assert_close(voltage_last, module_seq.voltage_state)
    _assert_close(current_seq, module_seq.i_seq)
    _assert_close(voltage_seq, module_seq.v_seq)


def test_lif_pre_spike_mean_helpers_match_save_v_lif_node():
    torch.manual_seed(9)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    module = spike_dhs.save_v_LIFNode(
        tau=1.25,
        decay_input=False,
        v_threshold=0.5,
        detach_reset=True,
        surrogate_function=_make_surrogate(),
    )
    v_module = v.detach().clone().requires_grad_()
    x_module = x.detach().clone().requires_grad_()
    module.v = v_module
    spike_module = module.single_step_forward(x_module)
    loss_module = spike_module.sum() + module.v.sum() + module.v_before_spike
    loss_module.backward()

    x_func = x.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    spike_func, v_next, pre_spike_mean = functional.lif_single_step_with_pre_spike_mean(
        x_func,
        v_func,
        tau=1.25,
        decay_input=False,
        v_threshold=0.5,
        v_reset=module.v_reset,
        surrogate_function=_make_surrogate(),
        detach_reset=True,
    )
    loss_func = spike_func.sum() + v_next.sum() + pre_spike_mean
    loss_func.backward()

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)
    _assert_close(pre_spike_mean, module.v_before_spike)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(v_func.grad, v_module.grad)

    x_seq = torch.randn(4, 2, 3)
    module_seq = spike_dhs.save_v_LIFNode(
        tau=1.25,
        decay_input=False,
        v_threshold=0.5,
        detach_reset=True,
        surrogate_function=_make_surrogate(),
        store_v_seq=True,
    )
    module_seq.v = v.detach().clone()
    spike_seq_module = module_seq.multi_step_forward(x_seq)
    spike_seq_func, v_last, pre_spike_mean_seq = (
        functional.lif_multi_step_with_pre_spike_mean(
            x_seq,
            v,
            tau=1.25,
            decay_input=False,
            v_threshold=0.5,
            v_reset=module_seq.v_reset,
            surrogate_function=_make_surrogate(),
            detach_reset=True,
            store_pre_spike_mean_seq=True,
        )
    )

    _assert_close(spike_seq_func, spike_seq_module)
    _assert_close(v_last, module_seq.v)
    _assert_close(pre_spike_mean_seq, module_seq.v_seq)


def test_masked_psn_single_step_helpers_match_module_and_gradients():
    torch.manual_seed(10)
    x_seq = torch.randn(3, 8, requires_grad=True)
    module = neuron_psn.MaskedPSN(
        k=2,
        T=3,
        lambda_init=1.0,
        surrogate_function=_make_surrogate(),
    )
    module_queue = module.queue

    x_module = x_seq.detach().clone().requires_grad_()
    y_module = torch.stack([module(x) for x in x_module])
    y_module.sum().backward()

    x_func = x_seq.detach().clone().requires_grad_()
    weight_func = module.weight.detach().clone().requires_grad_()
    bias_func = module.bias.detach().clone().requires_grad_()
    queue = ()
    time_step = 0
    ys = []
    for x in x_func:
        queue = functional.masked_psn_advance_queue(x, queue, module.k)
        y, time_step = functional.masked_psn_single_step_from_queue(
            x.shape,
            queue,
            time_step,
            module.T,
            module.lambda_,
            module.mask0,
            module.mask1,
            weight_func,
            bias_func,
            _make_surrogate(),
        )
        ys.append(y)
    y_func = torch.stack(ys)
    y_func.sum().backward()

    _assert_close(y_func, y_module)
    assert module.queue is module_queue
    assert time_step == module.time_step
    assert len(queue) == len(module.queue)
    for actual, expected in zip(queue, module.queue):
        _assert_close(actual, expected)
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(weight_func.grad, module.weight.grad)
    _assert_close(bias_func.grad, module.bias.grad)


def test_masked_psn_overflow_preserves_existing_queue_side_effect():
    module = neuron_psn.MaskedPSN(
        k=1,
        T=1,
        lambda_init=1.0,
        surrogate_function=_make_surrogate(),
    )
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0]])

    module(first)
    assert module.time_step == 1
    _assert_close(module.queue[0], first.flatten())

    with pytest.raises(OverflowError):
        module(second)

    assert module.time_step == 1
    assert len(module.queue) == 1
    _assert_close(module.queue[0], second.flatten())


def test_sliding_psn_single_step_matches_module_and_gradients():
    torch.manual_seed(14)
    x_seq = torch.randn(5, 2, 3)

    module = neuron.SlidingPSN(
        k=3, exp_init=False, surrogate_function=_make_surrogate(), step_mode="s"
    ).train()
    module_queue = module.queue
    x_module = x_seq.detach().clone().requires_grad_()
    y_module = torch.stack([module(x) for x in x_module])
    y_module.sum().backward()

    x_func = x_seq.detach().clone().requires_grad_()
    weight_func = module.weight.detach().clone().requires_grad_()
    bias_func = module.bias.detach().clone().requires_grad_()
    queue = ()
    ys = []
    for x in x_func:
        y, queue = functional.sliding_psn_single_step(
            x, queue, weight_func, bias_func, _make_surrogate()
        )
        ys.append(y)
    y_func = torch.stack(ys)
    y_func.sum().backward()

    _assert_close(y_func, y_module)
    assert module.queue is module_queue
    _assert_close(x_func.grad, x_module.grad)
    _assert_close(weight_func.grad, module.weight.grad)
    _assert_close(bias_func.grad, module.bias.grad)
    assert len(queue) == module.k
    assert all(q.shape == x_seq[0].flatten().shape for q in queue)


def test_sliding_psn_single_step_does_not_mutate_queue_tuple():
    x = torch.randn(2, 3)
    old0 = torch.ones(6)
    old1 = torch.ones(6) * 2
    old2 = torch.ones(6) * 3
    old3 = torch.ones(6) * 4
    queue = (old0, old1, old2, old3)
    weight = torch.tensor([0.25, 0.5, 1.0])
    bias = torch.tensor(-1.0)

    _, queue_next = functional.sliding_psn_single_step(
        x, queue, weight, bias, _make_surrogate()
    )

    assert queue == (old0, old1, old2, old3)
    assert torch.allclose(old0, torch.ones(6))
    assert len(queue_next) == len(queue)
    assert queue_next[0] is old1
    assert queue_next[-1].shape == x.flatten().shape


def _gated_lif_reference(
    x_seq,
    v,
    time_steps,
    alpha,
    beta,
    gamma,
    tau,
    v_threshold,
    linear_decay,
    v_subreset,
    conduct,
    surrogate_function,
):
    alpha = alpha.view(1, -1, 1, 1).sigmoid()
    beta = beta.view(1, -1, 1, 1).sigmoid()
    gamma = gamma.view(1, -1, 1, 1).sigmoid()
    tau = tau.view(1, -1, 1, 1).sigmoid()
    v_threshold = v_threshold.view(1, -1, 1, 1).sigmoid()
    linear_decay = linear_decay.view(1, -1, 1, 1).sigmoid()
    v_subreset = v_subreset.view(1, -1, 1, 1).sigmoid()

    spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
    ys = []
    u = v
    for t in range(time_steps):
        input_current = x_seq[t] * (
            1 - beta * (1 - conduct[t].view(1, -1, 1, 1).sigmoid())
        )
        u = ((1 - alpha * (1 - tau)) * v - (1 - alpha) * linear_decay) + input_current
        u = (
            u
            - (1 - alpha * (1 - tau)) * v * gamma * spike
            - (1 - gamma) * v_subreset * spike
        )
        spike = surrogate_function(u - v_threshold)
        v = u
        ys.append(spike)
    return torch.stack(ys), u, v


@pytest.mark.parametrize("channel_wise", [False, True])
def test_gated_lif_multi_step_matches_reference_and_module(channel_wise):
    torch.manual_seed(15)
    time_steps = 4
    channels = 3
    x_seq = torch.randn(time_steps + 1, 2, channels, 4, 4)
    module = neuron.GatedLIFNode(
        T=time_steps,
        inplane=channels if channel_wise else None,
        surrogate_function=_make_surrogate(),
    ).train()

    x_func = x_seq.detach().clone().requires_grad_()
    v_func = torch.zeros_like(x_func[0]).requires_grad_()
    param_names = (
        "alpha",
        "beta",
        "gamma",
        "tau",
        "v_threshold",
        "linear_decay",
        "v_subreset",
        "conduct",
    )
    params_func = {
        name: getattr(module, name).detach().clone().requires_grad_()
        for name in param_names
    }
    y_func, u_func, v_next_func = functional.gated_lif_multi_step(
        x_func,
        v_func,
        time_steps,
        params_func["alpha"],
        params_func["beta"],
        params_func["gamma"],
        params_func["tau"],
        params_func["v_threshold"],
        params_func["linear_decay"],
        params_func["v_subreset"],
        params_func["conduct"],
        _make_surrogate(),
    )
    loss_func = y_func.sum() + u_func.sum() + v_next_func.sum()
    loss_func.backward()

    x_ref = x_seq.detach().clone().requires_grad_()
    v_ref = torch.zeros_like(x_ref[0]).requires_grad_()
    params_ref = {
        name: getattr(module, name).detach().clone().requires_grad_()
        for name in param_names
    }
    y_ref, u_ref, v_next_ref = _gated_lif_reference(
        x_ref,
        v_ref,
        time_steps,
        params_ref["alpha"],
        params_ref["beta"],
        params_ref["gamma"],
        params_ref["tau"],
        params_ref["v_threshold"],
        params_ref["linear_decay"],
        params_ref["v_subreset"],
        params_ref["conduct"],
        _make_surrogate(),
    )
    loss_ref = y_ref.sum() + u_ref.sum() + v_next_ref.sum()
    loss_ref.backward()

    _assert_close(y_func, y_ref)
    _assert_close(u_func, u_ref)
    _assert_close(v_next_func, v_next_ref)
    _assert_close(x_func.grad, x_ref.grad)
    _assert_close(v_func.grad, v_ref.grad)
    for name in param_names:
        _assert_close(params_func[name].grad, params_ref[name].grad)

    x_module = x_seq.detach().clone().requires_grad_()
    y_module = module(x_module)
    _assert_close(y_module, y_ref.detach())
    _assert_close(module.u, u_ref.detach())
    _assert_close(module.v, v_next_ref.detach())
    assert y_module.shape[0] == time_steps


def _activation_aware_if_single_reference(
    x,
    v,
    threshold,
    offset,
    v_reset,
    surrogate_function,
    detach_reset,
):
    h = v + x
    spike = surrogate_function(h + offset - threshold)
    spike_d = spike.detach() if detach_reset else spike
    if v_reset is None:
        v_next = h - spike_d * threshold
    else:
        v_next = spike_d * v_reset + (1.0 - spike_d) * h
    return spike, v_next


@pytest.mark.parametrize("v_reset", [None, 0.25])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_activation_aware_if_single_step_matches_reference_and_gradients(
    v_reset, detach_reset
):
    torch.manual_seed(16)
    x = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)
    threshold = torch.tensor([0.8, 1.0, 1.2]).view(1, 3).requires_grad_()
    offset = torch.tensor([-0.1, 0.0, 0.1]).view(1, 3).requires_grad_()

    spike_func, v_next_func = functional.activation_aware_if_single_step(
        x,
        v,
        threshold,
        offset,
        v_reset,
        _make_surrogate(),
        detach_reset,
    )
    loss_func = spike_func.sum() + v_next_func.sum()
    loss_func.backward()

    x_ref = x.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    threshold_ref = threshold.detach().clone().requires_grad_()
    offset_ref = offset.detach().clone().requires_grad_()
    spike_ref, v_next_ref = _activation_aware_if_single_reference(
        x_ref,
        v_ref,
        threshold_ref,
        offset_ref,
        v_reset,
        _make_surrogate(),
        detach_reset,
    )
    loss_ref = spike_ref.sum() + v_next_ref.sum()
    loss_ref.backward()

    _assert_close(spike_func, spike_ref)
    _assert_close(v_next_func, v_next_ref)
    _assert_close(x.grad, x_ref.grad)
    _assert_close(v.grad, v_ref.grad)
    _assert_close(threshold.grad, threshold_ref.grad)
    _assert_close(offset.grad, offset_ref.grad)


@pytest.mark.parametrize("store_v_seq", [False, True])
def test_activation_aware_if_multi_step_matches_module_channelwise(store_v_seq):
    torch.manual_seed(17)
    x_seq = torch.randn(5, 2, 3, 4).requires_grad_()
    v_init = torch.randn_like(x_seq[0]).requires_grad_()
    threshold = torch.tensor([0.8, 1.0, 1.2]).view(1, 3, 1)
    offset = torch.tensor([-0.1, 0.0, 0.1]).view(1, 3, 1)

    spike_func, v_next_func, v_seq_func = functional.activation_aware_if_multi_step(
        x_seq,
        v_init,
        threshold,
        offset,
        v_reset=None,
        surrogate_function=_make_surrogate(),
        detach_reset=True,
        store_v_seq=store_v_seq,
    )

    module = neuron.ActivationAwareIFNode(
        v_threshold=torch.tensor([0.8, 1.0, 1.2]),
        v_offset=torch.tensor([-0.1, 0.0, 0.1]),
        channel_dim=1,
        surrogate_function=_make_surrogate(),
        detach_reset=True,
        step_mode="m",
        store_v_seq=store_v_seq,
    )
    module.v = v_init.detach().clone()
    spike_module = module(x_seq.detach())

    _assert_close(spike_func, spike_module)
    _assert_close(v_next_func, module.v)
    if store_v_seq:
        _assert_close(v_seq_func, module.v_seq)
    else:
        assert v_seq_func is None
        assert not hasattr(module, "v_seq") or module.v_seq is None


@pytest.mark.parametrize("store_v_seq", [False, True])
def test_activation_aware_if_multi_step_triton_normalizes_kernel_output(
    monkeypatch, store_v_seq
):
    from spikingjelly.activation_based.triton_kernel import neuron_kernel

    x_seq = torch.randn(3, 2, 4)
    v_init = torch.randn_like(x_seq[0])
    threshold = torch.tensor(1.0)
    offset = torch.tensor(0.0)
    v_seq = x_seq.cumsum(0) + v_init
    spike_seq = (v_seq >= threshold).to(x_seq)
    kernel_v_out = v_seq if store_v_seq else v_seq[-1]
    calls = []

    class FakeKernel:
        @staticmethod
        def _multistep_activation_aware_if(*args, **kwargs):
            calls.append((args, kwargs))
            return spike_seq, kernel_v_out

    monkeypatch.setattr(neuron_kernel, "activation_aware_if", FakeKernel)
    spike, v_next, stored_v_seq = functional.activation_aware_if_multi_step_triton(
        x_seq,
        v_init,
        threshold,
        offset,
        channel_size=1,
        inner_size=x_seq[0].numel(),
        v_reset=None,
        store_v_seq=store_v_seq,
    )

    assert len(calls) == 1
    _assert_close(spike, spike_seq)
    _assert_close(v_next, v_seq[-1])
    if store_v_seq:
        _assert_close(stored_v_seq, v_seq)
        assert v_next.data_ptr() != v_seq[-1].data_ptr()
    else:
        assert stored_v_seq is None
        assert v_next is kernel_v_out


def _stbif_single_reference(x, q, acc_q, q_threshold, pos_max, neg_min):
    normalized = x / q_threshold
    q_next = q + normalized.detach()
    acc_q_next = torch.round(acc_q)
    spike_position = (q_next - 1 >= 0) & (acc_q_next < pos_max)
    neg_spike_position = (q_next < 0) & (acc_q_next > neg_min)
    cur_output_next = spike_position.to(x.dtype) - neg_spike_position.to(x.dtype)
    acc_q_next = acc_q_next + cur_output_next
    q_next = torch.where(spike_position, q_next - 1, q_next)
    q_next = torch.where(neg_spike_position, q_next + 1, q_next)
    is_work = bool((normalized != 0).any() | (cur_output_next != 0).any())
    return cur_output_next * q_threshold, q_next, acc_q_next, cur_output_next, is_work


def test_stbif_single_step_matches_reference_and_does_not_mutate_state():
    x = torch.tensor([[0.35, -0.25, 0.0], [0.6, -0.8, 0.15]])
    q = torch.full_like(x, 0.5)
    acc_q = torch.zeros_like(x)
    q_threshold = torch.tensor(0.25)
    pos_max = torch.tensor(1.0)
    neg_min = torch.tensor(-2.0)
    q_before = q.clone()
    acc_before = acc_q.clone()

    actual = functional.stbif_single_step(x, q, acc_q, q_threshold, pos_max, neg_min)
    expected = _stbif_single_reference(x, q, acc_q, q_threshold, pos_max, neg_min)

    for actual_value, expected_value in zip(actual[:4], expected[:4], strict=True):
        _assert_close(actual_value, expected_value)
    assert actual[4] is expected[4]
    _assert_close(q, q_before)
    _assert_close(acc_q, acc_before)


def test_stbif_single_step_preserves_cur_output_identity():
    module = neuron.STBIFNeuron(q_threshold=0.25, level=4, sym=True)
    module(torch.tensor([0.5, -0.25]))
    cur_output = module.cur_output

    module(torch.tensor([0.0, 0.25]))

    assert module.cur_output is cur_output


def test_stbif_multi_step_torch_matches_reference_and_module():
    x_seq = torch.tensor(
        [
            [[0.35, -0.25, 0.0], [0.6, -0.8, 0.15]],
            [[0.2, 0.1, -0.3], [-0.4, 0.5, 0.0]],
            [[0.0, -0.6, 0.7], [0.1, 0.0, -0.2]],
        ]
    )
    q_threshold = torch.tensor(0.25)
    pos_max = torch.tensor(1.0)
    neg_min = torch.tensor(-2.0)
    q = torch.full_like(x_seq[0], 0.5)
    acc_q = torch.zeros_like(x_seq[0])

    out_seq, q_next, acc_q_next, cur_output_next, is_work = (
        functional.stbif_multi_step_torch(
            x_seq, q, acc_q, q_threshold, pos_max, neg_min
        )
    )
    expected_out = []
    q_ref = q
    acc_ref = acc_q
    cur_ref = None
    work_ref = False
    for x in x_seq:
        out, q_ref, acc_ref, cur_ref, step_work = _stbif_single_reference(
            x, q_ref, acc_ref, q_threshold, pos_max, neg_min
        )
        expected_out.append(out)
        work_ref = work_ref or step_work

    _assert_close(out_seq, torch.stack(expected_out))
    _assert_close(q_next, q_ref)
    _assert_close(acc_q_next, acc_ref)
    _assert_close(cur_output_next, cur_ref)
    assert is_work is work_ref

    module = neuron.STBIFNeuron(
        q_threshold=q_threshold,
        level=4,
        sym=True,
        pos_max=pos_max,
        neg_min=neg_min,
        step_mode="m",
    )
    y_module = module(x_seq)
    _assert_close(y_module, out_seq)
    _assert_close(module.q, q_next)
    _assert_close(module.acc_q, acc_q_next)
    _assert_close(module.cur_output, cur_output_next)
    assert module.is_work is is_work


def test_stbif_multi_step_torch_synchronizes_work_flag_once(monkeypatch):
    original_bool = torch.Tensor.__bool__
    bool_calls = []

    def counted_bool(value):
        bool_calls.append(value)
        return original_bool(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    x_seq = torch.randn(4, 2, 3)
    functional.stbif_multi_step_torch(
        x_seq,
        torch.zeros_like(x_seq[0]),
        torch.zeros_like(x_seq[0]),
        torch.tensor(0.25),
        torch.tensor(1.0),
        torch.tensor(-2.0),
    )

    assert len(bool_calls) == 1


@pytest.mark.parametrize(
    ("module_factory", "functional_call"),
    [
        (
            lambda store_v_seq: neuron.IFNode(
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                backend="torch",
                step_mode="m",
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, store_v_seq: functional.if_multi_step(
                x_seq,
                v,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
        ),
        (
            lambda store_v_seq: neuron.LIFNode(
                tau=2.5,
                decay_input=False,
                v_threshold=0.6,
                v_reset=0.0,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                backend="torch",
                step_mode="m",
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, store_v_seq: functional.lif_multi_step(
                x_seq,
                v,
                tau=2.5,
                decay_input=False,
                v_threshold=0.6,
                v_reset=0.0,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
        ),
        (
            lambda store_v_seq: neuron.ParametricLIFNode(
                init_tau=2.5,
                decay_input=True,
                v_threshold=0.6,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                backend="torch",
                step_mode="m",
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, store_v_seq: functional.plif_multi_step(
                x_seq,
                v,
                torch.tensor(-math.log(2.5 - 1.0), requires_grad=True),
                decay_input=True,
                v_threshold=0.6,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
        ),
    ],
)
@pytest.mark.parametrize("store_v_seq", [False, True])
def test_multi_step_matches_module_torch_training(
    module_factory, functional_call, store_v_seq
):
    torch.manual_seed(3)
    x_seq = torch.randn(4, 2, 3, requires_grad=True)
    v = torch.randn(2, 3, requires_grad=True)

    module = module_factory(store_v_seq).train()
    x_module = x_seq.detach().clone().requires_grad_()
    v_module = v.detach().clone().requires_grad_()
    module.v = v_module
    spike_module = module(x_module)

    x_func = x_seq.detach().clone().requires_grad_()
    v_func = v.detach().clone().requires_grad_()
    spike_func, v_next, v_seq = functional_call(x_func, v_func, store_v_seq)

    _assert_close(spike_func, spike_module)
    _assert_close(v_next, module.v)
    if store_v_seq:
        _assert_close(v_seq, module.v_seq)
    else:
        assert v_seq is None


@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize(
    ("reference_call", "inductor_call", "uses_w"),
    [
        (
            lambda x_seq, v, w, store_v_seq: functional.if_multi_step(
                x_seq,
                v,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, w, store_v_seq: functional.if_multi_step_inductor(
                x_seq,
                v,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            False,
        ),
        (
            lambda x_seq, v, w, store_v_seq: functional.lif_multi_step(
                x_seq,
                v,
                tau=2.5,
                decay_input=False,
                v_threshold=0.6,
                v_reset=0.0,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, w, store_v_seq: functional.lif_multi_step_inductor(
                x_seq,
                v,
                tau=2.5,
                decay_input=False,
                v_threshold=0.6,
                v_reset=0.0,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            False,
        ),
        (
            lambda x_seq, v, w, store_v_seq: functional.plif_multi_step(
                x_seq,
                v,
                w,
                decay_input=True,
                v_threshold=0.6,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            lambda x_seq, v, w, store_v_seq: functional.plif_multi_step_inductor(
                x_seq,
                v,
                w,
                decay_input=True,
                v_threshold=0.6,
                v_reset=None,
                surrogate_function=_make_surrogate(),
                detach_reset=True,
                store_v_seq=store_v_seq,
            ),
            True,
        ),
    ],
)
def test_inductor_multi_step_matches_torch_functional_and_reuses_cache(
    identity_inductor_compile,
    reference_call,
    inductor_call,
    uses_w,
    store_v_seq,
):
    torch.manual_seed(8)
    x_seq = torch.randn(4, 2, 3)
    v = torch.randn(2, 3)
    w = torch.tensor(-math.log(2.5 - 1.0))

    x_ref = x_seq.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    spike_ref, v_next_ref, v_seq_ref = reference_call(x_ref, v_ref, w_ref, store_v_seq)
    loss_ref = spike_ref.sum() + v_next_ref.sum()
    if v_seq_ref is not None:
        loss_ref = loss_ref + v_seq_ref.sum()
    loss_ref.backward()

    x_inductor = x_seq.detach().clone().requires_grad_()
    v_inductor = v.detach().clone().requires_grad_()
    w_inductor = w.detach().clone().requires_grad_()
    spike_inductor, v_next_inductor, v_seq_inductor = inductor_call(
        x_inductor, v_inductor, w_inductor, store_v_seq
    )
    loss_inductor = spike_inductor.sum() + v_next_inductor.sum()
    if v_seq_inductor is not None:
        loss_inductor = loss_inductor + v_seq_inductor.sum()
    loss_inductor.backward()

    _assert_close(spike_inductor, spike_ref)
    _assert_close(v_next_inductor, v_next_ref)
    _assert_close(x_inductor.grad, x_ref.grad)
    _assert_close(v_inductor.grad, v_ref.grad)
    if store_v_seq:
        _assert_close(v_seq_inductor, v_seq_ref)
    else:
        assert v_seq_inductor is None
    if uses_w:
        _assert_close(w_inductor.grad, w_ref.grad)
    else:
        assert w_inductor.grad is None

    inductor_call(
        x_seq.detach().clone().requires_grad_(),
        v.detach().clone().requires_grad_(),
        w.detach().clone().requires_grad_(),
        store_v_seq,
    )
    assert len(identity_inductor_compile) == 1
    assert inductor_cache.info()["entries"] == 1


@pytest.mark.parametrize(
    "functional_call",
    [
        lambda x, v: functional.if_single_step(
            x, v, 1.0, 0.0, _make_surrogate(), False
        ),
        lambda x, v: functional.lif_single_step(
            x, v, 2.0, True, 1.0, None, _make_surrogate(), True
        ),
        lambda x, v: functional.plif_single_step(
            x,
            v,
            torch.tensor(-math.log(2.0 - 1.0)),
            True,
            1.0,
            0.0,
            _make_surrogate(),
            False,
        ),
    ],
)
def test_single_step_does_not_mutate_input_or_state(functional_call):
    torch.manual_seed(4)
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    x_before = x.clone()
    v_before = v.clone()

    functional_call(x, v)

    assert torch.equal(x, x_before)
    assert torch.equal(v, v_before)


def test_functional_neuron_top_level_exports():
    for name in functional_neuron.__all__:
        assert getattr(functional, name) is getattr(functional_neuron, name)


def test_functional_neuron_exports_do_not_collide_with_existing_functional_modules():
    modules = (forward, loss, misc, net_config, functional_neuron, online_learning)
    seen = {}
    for module in modules:
        for name in module.__all__:
            previous = seen.setdefault(name, module.__name__)
            assert previous == module.__name__, (
                f"`{name}` exported by both {previous} and {module.__name__}"
            )


def test_functional_neuron_public_functions_have_annotations_and_bilingual_docstrings():
    for name in functional_neuron.__all__:
        fn = getattr(functional_neuron, name)
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
