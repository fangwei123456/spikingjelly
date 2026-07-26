import gc
import os

import pytest
import torch
from spikingjelly import configure
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.cuda_kernel import (
    multistep_eif_ptt,
    multistep_izhikevich_ptt,
    multistep_qif_ptt,
)
from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
    register_python_object,
    resolve_python_object,
)
from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear
from spikingjelly.activation_based.cuda_kernel.tensor_cache import BoolTensorCache


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")


def _require_cupy():
    pytest.importorskip("cupy")


def _maybe_skip_custom_op_unavailable():
    if not all(
        hasattr(torch.library, name)
        for name in ("custom_op", "register_fake", "register_autograd")
    ):
        pytest.skip("torch.library custom_op/register_autograd are unavailable.")


def test_python_object_registry_uses_identity_and_releases_objects():
    class Kernel:
        pass

    first = Kernel()
    second = Kernel()
    first_id = register_python_object(first)

    assert register_python_object(first) == first_id
    assert register_python_object(second) != first_id
    assert resolve_python_object(first_id) is first

    del first
    gc.collect()
    with pytest.raises(RuntimeError, match="Unknown python object"):
        resolve_python_object(first_id)


@pytest.mark.parametrize("level", [0, 1])
def test_bool_tensor_cache_balances_repeated_stores(level, monkeypatch):
    monkeypatch.setattr(configure, "save_bool_spike_level", level)
    spike = (torch.arange(17) % 3 == 0).float()
    cache = BoolTensorCache()

    first_key = cache.store_bool(spike)
    second_key = cache.store_bool(spike)

    assert first_key == second_key
    assert torch.equal(cache.get_float(first_key, spike.shape), spike)
    assert torch.equal(cache.get_float(second_key, spike.shape), spike)


def test_spike_linear_backward_no_bias_cuda():
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    x = (torch.randn(8, 32, device="cuda") > 0).float().requires_grad_(True)
    weight = torch.randn(16, 32, device="cuda", requires_grad=True)

    y = spike_linear(x, weight, None)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (8, 16)
    assert x.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize(
    "kernel_fn,args_builder",
    [
        (
            multistep_qif_ptt,
            lambda sg: (
                2.0,
                1.0,
                0.0,
                0.0,
                0.8,
                1.0,
                False,
                sg,
            ),
        ),
        (
            multistep_eif_ptt,
            lambda sg: (
                2.0,
                1.0,
                0.0,
                0.0,
                -52.0,
                2.0,
                False,
                sg,
            ),
        ),
        (
            multistep_izhikevich_ptt,
            lambda sg: (
                torch.zeros(64, device="cuda", requires_grad=True),
                2.0,
                1.0,
                0.0,
                -65.0,
                0.02,
                0.2,
                30.0,
                30.0,
                1.0,
                False,
                sg,
            ),
        ),
    ],
)
def test_multistep_ptt_wrappers_cuda_forward_backward(kernel_fn, args_builder):
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    sg = surrogate.ATan()
    x_seq = torch.randn(4, 64, device="cuda", requires_grad=True)
    v_init = torch.zeros(64, device="cuda", requires_grad=True)

    if kernel_fn is multistep_izhikevich_ptt:
        args = args_builder(sg)
        w_init = args[0]
        other = args[1:]
        spike_seq, v_seq, w_seq = kernel_fn(x_seq, v_init, w_init, *other)
        loss = spike_seq.mean() + v_seq.mean() + w_seq.mean()
    else:
        spike_seq, v_seq = kernel_fn(x_seq, v_init, *args_builder(sg))
        loss = spike_seq.mean() + v_seq.mean()

    loss.backward()

    assert x_seq.grad is not None
    assert v_init.grad is not None


def test_disable_cupy_custom_op_env_fallback():
    _require_cuda()
    _require_cupy()

    old = os.environ.get("SJ_USE_CUPY_OP")
    os.environ["SJ_USE_CUPY_OP"] = "0"
    try:
        x = (torch.randn(2, 8, device="cuda") > 0).float().requires_grad_(True)
        w = torch.randn(4, 8, device="cuda", requires_grad=True)
        y = spike_linear(x, w, None)
        y.sum().backward()
        assert y.shape == (2, 4)
        assert x.grad is not None
        assert w.grad is not None
    finally:
        if old is None:
            del os.environ["SJ_USE_CUPY_OP"]
        else:
            os.environ["SJ_USE_CUPY_OP"] = old
