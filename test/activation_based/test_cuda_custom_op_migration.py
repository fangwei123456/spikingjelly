import os

import pytest
import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.cuda_kernel import (
    multistep_eif_ptt,
    multistep_izhikevich_ptt,
    multistep_qif_ptt,
)
from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear


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
        w_init = args_builder(sg)[0]
        other = args_builder(sg)[1:]
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
