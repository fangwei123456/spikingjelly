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


def test_spike_linear_backward_with_bias_cuda():
    """spike_linear with a bias vector on CUDA: gradients flow to bias."""
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    spike = (torch.randn(6, 16, device="cuda") > 0).float().requires_grad_(True)
    weight = torch.randn(8, 16, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)

    y = spike_linear(spike, weight, bias)
    y.sum().backward()

    assert y.shape == (6, 8)
    assert weight.grad is not None
    assert bias.grad is not None


def test_spike_linear_grad_matches_reference_cuda():
    """Gradients from spike_linear match reference F.linear gradients."""
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    spike_data = (torch.randn(4, 8, device="cuda") > 0).float()

    # spike_linear path
    s1 = spike_data.clone().requires_grad_(True)
    w1 = torch.randn(6, 8, device="cuda", requires_grad=True)
    y1 = spike_linear(s1, w1, None)
    y1.sum().backward()

    # F.linear reference with same weights
    s2 = spike_data.clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.linear(s2, w2, None)
    y2.sum().backward()

    assert torch.allclose(w1.grad, w2.grad, atol=1e-5), (
        "Weight gradients from spike_linear and F.linear should match"
    )


def test_multistep_lif_ptt_cuda():
    """multistep_lif_ptt forward+backward runs without error on CUDA."""
    _require_cuda()
    _require_cupy()

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel import multistep_lif_ptt

    sg = surrogate.ATan()
    x_seq = torch.randn(6, 64, device="cuda", requires_grad=True)
    v_init = torch.zeros(64, device="cuda", requires_grad=True)

    spike_seq, v_seq = multistep_lif_ptt(
        x_seq, v_init, True, 2.0, 1.0, None, False, sg
    )
    loss = spike_seq.mean() + v_seq.mean()
    loss.backward()

    assert spike_seq.shape == (6, 64)
    assert v_seq.shape == (6, 64)
    assert x_seq.grad is not None
    assert v_init.grad is not None


def test_multistep_if_ptt_hard_reset_cuda():
    """multistep_if_ptt with hard reset works on CUDA."""
    _require_cuda()
    _require_cupy()

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.cuda_kernel import multistep_if_ptt

    sg = surrogate.ATan()
    x_seq = torch.randn(4, 64, device="cuda", requires_grad=True)
    v_init = torch.zeros(64, device="cuda", requires_grad=True)

    spike_seq, v_seq = multistep_if_ptt(
        x_seq, v_init, 1.0, 0.0, False, sg
    )
    (spike_seq.mean() + v_seq.mean()).backward()

    assert spike_seq.shape == (4, 64)
    assert torch.all((spike_seq == 0) | (spike_seq == 1))
    assert x_seq.grad is not None


@pytest.mark.parametrize("env_value", ["0", "false", "off", "no", "False", "OFF"])
def test_env_flag_disabled_values(env_value):
    """All 'disabled' string values for SJ_USE_CUPY_OP should disable the flag."""
    old = os.environ.get("SJ_USE_CUPY_OP")
    os.environ["SJ_USE_CUPY_OP"] = env_value
    try:
        from spikingjelly.activation_based.cuda_kernel.cuda_utils import env_flag_enabled
        assert env_flag_enabled("SJ_USE_CUPY_OP") is False, (
            f"env_flag_enabled should be False for value={env_value!r}"
        )
    finally:
        if old is None:
            os.environ.pop("SJ_USE_CUPY_OP", None)
        else:
            os.environ["SJ_USE_CUPY_OP"] = old


def test_spike_conv2d_cuda_backward_with_bias():
    """spike_conv2d on CUDA with bias: gradients flow through correctly."""
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    spike = (torch.rand(2, 3, 8, 8, device="cuda") > 0.5).float().requires_grad_(True)
    weight = torch.randn(4, 3, 3, 3, device="cuda", requires_grad=True)
    bias = torch.randn(4, device="cuda", requires_grad=True)

    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv2d
    out = spike_conv2d(spike, weight, bias, padding=1)
    out.sum().backward()

    assert out.shape == (2, 4, 8, 8)
    assert weight.grad is not None
    assert bias.grad is not None


def test_spike_conv3d_cuda_no_bias():
    """spike_conv3d on CUDA runs forward and backward."""
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    spike = (torch.rand(1, 2, 4, 4, 4, device="cuda") > 0.5).float().requires_grad_(True)
    weight = torch.randn(2, 2, 3, 3, 3, device="cuda", requires_grad=True)

    from spikingjelly.activation_based.cuda_kernel.spike_op import spike_conv3d
    out = spike_conv3d(spike, weight, padding=1)
    out.sum().backward()

    assert out.shape == (1, 2, 4, 4, 4)
    assert weight.grad is not None
