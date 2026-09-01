import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.cuda_kernel.neuron_linear import (
    if_linear,
    lif_linear,
)


try:
    __import__("cupy")
    _HAS_CUPY = True
except (ImportError, OSError):
    _HAS_CUPY = False


pytestmark = pytest.mark.skipif(
    not _HAS_CUPY or not torch.cuda.is_available(),
    reason="requires cupy and CUDA",
)


def _reference(
    x,
    v,
    weight,
    bias,
    *,
    tau,
    decay_input,
    v_threshold,
    v_reset,
    detach_reset,
    sg,
):
    single_step = x.dim() == 2
    x_seq = x.unsqueeze(0) if single_step else x
    outputs = []
    for x_t in x_seq:
        if tau is None:
            spike, v = functional.if_step(
                x_t, v, v_threshold, v_reset, sg, detach_reset
            )
        else:
            spike, v = functional.lif_step(
                x_t,
                v,
                tau,
                decay_input,
                v_threshold,
                v_reset,
                sg,
                detach_reset,
            )
        outputs.append(F.linear(spike, weight, bias))
    y = torch.stack(outputs)
    return (y[0] if single_step else y), v


@pytest.mark.parametrize(
    "shape,decay_input,v_reset,detach_reset",
    [
        ((3, 130), True, 0.0, False),
        ((3, 130), False, None, True),
        ((4, 3, 130), False, 0.25, True),
    ],
)
def test_lif_linear_forward_backward(shape, decay_input, v_reset, detach_reset):
    torch.manual_seed(0)
    K, N = shape[-1], 70
    sg = surrogate.ATan()
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    v = torch.randn(shape[-2], K, device="cuda", requires_grad=True)
    weight = torch.randn(N, K, device="cuda", requires_grad=True)
    bias = torch.randn(N, device="cuda", requires_grad=True)
    refs = [z.detach().clone().requires_grad_() for z in (x, v, weight, bias)]
    kwargs = {
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.75,
        "v_reset": v_reset,
        "detach_reset": detach_reset,
    }

    y, v_next = lif_linear(
        x,
        v,
        weight.t().contiguous(),
        bias,
        threads=128,
        surrogate_function=sg,
        **kwargs,
    )
    y_ref, v_ref = _reference(*refs, sg=sg, **kwargs)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(v_next, v_ref, rtol=1e-5, atol=1e-6)

    grad_y = torch.randn_like(y)
    grad_v = torch.randn_like(v_next)
    torch.autograd.backward((y, v_next), (grad_y, grad_v))
    torch.autograd.backward((y_ref, v_ref), (grad_y, grad_v))
    for actual, expected in zip((x, v, weight, bias), refs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-5)


def test_if_linear_forward_backward():
    torch.manual_seed(2)
    shape, v_reset = (4, 3, 130), None
    K, N = shape[-1], 70
    sg = surrogate.ATan()
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    v = torch.randn(shape[-2], K, device="cuda", requires_grad=True)
    weight = torch.randn(N, K, device="cuda", requires_grad=True)
    bias = torch.randn(N, device="cuda", requires_grad=True)
    refs = [z.detach().clone().requires_grad_() for z in (x, v, weight, bias)]
    kwargs = {
        "v_threshold": 0.75,
        "v_reset": v_reset,
        "detach_reset": True,
    }

    y, v_next = if_linear(
        x,
        v,
        weight.t().contiguous(),
        bias,
        threads=128,
        surrogate_function=sg,
        **kwargs,
    )
    y_ref, v_ref = _reference(
        *refs,
        tau=None,
        decay_input=True,
        sg=sg,
        **kwargs,
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(v_next, v_ref, rtol=1e-5, atol=1e-6)

    grad_y = torch.randn_like(y)
    grad_v = torch.randn_like(v_next)
    torch.autograd.backward((y, v_next), (grad_y, grad_v))
    torch.autograd.backward((y_ref, v_ref), (grad_y, grad_v))
    for actual, expected in zip((x, v, weight, bias), refs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-5)


def test_lif_linear_without_bias_and_nondefault_stream():
    torch.manual_seed(1)
    x = torch.randn(2, 5, 64, device="cuda", requires_grad=True)
    v = torch.randn(5, 64, device="cuda", requires_grad=True)
    weight = torch.randn(33, 64, device="cuda", requires_grad=True)
    refs = [z.detach().clone().requires_grad_() for z in (x, v, weight)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y, v_next = lif_linear(x, v, weight.t().contiguous(), threads=256)
        y_ref, v_ref = _reference(
            *refs,
            None,
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            detach_reset=False,
            sg=surrogate.Sigmoid(),
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(v_next, v_ref)
    grad_y, grad_v = torch.randn_like(y), torch.randn_like(v_next)
    torch.autograd.backward((y, v_next), (grad_y, grad_v))
    torch.autograd.backward((y_ref, v_ref), (grad_y, grad_v))
    for actual, expected in zip((x, v, weight), refs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-5)


def test_lif_linear_backward_recomputes_forward_spikes():
    torch.manual_seed(3)
    K, tau, threshold, v_reset = 130, 3.1, 0.75, 0.25
    v = torch.randn(1, K, device="cuda") * 8
    x = (threshold - v) * tau + (v - v_reset)
    weight_t = torch.eye(K, device="cuda", requires_grad=True)
    y, _ = lif_linear(
        x,
        v,
        weight_t,
        tau=tau,
        v_threshold=threshold,
        v_reset=v_reset,
        threads=128,
    )
    y.sum().backward()
    torch.testing.assert_close(weight_t.grad[:, 0], y[0], rtol=0, atol=0)


@pytest.mark.parametrize("fused_op", [if_linear, lif_linear], ids=["if", "lif"])
def test_neuron_linear_fake_and_compile(fused_op):
    x_meta = torch.empty(4, 3, 16, device="meta")
    v_meta = torch.empty(3, 16, device="meta")
    weight_t_meta = torch.empty(16, 8, device="meta")
    y, v_next = fused_op(x_meta, v_meta, weight_t_meta, threads=128)
    assert y.shape == (4, 3, 8)
    assert v_next.shape == v_meta.shape
    y, v_next = fused_op(x_meta[0], v_meta, weight_t_meta, threads=128)
    assert y.shape == (3, 8)
    assert v_next.shape == v_meta.shape

    x = torch.randn(2, 3, 16, device="cuda")
    v = torch.randn(3, 16, device="cuda")
    weight_t = torch.randn(16, 8, device="cuda")

    def f(x, v, weight_t):
        return fused_op(x, v, weight_t, threads=128)

    expected = f(x, v, weight_t)
    actual = torch.compile(f, fullgraph=True)(x, v, weight_t)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
