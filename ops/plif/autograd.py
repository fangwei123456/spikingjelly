from functools import partial

import torch

from ..autograd import _higher_order_grad, _save_for_higher_order
from ..layout import _fake_empty_like
from ..surrogate import _DTYPES
from ..validation import _check_gradients, _check_inputs


def _check_forward(x, v, w, threshold, reset, alpha, surrogate_id: int = 0):
    _check_inputs(x, v, threshold, reset, alpha, surrogate_id)
    torch._check(w.ndim == 0, lambda: "w must be a scalar tensor")
    torch._check(
        w.dtype in _DTYPES, lambda: "w must have dtype float32, float16 or bfloat16"
    )
    torch._check(w.device == x.device, lambda: "w and input devices must match")
    torch._check(w.layout == torch.strided, lambda: "w must have strided layout")


def _check_backward(
    gs,
    gv,
    x,
    v,
    w,
    h,
    threshold,
    reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
):
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    torch._check(gs.dtype == x.dtype, lambda: "spike gradient dtype must match input")
    _check_gradients(gs, gv, h, threshold, reset, alpha, store_v_seq, surrogate_id)
    torch._check(h.shape == x.shape, lambda: "workspace shape must match input")
    torch._check(h.device == x.device, lambda: "workspace device must match input")


def _forward_fake(
    x,
    v,
    w,
    decay_input,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _strided=False,
):
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    return tuple(
        _fake_empty_like(t, dtype=dtype, strided=_strided)
        for t, dtype in (
            (x, x.dtype),
            (x if store_v_seq else v, torch.float32),
            (x, torch.float32),
        )
    )


def _backward_fake(
    gs,
    gv,
    x,
    v,
    w,
    h,
    decay_input,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
    *,
    _strided=False,
):
    _check_backward(
        gs, gv, x, v, w, h, threshold, reset, alpha, store_v_seq, surrogate_id
    )
    return (
        _fake_empty_like(x, strided=_strided),
        _fake_empty_like(v, strided=_strided),
        torch.empty_like(w),
    )


def _setup_context(ctx, inputs, output):
    ctx.input_dtype = inputs[0].dtype
    ctx.parameters = inputs[3:]
    _save_for_higher_order(ctx, inputs, (output[2],))
    ctx.mark_non_differentiable(output[2])
    ctx.set_materialize_grads(False)


def _register_ops(forward_name: str, backward_name: str, *, register_fake: bool = True):
    if register_fake:
        torch.library.register_fake(forward_name, partial(_forward_fake, _strided=True))
        torch.library.register_fake(
            backward_name, partial(_backward_fake, _strided=True)
        )
    namespace, name = backward_name.split("::")
    backward_op = getattr(getattr(torch.ops, namespace), name).default

    def backward(ctx, gs, gv, gh):
        if torch.is_grad_enabled():
            from .reference import _forward_impl

            return _higher_order_grad(ctx, _forward_impl, (gs, gv, gh))
        h, x, v, w = ctx.saved_tensors
        if gs is None:
            gs = torch.zeros_like(h, dtype=ctx.input_dtype)
        if gv is None:
            gv = torch.zeros_like(h if ctx.parameters[-2] else h[0])
        gx, gv_init, gw = backward_op(gs, gv, x, v, w, h, *ctx.parameters)
        return gx, gv_init, gw, None, None, None, None, None, None, None

    torch.library.register_autograd(
        forward_name, backward, setup_context=_setup_context
    )
