import torch
from torch.autograd.function import once_differentiable

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
):
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    return tuple(
        torch.empty_like(t, dtype=dtype, memory_format=torch.contiguous_format)
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
):
    _check_backward(
        gs, gv, x, v, w, h, threshold, reset, alpha, store_v_seq, surrogate_id
    )
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(w),
    )


def _setup_context(ctx, inputs, output):
    ctx.input_dtype = inputs[0].dtype
    ctx.parameters = inputs[3:]
    ctx.save_for_backward(*inputs[:3], output[2])
    ctx.mark_non_differentiable(output[2])
    ctx.set_materialize_grads(False)


def _register_ops(forward_name: str, backward_name: str, *, register_fake: bool = True):
    if register_fake:
        torch.library.register_fake(forward_name, _forward_fake)
        torch.library.register_fake(backward_name, _backward_fake)
    namespace, name = backward_name.split("::")
    backward_op = getattr(getattr(torch.ops, namespace), name).default

    @once_differentiable
    def backward(ctx, gs, gv, gh):
        x, v, w, h = ctx.saved_tensors
        if gs is None:
            gs = torch.zeros_like(h, dtype=ctx.input_dtype)
        if gv is None:
            gv = torch.zeros_like(h if ctx.parameters[-2] else h[0])
        gx, gv_init, gw = backward_op(gs, gv, x, v, w, h, *ctx.parameters)
        return gx, gv_init, gw, None, None, None, None, None, None, None

    torch.library.register_autograd(
        forward_name, backward, setup_context=_setup_context
    )
