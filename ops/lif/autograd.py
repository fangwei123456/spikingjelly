import math

import torch
from torch.autograd.function import once_differentiable

from ..validation import _check_gradients, _check_inputs


def _check_forward(x, v, tau, threshold, reset, alpha, surrogate_id: int = 0):
    _check_inputs(x, v, threshold, reset, alpha, surrogate_id)
    if not math.isfinite(tau) or tau <= 1:
        raise ValueError("tau must be finite and greater than one")


def _check_backward(
    gs,
    gv,
    h,
    tau,
    threshold,
    reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
):
    _check_gradients(gs, gv, h, threshold, reset, alpha, store_v_seq, surrogate_id)
    if not math.isfinite(tau) or tau <= 1:
        raise ValueError("tau must be finite and greater than one")


def _forward_fake(
    x,
    v,
    tau,
    decay_input,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
):
    _check_forward(x, v, tau, threshold, reset, alpha, surrogate_id)
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
    h,
    tau,
    decay_input,
    threshold,
    reset,
    detach_reset,
    alpha,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
):
    _check_backward(gs, gv, h, tau, threshold, reset, alpha, store_v_seq, surrogate_id)
    return torch.empty_like(
        h, dtype=gs.dtype, memory_format=torch.contiguous_format
    ), torch.empty_like(h[0], memory_format=torch.contiguous_format)


def _setup_context(ctx, inputs, output):
    ctx.input_dtype = inputs[0].dtype
    ctx.parameters = inputs[2:]
    ctx.save_for_backward(output[2])
    ctx.mark_non_differentiable(output[2])
    ctx.set_materialize_grads(False)


def _register_ops(forward_name: str, backward_name: str, *, register_fake: bool = True):
    if register_fake:
        torch.library.register_fake(forward_name, _forward_fake)
        torch.library.register_fake(backward_name, _backward_fake)
    namespace, opname = backward_name.split("::")
    backward_op = getattr(getattr(torch.ops, namespace), opname).default

    @once_differentiable
    def backward(ctx, gs, gv, gh):
        (h,) = ctx.saved_tensors
        if gs is None:
            gs = torch.zeros_like(h, dtype=ctx.input_dtype)
        if gv is None:
            gv = torch.zeros_like(h if ctx.parameters[-2] else h[0])
        gx, gv_init = backward_op(gs, gv, h, *ctx.parameters)
        return gx, gv_init, None, None, None, None, None, None, None, None

    torch.library.register_autograd(
        forward_name, backward, setup_context=_setup_context
    )
