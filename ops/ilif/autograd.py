import math

import torch

from ..autograd import _higher_order_grad, _save_for_higher_order
from ..validation import _check_gradients, _check_inputs


def _check(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
):
    _check_inputs(x, v, threshold, None, 4.0)
    if not all(
        math.isfinite(p)
        for p in (
            tau,
            count,
            lower,
            upper,
        )
    ):
        raise ValueError("ilif parameters must be finite")
    if tau <= 1:
        raise ValueError("tau must exceed one")
    if threshold <= 0 or count < 1 or count != int(count) or lower > upper:
        raise ValueError(
            "I-LIF requires a positive threshold, integer count >= 1, and ordered STE window"
        )


def _forward_fake(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
):
    _check(x, v, tau, count, lower, upper, threshold, detach_reset, store_v_seq)
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(
            x if store_v_seq else v,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        ),
        torch.empty_like(x, dtype=torch.float32, memory_format=torch.contiguous_format),
    )


def _check_backward(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
):
    _check(h, h[0], tau, count, lower, upper, threshold, detach_reset, store_v_seq)
    _check_gradients(gs, gv, h, threshold, None, 4.0, store_v_seq)


def _backward_fake(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
):
    _check_backward(
        gs, gv, h, tau, count, lower, upper, threshold, detach_reset, store_v_seq
    )
    return (
        torch.empty_like(gs, memory_format=torch.contiguous_format),
        torch.empty_like(h[0], memory_format=torch.contiguous_format),
    )


def _register_ops(forward_name: str, backward_name: str, *, register_fake=True):
    namespace, opname = backward_name.split("::")
    backward_op = getattr(getattr(torch.ops, namespace), opname).default
    if register_fake:
        torch.library.register_fake(forward_name, _forward_fake)
        torch.library.register_fake(backward_name, _backward_fake)

    def setup_context(ctx, inputs, output):
        ctx.dtype = inputs[0].dtype
        ctx.parameters = inputs[2:]
        _save_for_higher_order(ctx, inputs, output[2:])
        ctx.mark_non_differentiable(*output[2:])
        ctx.set_materialize_grads(False)

    def backward(ctx, gs, gv, gh):
        if torch.is_grad_enabled():
            from .reference import _forward_impl

            return _higher_order_grad(ctx, _forward_impl, (gs, gv, gh))
        h = ctx.saved_tensors[0]
        state_output = h if ctx.parameters[-1] else h[0]
        if gs is None:
            gs = torch.zeros_like(h, dtype=ctx.dtype)
        if gv is None:
            gv = torch.zeros_like(state_output)
        gx, v0 = backward_op(gs, gv, h, *ctx.parameters)
        return gx, v0, *(None for _ in ctx.parameters)

    torch.library.register_autograd(forward_name, backward, setup_context=setup_context)
