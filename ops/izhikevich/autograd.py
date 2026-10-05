import math
from typing import Optional

import torch
from torch.autograd.function import once_differentiable

from ..validation import _check_gradients, _check_inputs


def _check(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
):
    _check_inputs(x, v, threshold, reset, alpha, surrogate_id)
    if not all(
        math.isfinite(p)
        for p in (
            tau,
            rest,
            critical,
            a0,
            a,
            b,
            tau_w,
        )
    ):
        raise ValueError("izhikevich parameters must be finite")
    if tau <= 1:
        raise ValueError("tau must exceed one")
    if tau_w <= 0:
        raise ValueError("tau_w must be positive")
    torch._check(
        w.shape == v.shape
        and w.dtype == torch.float32
        and w.device == v.device
        and w.layout == torch.strided,
        lambda: "recovery state must match v shape/device and be strided FP32",
    )


def _forward_fake(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
):
    _check(
        x,
        v,
        w,
        tau,
        rest,
        critical,
        a0,
        a,
        b,
        tau_w,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(
            x if store_v_seq else v,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        ),
        torch.empty_like(
            x if store_v_seq else v,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        ),
        torch.empty_like(x, dtype=torch.float32, memory_format=torch.contiguous_format),
        torch.empty_like(x, dtype=torch.float32, memory_format=torch.contiguous_format),
    )


def _check_backward(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gw: torch.Tensor,
    h: torch.Tensor,
    previous: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
):
    _check(
        h,
        h[0],
        h[0],
        tau,
        rest,
        critical,
        a0,
        a,
        b,
        tau_w,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    _check_gradients(gs, gv, h, threshold, reset, alpha, store_v_seq, surrogate_id)
    torch._check(
        previous.shape == h.shape
        and previous.dtype == torch.float32
        and previous.device == h.device
        and previous.layout == torch.strided,
        lambda: "invalid previous-voltage workspace",
    )
    _check_gradients(gs, gw, h, threshold, reset, alpha, store_v_seq, surrogate_id)


def _backward_fake(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gw: torch.Tensor,
    h: torch.Tensor,
    previous: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
):
    _check_backward(
        gs,
        gv,
        gw,
        h,
        previous,
        tau,
        rest,
        critical,
        a0,
        a,
        b,
        tau_w,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    return (
        torch.empty_like(gs, memory_format=torch.contiguous_format),
        torch.empty_like(h[0], memory_format=torch.contiguous_format),
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
        ctx.parameters = inputs[3:]
        ctx.save_for_backward(*output[3:])
        ctx.mark_non_differentiable(*output[3:])
        ctx.set_materialize_grads(False)

    @once_differentiable
    def backward(ctx, gs, gv, gw, gh, gp):
        h, previous = ctx.saved_tensors
        state_output = h if ctx.parameters[-2] else h[0]
        if gs is None:
            gs = torch.zeros_like(h, dtype=ctx.dtype)
        if gv is None:
            gv = torch.zeros_like(state_output)
        if gw is None:
            gw = torch.zeros_like(state_output)
        gx, v0, w0 = backward_op(gs, gv, gw, h, previous, *ctx.parameters)
        return gx, v0, w0, *(None for _ in ctx.parameters)

    torch.library.register_autograd(forward_name, backward, setup_context=setup_context)
