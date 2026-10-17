import math

import torch

from ..surrogate import _DTYPES


def _check(x, tensors, tau, threshold, alpha, surrogate_id):
    for state in tensors:
        torch._check(
            state.shape == x.shape, lambda: "CLIF tensor shapes must match input"
        )
        torch._check(state.device == x.device, lambda: "CLIF devices must match")
        torch._check(state.dtype == x.dtype, lambda: "CLIF dtypes must match")
    torch._check(x.dtype in _DTYPES, lambda: "CLIF requires FP32, FP16 or BF16")
    if not math.isfinite(tau) or tau <= 1:
        raise ValueError("CLIF tau must be finite and greater than one")
    if not math.isfinite(threshold) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("CLIF threshold must be finite and alpha finite and positive")
    if surrogate_id not in range(9):
        raise ValueError("CLIF surrogate_id must be in [0, 8]")


def _forward_fake(
    x, v, leak_v, m, tau, threshold, alpha, surrogate_id, *, _strided=False
):
    _check(x, (v, leak_v, m), tau, threshold, alpha, surrogate_id)
    return tuple(
        torch.empty_like(x, memory_format=torch.contiguous_format) for _ in range(4)
    )


def _backward_fake(
    gs, gv, gm, h, m, m_next, tau, threshold, alpha, surrogate_id, *, _strided=False
):
    _check(h, (gs, gv, gm, m, m_next), tau, threshold, alpha, surrogate_id)
    return tuple(
        torch.empty_like(h, memory_format=torch.contiguous_format) for _ in range(3)
    )


def _setup_context(ctx, inputs, output):
    ctx.parameters = inputs[4:]
    ctx.save_for_backward(output[3], output[2], *inputs[:4])
    ctx.mark_non_differentiable(
        *((output[0], output[3]) if inputs[-1] == 8 else (output[3],))
    )
    ctx.set_materialize_grads(False)


def _register_ops(forward_name: str, backward_name: str, *, register_fake: bool = True):
    if register_fake:
        torch.library.register_fake(forward_name, _forward_fake)
        torch.library.register_fake(backward_name, _backward_fake)
    namespace, opname = backward_name.split("::")
    backward_op = getattr(getattr(torch.ops, namespace), opname).default

    def backward(ctx, gs, gv, gm, gh):
        if torch.is_grad_enabled():
            from .reference import _forward_impl

            # Separate differentiable copies preserve partials for repeated v inputs.
            inputs = tuple(t.clone() for t in ctx.saved_tensors[2:])
            outputs = _forward_impl(*inputs, *ctx.parameters)
            active = [
                (y, g) for y, g in zip(outputs, (gs, gv, gm, gh)) if g is not None
            ]
            targets = tuple(t for t in inputs if t.requires_grad)
            gradients = iter(
                torch.autograd.grad(
                    tuple(y for y, _ in active),
                    targets,
                    tuple(g for _, g in active),
                    create_graph=True,
                    allow_unused=True,
                )
            )
            return (
                tuple(next(gradients) if t.requires_grad else None for t in inputs)
                + (None,) * 4
            )
        h, m_next, _, _, _, m = ctx.saved_tensors
        gs, gv, gm = (
            torch.zeros_like(h) if grad is None else grad for grad in (gs, gv, gm)
        )
        gx, gv_leak, gm = backward_op(gs, gv, gm, h, m, m_next, *ctx.parameters)
        return gx, gx, gv_leak, gm, None, None, None, None

    torch.library.register_autograd(
        forward_name, backward, setup_context=_setup_context
    )
