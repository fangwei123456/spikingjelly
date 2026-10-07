from typing import Literal, Optional

import torch
from torch.autograd.function import once_differentiable

from spikingjelly.activation_based import surrogate

from .. import surrogate_runtime as surrogate_objects
from ..native_loader import _native_available
from ..projection import (
    _check_forward_inputs,
    _fake_outputs,
    _neuron_backward,
    _prepare_inputs,
    _rematerialize,
    _save_context,
)


@torch.library.custom_op("sj_lif_linear::forward", mutates_args=(), device_types="cuda")
def _forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    surrogate_id: int,
    alpha: float,
    surrogate_handle: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tau <= 1:
        raise ValueError("tau must be greater than one")
    dimensions = _check_forward_inputs(x_seq, v_init, weight_t, bias, threads)
    if _native_available(__package__, dimensions[4]):
        return torch.ops.sj_lif_linear.kernel_forward(
            x_seq,
            v_init,
            weight_t,
            bias,
            tau,
            decay_input,
            v_threshold,
            v_reset,
            soft_reset,
            threads,
        )
    spikes, _, final = _rematerialize(
        x_seq,
        v_init,
        v_threshold,
        None if soft_reset else v_reset,
        tau=tau,
        decay_input=decay_input,
    )
    output = spikes @ weight_t
    if bias is not None:
        output = output + bias
    return output, final


@torch.library.register_fake("sj_lif_linear::forward")
def _forward_fake(
    x_seq,
    v_init,
    weight_t,
    bias,
    tau,
    decay_input,
    v_threshold,
    v_reset,
    soft_reset,
    detach_reset,
    surrogate_id,
    alpha,
    surrogate_handle,
    threads,
):
    return _fake_outputs(x_seq, v_init, weight_t)


def _setup_lif_context(ctx, inputs, output):
    del output
    (
        x_seq,
        v_init,
        weight_t,
        bias,
        tau,
        decay_input,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        surrogate_id,
        alpha,
        surrogate_handle,
        _,
    ) = inputs
    _save_context(
        ctx,
        x_seq,
        v_init,
        weight_t,
        bias,
        v_threshold,
        v_reset,
        soft_reset,
        detach_reset,
        surrogate_id,
        alpha,
        surrogate_handle,
    )
    ctx.tau = tau
    ctx.decay_input = decay_input


@torch.library.custom_op(
    "sj_lif_linear::backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor x_seq, Tensor v_init, Tensor weight_t, Tensor? bias, "
        "Tensor grad_y, Tensor grad_v_out, "
        "float tau, bool decay_input, "
        "float v_threshold, float? v_reset, bool detach_reset, "
        "int surrogate_id, float alpha, int surrogate_handle) -> (Tensor, Tensor, Tensor, Tensor?)"
    ),
)
def _backward_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_y: torch.Tensor,
    grad_v_out: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_id: int,
    alpha: float,
    surrogate_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T, M, K = x_seq.shape
    native = _native_available(__package__, x_seq.get_device())
    if native:
        spikes, charged = torch.ops.sj_lif_linear.rematerialize(
            x_seq,
            v_init,
            tau,
            decay_input,
            v_threshold,
            0.0 if v_reset is None else v_reset,
            v_reset is None,
        )
    else:
        spikes, charged, _ = _rematerialize(
            x_seq, v_init, v_threshold, v_reset, tau=tau, decay_input=decay_input
        )
    sg_id = surrogate_id
    if sg_id < 0:
        surrogate_function = surrogate_objects.resolve_python_object(surrogate_handle)
        with torch.enable_grad():
            over_threshold = (charged - v_threshold).requires_grad_()
            output = surrogate_function(over_threshold)
            sg = torch.autograd.grad(output, over_threshold, torch.ones_like(output))[0]
    else:
        sg = charged
    grad_spike = torch.matmul(grad_y, weight_t.t()).contiguous()
    grad_v_out, sg = grad_v_out.contiguous(), sg.contiguous()
    if native:
        grad_x, grad_v = torch.ops.sj_lif_linear.neuron_backward(
            grad_spike,
            grad_v_out,
            charged,
            sg,
            tau,
            decay_input,
            v_threshold,
            0.0 if v_reset is None else v_reset,
            v_reset is None,
            detach_reset,
            sg_id,
            alpha,
        )
    else:
        grad_x, grad_v = _neuron_backward(
            grad_spike,
            grad_v_out,
            charged,
            sg,
            v_threshold,
            v_reset,
            detach_reset,
            alpha,
            sg_id,
            tau=tau,
            decay_input=decay_input,
        )
    N = weight_t.shape[1]
    grad_w = torch.mm(spikes.reshape(-1, K).t(), grad_y.reshape(-1, N))
    grad_b = grad_y.reshape(-1, N).sum(0) if bias is not None else None
    return grad_x, grad_v, grad_w, grad_b


@torch.library.register_fake("sj_lif_linear::backward")
def _backward_fake(
    x_seq,
    v_init,
    weight_t,
    bias,
    grad_y,
    grad_v_out,
    tau,
    decay_input,
    v_threshold,
    v_reset,
    detach_reset,
    surrogate_id,
    alpha,
    surrogate_handle,
):
    return (
        torch.empty_like(x_seq),
        torch.empty_like(v_init),
        torch.empty_like(weight_t),
        None if bias is None else torch.empty_like(bias),
    )


@once_differentiable
def _lif_backward(ctx, grad_y, grad_v_out):
    x_seq, v_init, weight_t, bias = ctx.saved_tensors
    if grad_y is None:
        grad_y = x_seq.new_zeros((*x_seq.shape[:2], weight_t.shape[1]))
    if grad_v_out is None:
        grad_v_out = torch.zeros_like(v_init)
    return (
        _backward_kernel(
            x_seq,
            v_init,
            weight_t,
            bias,
            grad_y,
            grad_v_out,
            ctx.tau,
            ctx.decay_input,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.detach_reset,
            ctx.surrogate_id,
            ctx.alpha,
            ctx.surrogate_handle,
        )
        + (None,) * 10
    )


torch.library.register_autograd(
    "sj_lif_linear::forward",
    _lif_backward,
    setup_context=_setup_lif_context,
)


def lif_linear(
    x: torch.Tensor,
    v: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    tau: float = 2.0,
    decay_input: bool = True,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    detach_reset: bool = False,
    surrogate_function=surrogate.Sigmoid(),
    threads: Literal[128, 256, 512] = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused single- or multi-step LIF followed by Linear.

    ``x`` is ``[M, K]`` or ``[T, M, K]``; ``weight_t`` is contiguous ``[K, N]``.
    Cache a contiguous ``weight_t`` to avoid copying it on every call.
    Backward rematerializes spikes and supports first-order gradients only.
    """
    x_seq, v, weight_t, bias, surrogate_id, alpha, surrogate_handle, single_step = (
        _prepare_inputs(x, v, weight_t, bias, surrogate_function)
    )
    y_seq, v_out = _forward(
        x_seq,
        v,
        weight_t,
        bias,
        tau=tau,
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=0.0 if v_reset is None else v_reset,
        soft_reset=v_reset is None,
        detach_reset=detach_reset,
        surrogate_id=surrogate_id,
        alpha=alpha,
        surrogate_handle=surrogate_handle,
        threads=threads,
    )
    return (y_seq[0] if single_step else y_seq), v_out
