"""Fused projection inputs, autograd context, and Torch reference equations."""

from typing import Optional

import torch

from . import surrogate_runtime as surrogate_objects
from .surrogate import _surrogate_gradient, _surrogate_spec


_MAX_CUDA_ELEMENTS = 2**31 - 1


def _rematerialize(x, v, threshold, reset, *, tau=None, decay_input=True):
    spikes, charged = [], []
    voltage = v
    for current in x:
        if tau is None:
            h = voltage + current
        else:
            resting = voltage if reset is None else voltage - reset
            h = voltage + (
                (current - resting) / tau if decay_input else current - resting / tau
            )
        spike = (h >= threshold).to(x.dtype)
        voltage = (
            h - spike * threshold
            if reset is None
            else torch.where(spike.bool(), reset, h)
        )
        spikes.append(spike)
        charged.append(h)
    return torch.stack(spikes), torch.stack(charged), voltage


def _neuron_backward(
    gs,
    gv,
    h,
    sg,
    threshold,
    reset,
    detach,
    alpha,
    surrogate_id,
    *,
    tau=None,
    decay_input=True,
):
    gradients = []
    carry = gv
    for index in range(h.shape[0] - 1, -1, -1):
        value = h[index]
        derivative = (
            sg[index]
            if surrogate_id < 0
            else _surrogate_gradient(value - threshold, alpha, surrogate_id)
        )
        if reset is None:
            dr = 1 if detach else 1 - threshold * derivative
        else:
            dr = 1 - (value >= threshold).to(h.dtype)
            if not detach:
                dr = dr + (reset - value) * derivative
        gh = gs[index] * derivative + carry * dr
        gradients.append(gh / tau if tau is not None and decay_input else gh)
        carry = gh if tau is None else gh * (1 - 1 / tau)
    return torch.stack(gradients[::-1]), carry


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    ndim: int,
    device: Optional[torch.device] = None,
) -> None:
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.numel() > _MAX_CUDA_ELEMENTS:
        raise ValueError(f"{name} exceeds the CUDA kernel element limit")


def _check_forward_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    threads: int,
) -> tuple[int, int, int, int, int, int]:
    _check_tensor(x_seq, "x_seq", 3)
    _check_tensor(v_init, "v_init", 2, x_seq.device)
    _check_tensor(weight_t, "weight_t", 2, x_seq.device)
    if bias is not None:
        _check_tensor(bias, "bias", 1, x_seq.device)
    T, M, K = x_seq.shape
    N = weight_t.shape[1]
    if T <= 0 or M <= 0 or K <= 0 or N <= 0:
        raise ValueError("T, M, K, and N must be positive")
    if v_init.shape != (M, K):
        raise ValueError("v_init must have shape [M, K]")
    if weight_t.shape[0] != K:
        raise ValueError("weight_t must have shape [K, N]")
    if bias is not None and bias.shape != (N,):
        raise ValueError("bias must have shape [N]")
    if threads not in (128, 256, 512):
        raise ValueError("threads must be 128, 256, or 512")
    if T * M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    device = x_seq.get_device()
    shared_bytes = K * 4 + threads // 8
    max_shared = torch.cuda.get_device_properties(device).shared_memory_per_block
    if shared_bytes > max_shared:
        raise ValueError(
            f"K={K} requires {shared_bytes} shared bytes, limit is {max_shared}"
        )
    return T, M, K, N, device, shared_bytes


def _fake_outputs(x_seq, v_init, weight_t):
    torch._check(x_seq.dim() == 3)
    torch._check(v_init.dim() == 2)
    torch._check(weight_t.dim() == 2)
    torch._check(
        v_init.shape == x_seq.shape[1:],
        lambda: "v_init must have shape [M, K]",
    )
    torch._check(
        weight_t.shape[0] == x_seq.shape[2],
        lambda: "weight_t must have shape [K, N]",
    )
    return (
        x_seq.new_empty((x_seq.shape[0], x_seq.shape[1], weight_t.shape[1])),
        v_init.new_empty(v_init.shape),
    )


def _save_context(
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
):
    ctx.save_for_backward(x_seq, v_init, weight_t, bias)
    ctx.v_threshold = v_threshold
    ctx.v_reset = None if soft_reset else v_reset
    ctx.detach_reset = detach_reset
    ctx.surrogate_id = surrogate_id
    ctx.alpha = alpha
    # Keep custom surrogates alive until the saved autograd context is released.
    ctx.surrogate_function = (
        surrogate_objects.resolve_python_object(surrogate_handle)
        if surrogate_id < 0
        else None
    )
    ctx.surrogate_handle = surrogate_handle


def _prepare_inputs(
    x: torch.Tensor,
    v: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    surrogate_function,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    int,
    float,
    int,
    bool,
]:
    if x.dim() == 2:
        single_step = True
        x_seq = x.unsqueeze(0)
    elif x.dim() == 3:
        single_step = False
        x_seq = x
    else:
        raise ValueError("x must have shape [M, K] or [T, M, K]")
    if not getattr(surrogate_function, "spiking", True):
        raise ValueError("surrogate_function must use spiking=True")
    needs_backward = torch.is_grad_enabled() and (
        x.requires_grad
        or v.requires_grad
        or weight_t.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    spec = _surrogate_spec(surrogate_function)
    surrogate_handle = (
        surrogate_objects.register_python_object(surrogate_function)
        if spec is None and needs_backward
        else 0
    )
    surrogate_id, alpha = (-1, 0.0) if spec is None else spec
    return (
        x_seq.contiguous(),
        v.contiguous(),
        weight_t.contiguous(),
        None if bias is None else bias.contiguous(),
        surrogate_id,
        alpha,
        surrogate_handle,
        single_step,
    )
