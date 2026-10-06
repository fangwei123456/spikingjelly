from functools import cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from spikingjelly.activation_based import surrogate
from spikingjelly.logger import logger

from .. import cuda_runtime as cuda_utils
from ..surrogate import _surrogate_spec

try:
    import cupy
    from cupy import RawModule
except (ImportError, OSError) as e:
    logger.info("Optional CuPy dependency unavailable: {}", e)
    cupy = None


_OUTPUTS_PER_THREAD = 4


_MAX_CUDA_ELEMENTS = 2**31 - 1


@cache
def _get_kernels(device: int, soft_reset: bool):
    source = Path(__file__).with_name("kernels.cuh")
    options = (
        "--std=c++17",
        "--use_fast_math",
        f"-I{source.parent.parent}",
        f"-DSOFT_RESET={int(soft_reset)}",
    )
    with cuda_utils.DeviceEnvironment(device):
        module = RawModule(
            code=source.read_text(encoding="utf-8"),
            options=options,
            name_expressions=tuple(f"neuron_backward<{i}>" for i in range(-1, 7)),
        )
        return (
            module.get_function("if_linear_kernel"),
            module.get_function("rematerialize"),
            tuple(module.get_function(f"neuron_backward<{i}>") for i in range(-1, 7)),
        )


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
    if cupy is None:
        raise RuntimeError("cupy is required for fused CUDA neuron-Linear")
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


def _launch(
    kernel,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    dimensions: tuple[int, int, int, int, int, int],
    kernel_parameters: tuple,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, M, K, N, device, shared_bytes = dimensions
    y_seq = torch.empty((T, M, N), device=x_seq.device, dtype=torch.float32)
    v_out = torch.empty_like(v_init)
    bias_ptr = y_seq.data_ptr() if bias is None else bias.data_ptr()
    n_per_block = threads * _OUTPUTS_PER_THREAD
    n_groups = (N + n_per_block - 1) // n_per_block
    blocks = M * n_groups
    args = (
        x_seq.data_ptr(),
        v_init.data_ptr(),
        weight_t.data_ptr(),
        bias_ptr,
        y_seq.data_ptr(),
        v_out.data_ptr(),
        np.int32(T),
        np.int32(M),
        np.int32(K),
        np.int32(N),
        *kernel_parameters,
        np.int32(bias is not None),
    )
    with cuda_utils.DeviceEnvironment(device):
        stream = cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        kernel(
            (blocks,),
            (threads,),
            args,
            shared_mem=shared_bytes,
            stream=stream,
        )
    return y_seq, v_out


@torch.library.custom_op(
    "sj_if_linear::cupy_if_linear_forward", mutates_args=(), device_types="cuda"
)
def cupy_if_linear_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    v_threshold: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    surrogate_id: int,
    alpha: float,
    surrogate_handle: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = _check_forward_inputs(x_seq, v_init, weight_t, bias, threads)
    kernel = _get_kernels(dimensions[4], soft_reset)[0]
    return _launch(
        kernel,
        x_seq,
        v_init,
        weight_t,
        bias,
        dimensions,
        (np.float32(v_threshold), np.float32(v_reset)),
        threads,
    )


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


@torch.library.register_fake("sj_if_linear::cupy_if_linear_forward")
def _cupy_if_linear_forward_fake(
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
    threads,
):
    return _fake_outputs(x_seq, v_init, weight_t)


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
        cuda_utils.resolve_python_object(surrogate_handle) if surrogate_id < 0 else None
    )
    ctx.surrogate_handle = surrogate_handle


def _setup_if_context(ctx, inputs, output):
    del output
    (
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


@torch.library.custom_op(
    "sj_if_linear::cupy_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor x_seq, Tensor v_init, Tensor weight_t, Tensor? bias, "
        "Tensor grad_y, Tensor grad_v_out, "
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
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_id: int,
    alpha: float,
    surrogate_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T, M, K = x_seq.shape
    spikes, charged = torch.empty_like(x_seq), torch.empty_like(x_seq)
    grad_x, grad_v = torch.empty_like(x_seq), torch.empty_like(v_init)
    parameters = (
        np.float32(v_threshold),
        np.float32(0.0 if v_reset is None else v_reset),
    )
    with cuda_utils.DeviceEnvironment(x_seq.get_device()):
        _, rematerialize, backwards = _get_kernels(x_seq.get_device(), v_reset is None)
        grid, block = (min((M * K + 255) // 256, 65535),), (256,)
        rematerialize(
            grid,
            block,
            (
                *(t.data_ptr() for t in (x_seq, v_init, spikes, charged)),
                np.int32(T),
                np.int32(M * K),
                *parameters,
            ),
        )
        # Custom surrogates supply their PyTorch derivative as a tensor. Built-in
        # surrogates evaluate the same explicit CUDA formula inside the kernel.
        sg_id = surrogate_id
        if sg_id < 0:
            surrogate_function = cuda_utils.resolve_python_object(surrogate_handle)
            with torch.enable_grad():
                over_threshold = (charged - v_threshold).requires_grad_()
                output = surrogate_function(over_threshold)
                sg = torch.autograd.grad(
                    output, over_threshold, torch.ones_like(output)
                )[0]
        else:
            sg = charged
        grad_spike = torch.matmul(grad_y, weight_t.t()).contiguous()
        grad_v_out, sg = grad_v_out.contiguous(), sg.contiguous()
        backwards[sg_id + 1](
            grid,
            block,
            (
                *(
                    t.data_ptr()
                    for t in (grad_spike, grad_v_out, charged, sg, grad_x, grad_v)
                ),
                np.int32(T),
                np.int32(M * K),
                *parameters,
                np.int32(detach_reset),
                np.float32(alpha),
            ),
        )
    N = weight_t.shape[1]
    grad_w = torch.mm(spikes.reshape(-1, K).t(), grad_y.reshape(-1, N))
    grad_b = grad_y.reshape(-1, N).sum(0) if bias is not None else None
    return grad_x, grad_v, grad_w, grad_b


@torch.library.register_fake("sj_if_linear::cupy_backward")
def _backward_fake(
    x_seq,
    v_init,
    weight_t,
    bias,
    grad_y,
    grad_v_out,
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
def _if_backward(ctx, grad_y, grad_v_out):
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
            ctx.v_threshold,
            ctx.v_reset,
            ctx.detach_reset,
            ctx.surrogate_id,
            ctx.alpha,
            ctx.surrogate_handle,
        )
        + (None,) * 8
    )


torch.library.register_autograd(
    "sj_if_linear::cupy_if_linear_forward",
    _if_backward,
    setup_context=_setup_if_context,
)


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
        cuda_utils.register_python_object(surrogate_function)
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


def if_linear(
    x: torch.Tensor,
    v: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    detach_reset: bool = False,
    surrogate_function=surrogate.Sigmoid(),
    threads: Literal[128, 256, 512] = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused single- or multi-step IF followed by Linear.

    ``x`` is ``[M, K]`` or ``[T, M, K]``; ``weight_t`` is contiguous ``[K, N]``.
    Cache a contiguous ``weight_t`` to avoid copying it on every call.
    Backward rematerializes spikes and supports first-order gradients only.
    """
    x_seq, v, weight_t, bias, surrogate_id, alpha, surrogate_handle, single_step = (
        _prepare_inputs(x, v, weight_t, bias, surrogate_function)
    )
    y_seq, v_out = cupy_if_linear_forward(
        x_seq,
        v,
        weight_t,
        bias,
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
