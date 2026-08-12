"""Experimental fused CUDA IF/LIF-Linear kernel.

The kernels consume a pre-transposed ``[K, N]`` weight and never materialize
the intermediate spike tensor. They are intentionally not exported from the
``cuda_kernel`` package while their useful workload range is being measured.
"""

from functools import cache
from typing import Literal, Optional

import numpy as np
import torch

from .. import surrogate
from . import cuda_utils
from spikingjelly.logger import logger

try:
    import cupy
    from cupy import RawModule
except (ImportError, OSError) as e:
    logger.debug("Optional CuPy dependency unavailable: {}", e)
    cupy = None


__all__ = ["if_linear", "lif_linear"]


_OUTPUTS_PER_THREAD = 4


_IF_CHARGE = r"""
__device__ __forceinline__ float neuron_charge(float x, float v)
{
    return v + x;
}
"""

_LIF_CHARGE = r"""
#define DECAY_INPUT {decay_input}

__device__ __forceinline__ float neuron_charge(
    float x, float v, float r_tau, float v_reset)
{{
    float temp;
#if SOFT_RESET
    temp = v;
#else
    temp = v - v_reset;
#endif
#if DECAY_INPUT
    temp = r_tau * (x - temp);
#else
    temp = x - r_tau * temp;
#endif
    return temp + v;
}}
"""

_CUDA_TEMPLATE = r"""
#define OUTPUTS_PER_THREAD {outputs_per_thread}
#define SOFT_RESET {soft_reset}

{charge_code}

__device__ __forceinline__ float neuron_reset(
    float h, bool spike, float v_threshold, float v_reset)
{{
#if SOFT_RESET
    return h - (float)spike * v_threshold;
#else
    return spike ? v_reset : h;
#endif
}}

extern "C" __global__ void {kernel_name}(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ weight_t,
    const float* __restrict__ bias,
    float* __restrict__ y_seq,
    float* __restrict__ v_out,
    int T, int M, int K, int N,
    {charge_parameter}float v_threshold, float v_reset, int has_bias)
{{
    int n_per_block = blockDim.x * OUTPUTS_PER_THREAD;
    int n_groups = (N + n_per_block - 1) / n_per_block;
    int m = blockIdx.x / n_groups;
    int n_group = blockIdx.x % n_groups;
    int tid = threadIdx.x;

    // Output groups recompute neuron state to avoid materializing spikes.
    extern __shared__ unsigned char shared[];
    float* v = reinterpret_cast<float*>(shared);
    unsigned int* spike_masks = reinterpret_cast<unsigned int*>(
        shared + K * sizeof(float));

    for (int k = tid; k < K; k += blockDim.x)
        v[k] = v_init[m * K + k];
    __syncthreads();

    for (int t = 0; t < T; t++) {{
        float acc[OUTPUTS_PER_THREAD];
#pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {{
            int n = n_group * n_per_block + tid + q * blockDim.x;
            acc[q] = has_bias && n < N ? bias[n] : 0.0f;
        }}

        for (int k0 = 0; k0 < K; k0 += blockDim.x) {{
            int k = k0 + tid;
            bool spike = false;
            if (k < K) {{
                float h = neuron_charge(
                    x_seq[(t * M + m) * K + k], v[k]{charge_argument});
                spike = h >= v_threshold;
                v[k] = neuron_reset(h, spike, v_threshold, v_reset);
            }}
            unsigned int mask = __ballot_sync(0xffffffffu, spike);
            if ((tid & 31) == 0) spike_masks[tid >> 5] = mask;
            __syncthreads();

            int tile_warps = (min((int)blockDim.x, K - k0) + 31) >> 5;
            for (int w = 0; w < tile_warps; w++) {{
                unsigned int active = spike_masks[w];
                while (active) {{
                    int k_active = k0 + (w << 5) + __ffs(active) - 1;
#pragma unroll
                    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {{
                        int n = n_group * n_per_block + tid + q * blockDim.x;
                        if (n < N) acc[q] += weight_t[k_active * N + n];
                    }}
                    active &= active - 1;
                }}
            }}
            __syncthreads();
        }}

#pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {{
            int n = n_group * n_per_block + tid + q * blockDim.x;
            if (n < N) y_seq[(t * M + m) * N + n] = acc[q];
        }}
    }}

    if (n_group == 0) {{
        for (int k = tid; k < K; k += blockDim.x)
            v_out[m * K + k] = v[k];
    }}
}}

"""


_MAX_CUDA_ELEMENTS = 2**31 - 1


@cache
def _get_if_kernel(device: int, soft_reset: bool):
    kernel_name = "if_linear_kernel"
    source = _CUDA_TEMPLATE.format(
        outputs_per_thread=_OUTPUTS_PER_THREAD,
        soft_reset=int(soft_reset),
        charge_code=_IF_CHARGE,
        kernel_name=kernel_name,
        charge_parameter="",
        charge_argument="",
    )
    with cuda_utils.DeviceEnvironment(device):
        return RawModule(code=source, options=("--use_fast_math",)).get_function(
            kernel_name
        )


@cache
def _get_lif_kernel(device: int, decay_input: bool, soft_reset: bool):
    kernel_name = "lif_linear_kernel"
    source = _CUDA_TEMPLATE.format(
        outputs_per_thread=_OUTPUTS_PER_THREAD,
        soft_reset=int(soft_reset),
        charge_code=_LIF_CHARGE.format(decay_input=int(decay_input)),
        kernel_name=kernel_name,
        charge_parameter="float r_tau, ",
        charge_argument=", r_tau, v_reset",
    )
    with cuda_utils.DeviceEnvironment(device):
        return RawModule(code=source, options=("--use_fast_math",)).get_function(
            kernel_name
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


@torch.library.custom_op("sj::cupy_if_linear_forward", mutates_args=())
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
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = _check_forward_inputs(x_seq, v_init, weight_t, bias, threads)
    kernel = _get_if_kernel(dimensions[4], soft_reset)
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


@torch.library.custom_op("sj::cupy_lif_linear_forward", mutates_args=())
def cupy_lif_linear_forward(
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
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tau <= 1.0:
        raise ValueError("tau must be greater than 1")
    dimensions = _check_forward_inputs(x_seq, v_init, weight_t, bias, threads)
    kernel = _get_lif_kernel(dimensions[4], decay_input, soft_reset)
    return _launch(
        kernel,
        x_seq,
        v_init,
        weight_t,
        bias,
        dimensions,
        (np.float32(1.0 / tau), np.float32(v_threshold), np.float32(v_reset)),
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


@torch.library.register_fake("sj::cupy_if_linear_forward")
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
    threads,
):
    return _fake_outputs(x_seq, v_init, weight_t)


@torch.library.register_fake("sj::cupy_lif_linear_forward")
def _cupy_lif_linear_forward_fake(
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
    threads,
):
    return _fake_outputs(x_seq, v_init, weight_t)


def _recompute_if(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    ctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .. import functional

    if x_seq.shape[0] == 1:
        spike, v = functional.if_step_cupy(
            x_seq[0],
            v,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.surrogate_function,
            ctx.detach_reset,
        )
        spike_seq = spike.unsqueeze(0)
    else:
        spike_seq, v, _ = functional.if_multi_step_cupy(
            x_seq,
            v,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.surrogate_function,
            ctx.detach_reset,
        )
    return spike_seq, v


def _recompute_lif(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    ctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .. import functional

    if x_seq.shape[0] == 1:
        spike, v = functional.lif_step_cupy(
            x_seq[0],
            v,
            ctx.tau,
            ctx.decay_input,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.surrogate_function,
            ctx.detach_reset,
        )
        spike_seq = spike.unsqueeze(0)
    else:
        spike_seq, v, _ = functional.lif_multi_step_cupy(
            x_seq,
            v,
            ctx.tau,
            ctx.decay_input,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.surrogate_function,
            ctx.detach_reset,
        )
    return spike_seq, v


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
):
    ctx.save_for_backward(x_seq, v_init, weight_t, bias)
    ctx.v_threshold = v_threshold
    ctx.v_reset = None if soft_reset else v_reset
    ctx.detach_reset = detach_reset
    ctx.surrogate_function = cuda_utils.resolve_python_object(surrogate_id)


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
    )


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
    )
    ctx.tau = tau
    ctx.decay_input = decay_input


def _linear_backward(ctx, grad_y, grad_v_out, recompute):
    x_seq, v_init, weight_t, bias = ctx.saved_tensors
    if grad_y is None:
        grad_y = torch.zeros(
            (*x_seq.shape[:2], weight_t.shape[1]),
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
    if grad_v_out is None:
        grad_v_out = torch.zeros_like(v_init)

    with torch.enable_grad():
        x = x_seq.detach().requires_grad_()
        v = v_init.detach().requires_grad_()
        spike, v_out = recompute(x, v, ctx)
        grad_spike = torch.matmul(grad_y, weight_t.t())
        grad_x, grad_v = torch.autograd.grad(
            (spike, v_out), (x, v), (grad_spike, grad_v_out)
        )
    K, N = weight_t.shape
    grad_w = torch.mm(spike.detach().reshape(-1, K).t(), grad_y.reshape(-1, N))
    grad_b = grad_y.reshape(-1, N).sum(0) if bias is not None else None
    return grad_x, grad_v, grad_w, grad_b


def _if_backward(ctx, grad_y, grad_v_out):
    return _linear_backward(ctx, grad_y, grad_v_out, _recompute_if) + (None,) * 6


torch.library.register_autograd(
    "sj::cupy_if_linear_forward",
    _if_backward,
    setup_context=_setup_if_context,
)


def _lif_backward(ctx, grad_y, grad_v_out):
    return _linear_backward(ctx, grad_y, grad_v_out, _recompute_lif) + (None,) * 8


torch.library.register_autograd(
    "sj::cupy_lif_linear_forward",
    _lif_backward,
    setup_context=_setup_lif_context,
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
    surrogate_id = (
        cuda_utils.register_python_object(surrogate_function) if needs_backward else 0
    )
    return (
        x_seq.contiguous(),
        v.contiguous(),
        weight_t.contiguous(),
        None if bias is None else bias.contiguous(),
        surrogate_id,
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
    x_seq, v, weight_t, bias, surrogate_id, single_step = _prepare_inputs(
        x, v, weight_t, bias, surrogate_function
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
        threads=threads,
    )
    return (y_seq[0] if single_step else y_seq), v_out


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
    x_seq, v, weight_t, bias, surrogate_id, single_step = _prepare_inputs(
        x, v, weight_t, bias, surrogate_function
    )
    y_seq, v_out = cupy_lif_linear_forward(
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
        threads=threads,
    )
    return (y_seq[0] if single_step else y_seq), v_out
