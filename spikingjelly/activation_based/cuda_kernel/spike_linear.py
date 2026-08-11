"""Experimental hand-written CUDA kernels for binary SpikeLinear.

``sparse_linear`` exposes a row-index sparse kernel and a cuBLAS fallback.
The slower v3 kernel remains available only as a low-level custom op for
pre-packed spike tensors. Both kernels require CUDA, CuPy, contiguous FP32
weights, and explicitly registered fake/autograd implementations.
"""

from typing import Literal, Optional

import torch

from . import cuda_utils
from spikingjelly.logger import logger

try:
    import cupy
    from cupy import RawModule
except (ImportError, OSError) as e:
    logger.debug("spikingjelly.activation_based.cuda_kernel.spike_linear: {}", e)
    cupy = None


__all__ = [
    "bit_pack_spike_dense",
    "sparse_linear",
]


# ----------------------------------------------------------------------
# CUDA sources: bit-pack helper, low-level v3, and sparse v15
# ----------------------------------------------------------------------

_BIT_PACK_SRC = r"""
extern "C" __global__ void pack_kernel(
    const void* __restrict__ S,
    unsigned char* __restrict__ S_packed,
    int M, int K, int K_PACKED, int dtype)
{
    int m = blockIdx.x;
    if (m >= M) return;
    for (int kp = threadIdx.x; kp < K_PACKED; kp += blockDim.x) {
        unsigned char b = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k = kp * 8 + i;
            int offset = m * K + k;
            bool active = false;
            if (k < K) {
                if (dtype == 0) {
                    active = static_cast<const float*>(S)[offset] > 0.5f;
                } else {
                    // FP16 and BF16 binary 0/1 values share zero/nonzero storage.
                    active = static_cast<const unsigned short*>(S)[offset] != 0;
                }
            }
            b |= ((unsigned char)active) << i;
        }
        S_packed[m * K_PACKED + kp] = b;
    }
}
"""


_CUDA_SRC = (
    _BIT_PACK_SRC
    + r"""
#include <cuda_runtime.h>


// ====================================================================
// v3: bit-packed dense GEMM with register tiling and shared-mem tile.
// Block: (16, 8) = 128 threads, each computes TM=8 x TN=8 = 64 outputs.
// Block tile: BM=64 rows, BN=128 cols, BK_PACKED=8 (=64 K values per
// inner iter). Shared mem: 128*64*4 + 64*8 = 33KB (within 48KB limit).
// ====================================================================

#define TY_V3 8
#define TX_V3 16
#define TM_V3 8
#define TN_V3 8
#define BK_PACKED_V3 8
#define BM_V3 (TY_V3 * TM_V3)
#define BN_V3 (TX_V3 * TN_V3)

extern "C" __global__ void spike_linear_v3_tiled_kernel(
    const unsigned char* __restrict__ S_packed,
    const float*   __restrict__ W,
    float*         __restrict__ Y,
    int M, int N, int K, int K_PACKED)
{
    __shared__ float s_W[BN_V3][BK_PACKED_V3 * 8];
    __shared__ unsigned char s_S[BM_V3][BK_PACKED_V3];

    int blocks_n = (N + BN_V3 - 1) / BN_V3;
    int block_index = blockIdx.x;
    int n0 = (block_index % blocks_n) * BN_V3;
    int m0 = (block_index / blocks_n) * BM_V3;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float acc[TM_V3][TN_V3];
    #pragma unroll
    for (int i = 0; i < TM_V3; i++)
        #pragma unroll
        for (int j = 0; j < TN_V3; j++)
            acc[i][j] = 0.0f;

    int tid = ty * TX_V3 + tx;
    int block_threads = TY_V3 * TX_V3;

    for (int k_chunk = 0; k_chunk < K_PACKED; k_chunk += BK_PACKED_V3) {
        int w_total = BN_V3 * BK_PACKED_V3 * 8;
        for (int idx = tid; idx < w_total; idx += block_threads) {
            int n_local = idx / (BK_PACKED_V3 * 8);
            int kk = idx % (BK_PACKED_V3 * 8);
            int n_global = n0 + n_local;
            int k_global = k_chunk * 8 + kk;
            float w = 0.0f;
            if (n_global < N && k_global < K) {
                w = W[n_global * K + k_global];
            }
            s_W[n_local][kk] = w;
        }

        int s_total = BM_V3 * BK_PACKED_V3;
        for (int idx = tid; idx < s_total; idx += block_threads) {
            int m_local = idx / BK_PACKED_V3;
            int kp_local = idx % BK_PACKED_V3;
            int m_global = m0 + m_local;
            int kp_global = k_chunk + kp_local;
            unsigned char s = 0;
            if (m_global < M && kp_global < K_PACKED) {
                s = S_packed[m_global * K_PACKED + kp_global];
            }
            s_S[m_local][kp_local] = s;
        }

        __syncthreads();

        #pragma unroll
        for (int kp = 0; kp < BK_PACKED_V3; kp++) {
            unsigned char s_bits[TM_V3];
            float w_vals[TN_V3][8];
            #pragma unroll
            for (int i = 0; i < TM_V3; i++) {
                s_bits[i] = s_S[ty * TM_V3 + i][kp];
            }
            #pragma unroll
            for (int j = 0; j < TN_V3; j++) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    w_vals[j][i] = s_W[tx * TN_V3 + j][kp * 8 + i];
                }
            }
            #pragma unroll
            for (int i = 0; i < TM_V3; i++) {
                #pragma unroll
                for (int j = 0; j < TN_V3; j++) {
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        acc[i][j] += w_vals[j][b] * (float)((s_bits[i] >> b) & 1);
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM_V3; i++) {
        int m_global = m0 + ty * TM_V3 + i;
        if (m_global >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN_V3; j++) {
            int n_global = n0 + tx * TN_V3 + j;
            if (n_global >= N) continue;
            Y[m_global * N + n_global] = acc[i][j];
        }
    }
}


// ====================================================================
// v15: per-row sparse indices + W transposed for coalesced reads.
// The fixed-capacity [M, K] index workspace avoids data-dependent host
// synchronization when allocating a compact CSR buffer.
// ====================================================================

extern "C" __global__ void spike_to_row_indices_kernel(
    const float* __restrict__ S,
    int* __restrict__ row_counts,
    int* __restrict__ row_indices,
    int M, int K)
{
    int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0) return;

    int count = 0;
    for (int k = 0; k < K; k++) {
        if (S[m * K + k] > 0.5f) {
            row_indices[m * K + count] = k;
            count++;
        }
    }
    row_counts[m] = count;
}

extern "C" __global__ void spike_linear_v15_sparse_wT_kernel(
    const int* __restrict__ row_counts,
    const int* __restrict__ row_indices,
    const float* __restrict__ W_T,
    float* __restrict__ Y,
    int M, int N, int K)
{
    int blocks_n = (N + blockDim.x - 1) / blockDim.x;
    int block_index = blockIdx.x;
    int m = block_index / blocks_n;
    if (m >= M) return;

    int n = (block_index % blocks_n) * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int row_nnz = row_counts[m];
    float acc = 0.0f;
    for (int j = 0; j < row_nnz; j++) {
        int k = row_indices[m * K + j];
        acc += W_T[k * N + n];
    }
    Y[m * N + n] = acc;
}

"""
)


# ----------------------------------------------------------------------
# Module-level compile / cache
# ----------------------------------------------------------------------

_main_modules: dict[int, object] = {}
_kernel_cache: dict[tuple[int, str], object] = {}
_MAX_CUDA_ELEMENTS = 2**31 - 1


def _get_main_module(device: int):
    if cupy is None:
        raise RuntimeError("cupy is required for CUDA SpikeLinear")
    if device not in _main_modules:
        with cuda_utils.DeviceEnvironment(device):
            _main_modules[device] = RawModule(
                code=_CUDA_SRC,
                options=("--use_fast_math",),
            )
    return _main_modules[device]


def _get_kernel(name: str, device: int):
    key = (device, name)
    if key not in _kernel_cache:
        with cuda_utils.DeviceEnvironment(device):
            _kernel_cache[key] = _get_main_module(device).get_function(name)
    return _kernel_cache[key]


def _launch(kernel, grid, block, args, device: int):
    with cuda_utils.DeviceEnvironment(device):
        stream = cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        kernel(grid, block, args, stream=stream)


def _check_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> None:
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() > _MAX_CUDA_ELEMENTS:
        raise ValueError(
            f"{name} exceeds the {_MAX_CUDA_ELEMENTS}-element CUDA kernel limit"
        )


def _check_weight_bias(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    device: torch.device,
) -> None:
    _check_cuda_tensor(weight, "weight", torch.float32, 2)
    if weight.shape[1] != K:
        raise ValueError(f"weight.shape[1] must equal {K}, got {weight.shape[1]}")
    if weight.device != device:
        raise ValueError("spike and weight must be on the same CUDA device")
    if bias is not None:
        _check_cuda_tensor(bias, "bias", torch.float32, 1)
        if bias.shape[0] != weight.shape[0]:
            raise ValueError("bias.shape[0] must equal weight.shape[0]")
        if bias.device != device:
            raise ValueError("spike and bias must be on the same CUDA device")


# ----------------------------------------------------------------------
# Bit-pack helper (also used directly as a public utility)
# ----------------------------------------------------------------------


def bit_pack_spike_dense(spike: torch.Tensor) -> torch.Tensor:
    """Pack a contiguous binary CUDA matrix into row-major uint8 bytes.

    ``spike`` may be FP32, FP16, or BF16. Bit ``i`` of byte ``b`` represents
    ``spike[:, b * 8 + i]``; a trailing partial byte is zero-padded.
    """
    if spike.dim() != 2:
        raise ValueError("spike must be 2D")
    dtype = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }.get(spike.dtype)
    if dtype is None:
        raise TypeError("spike must have dtype float32, float16, or bfloat16")
    if not spike.is_cuda:
        raise ValueError("spike must be a CUDA tensor")
    if not spike.is_contiguous():
        raise ValueError("spike must be contiguous")
    if spike.numel() > _MAX_CUDA_ELEMENTS:
        raise ValueError(
            f"spike exceeds the {_MAX_CUDA_ELEMENTS}-element CUDA kernel limit"
        )

    M, K = spike.shape
    K_PACKED = (K + 7) // 8
    out = torch.empty((M, K_PACKED), dtype=torch.uint8, device=spike.device)
    if M == 0 or K_PACKED == 0:
        return out

    device = spike.get_device()
    kernel = _get_kernel("pack_kernel", device)
    _launch(
        kernel,
        (M,),
        (min(K_PACKED, 256),),
        (spike.data_ptr(), out.data_ptr(), M, K, K_PACKED, dtype),
        device,
    )
    return out


# ----------------------------------------------------------------------
# v3 (dense, bit-packed) — custom_op
# ----------------------------------------------------------------------


@torch.library.custom_op("sj::cupy_spike_linear_v3_dense_forward", mutates_args=())
def cupy_spike_linear_v3_dense_forward(
    spike_packed: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply Linear to a pre-packed binary spike matrix.

    Args:
        spike_packed: contiguous uint8 tensor with shape
            ``[M, ceil(K / 8)]``.
        weight: contiguous FP32 CUDA tensor with shape ``[N, K]``.
        bias: optional contiguous FP32 CUDA tensor with shape ``[N]``.
    """
    if cupy is None:
        raise RuntimeError("cupy is required for CUDA SpikeLinear")
    _check_cuda_tensor(spike_packed, "spike_packed", torch.uint8, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K_PACKED = spike_packed.shape
    N, K = weight.shape
    _check_weight_bias(weight, bias, K, spike_packed.device)
    if K_PACKED != (K + 7) // 8:
        raise ValueError("spike_packed.shape[1] must equal ceil(weight.shape[1] / 8)")
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    Y = torch.empty((M, N), dtype=torch.float32, device=spike_packed.device)
    if M > 0 and N > 0:
        if K == 0:
            Y.zero_()
        else:
            device = spike_packed.get_device()
            kernel = _get_kernel("spike_linear_v3_tiled_kernel", device)
            gx = (N + 16 * 8 - 1) // (16 * 8)
            gy = (M + 8 * 8 - 1) // (8 * 8)
            _launch(
                kernel,
                (gx * gy,),
                (16, 8),
                (
                    spike_packed.data_ptr(),
                    weight.data_ptr(),
                    Y.data_ptr(),
                    M,
                    N,
                    K,
                    K_PACKED,
                ),
                device,
            )

    if bias is not None:
        Y = Y + bias
    return Y


@torch.library.register_fake("sj::cupy_spike_linear_v3_dense_forward")
def _cupy_spike_linear_v3_dense_forward_fake(
    spike_packed: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if spike_packed.dtype != torch.uint8 or weight.dtype != torch.float32:
        raise TypeError("v3 requires uint8 spike_packed and float32 weight")
    torch._check(spike_packed.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike_packed.shape[1] == (weight.shape[1] + 7) // 8,
        lambda: "packed width must equal ceil(K / 8)",
    )
    if bias is not None:
        if bias.dtype != torch.float32:
            raise TypeError("bias must be float32")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike_packed.shape[0], weight.shape[0]),
        dtype=torch.float32,
        device=spike_packed.device,
    )


def _setup_v3_context(ctx, inputs, output):
    del output
    spike_packed, weight, bias = inputs
    ctx.save_for_backward(spike_packed, weight, bias)


def _v3_backward(ctx, grad_output):
    spike_packed, weight, bias = ctx.saved_tensors
    M, K_PACKED = spike_packed.shape
    K = weight.shape[1]
    with torch.cuda.device(weight.device):
        bits = (
            spike_packed.unsqueeze(-1)
            >> torch.arange(8, dtype=torch.uint8, device=spike_packed.device)
        ) & 1
        spike = bits.reshape(M, K_PACKED * 8)[:, :K].to(grad_output.dtype)
        grad_weight = torch.mm(grad_output.t(), spike)
        grad_bias = grad_output.sum(0) if bias is not None else None
    return None, grad_weight, grad_bias


torch.library.register_autograd(
    "sj::cupy_spike_linear_v3_dense_forward",
    _v3_backward,
    setup_context=_setup_v3_context,
)


# ----------------------------------------------------------------------
# v15 (true-sparse, W transposed) — custom_op
# ----------------------------------------------------------------------


@torch.library.custom_op("sj::cupy_spike_linear_sparse_forward", mutates_args=())
def cupy_spike_linear_sparse_forward(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply Linear using per-row spike indices and transposed weights.

    The CUDA preprocessing kernel writes a fixed-capacity ``[M, K]`` index
    workspace. This avoids the host synchronization required to allocate a
    compact data-dependent CSR buffer.

    Args:
        spike: contiguous binary FP32 CUDA tensor with shape ``[M, K]``.
        weight: contiguous FP32 CUDA tensor with shape ``[N, K]``.
        bias: optional contiguous FP32 CUDA tensor with shape ``[N]``.
    """
    if cupy is None:
        raise RuntimeError("cupy is required for CUDA SpikeLinear")
    _check_cuda_tensor(spike, "spike", torch.float32, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K = spike.shape
    N = weight.shape[0]
    _check_weight_bias(weight, bias, K, spike.device)
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    Y = torch.empty((M, N), dtype=torch.float32, device=spike.device)
    if M > 0 and N > 0:
        device = spike.get_device()
        row_counts = torch.zeros(M, dtype=torch.int32, device=spike.device)
        row_indices = torch.empty((M, K), dtype=torch.int32, device=spike.device)
        if K > 0:
            index_kernel = _get_kernel("spike_to_row_indices_kernel", device)
            _launch(
                index_kernel,
                (M,),
                (1,),
                (
                    spike.data_ptr(),
                    row_counts.data_ptr(),
                    row_indices.data_ptr(),
                    M,
                    K,
                ),
                device,
            )

        weight_t = weight.t().contiguous()
        kernel = _get_kernel("spike_linear_v15_sparse_wT_kernel", device)
        n_blocks = (N + 255) // 256
        _launch(
            kernel,
            (M * n_blocks,),
            (256, 1, 1),
            (
                row_counts.data_ptr(),
                row_indices.data_ptr(),
                weight_t.data_ptr(),
                Y.data_ptr(),
                M,
                N,
                K,
            ),
            device,
        )

    if bias is not None:
        Y = Y + bias
    return Y


@torch.library.register_fake("sj::cupy_spike_linear_sparse_forward")
def _cupy_spike_linear_sparse_forward_fake(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if spike.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("sparse kernel requires float32 spike and weight")
    torch._check(spike.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike.shape[1] == weight.shape[1],
        lambda: "spike and weight K dimensions must match",
    )
    if bias is not None:
        if bias.dtype != torch.float32:
            raise TypeError("bias must be float32")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike.shape[0], weight.shape[0]),
        dtype=torch.float32,
        device=spike.device,
    )


def _setup_v15_context(ctx, inputs, output):
    del output
    spike, weight, bias = inputs
    ctx.save_for_backward(spike, weight, bias)


def _v15_backward(ctx, grad_output):
    spike, weight, bias = ctx.saved_tensors
    with torch.cuda.device(weight.device):
        grad_spike = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), spike.to(grad_output.dtype))
        grad_bias = grad_output.sum(0) if bias is not None else None
    return grad_spike, grad_weight, grad_bias


torch.library.register_autograd(
    "sj::cupy_spike_linear_sparse_forward",
    _v15_backward,
    setup_context=_setup_v15_context,
)


# ----------------------------------------------------------------------
# Public API: sparse_linear
# ----------------------------------------------------------------------


def sparse_linear(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    strategy: Literal["torch", "sparse"] = "torch",
) -> torch.Tensor:
    """Apply Linear to an unbitpacked binary spike tensor.

    Args:
        spike: input tensor containing only 0 and 1.
        weight: FP32 tensor with shape ``[N, K]``.
        bias: optional FP32 tensor with shape ``[N]``.
        strategy: ``"torch"`` (default) or ``"sparse"``.

    ``"sparse"`` requires a 2D FP32 CUDA ``spike`` and CuPy. It uses an
    int32 ``[M, K]`` workspace (``4 * M * K`` bytes) and is intended for
    explicitly profiled low-density workloads. It thresholds values at
    ``> 0.5`` and treats selected values as 1; the caller must guarantee the
    binary input contract because checking values would synchronize the device.

    ``"torch"`` calls :func:`torch.nn.functional.linear` and accepts all input
    shapes and dtypes supported by that function. There is no automatic
    strategy selection or density synchronization.
    """
    if strategy not in ("torch", "sparse"):
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from: 'torch', 'sparse'."
        )
    if strategy == "torch":
        return torch.nn.functional.linear(spike, weight, bias)
    return cupy_spike_linear_sparse_forward(
        spike.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
    )
