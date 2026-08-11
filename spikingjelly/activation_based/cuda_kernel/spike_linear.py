"""Experimental hand-written CUDA kernels for binary SpikeLinear.

``sparse_linear`` exposes a row-index sparse kernel and a cuBLAS fallback.
The slower v3 kernel remains available only as a low-level custom op for
pre-packed spike tensors. Both kernels require CUDA, CuPy, contiguous FP32,
FP16, or BF16 tensors, and explicitly registered fake/autograd implementations.
"""

from typing import Literal, Optional

import torch

from . import cuda_utils
from spikingjelly.logger import logger

try:
    import cupy
    from cupy import RawModule
except (ImportError, OSError) as e:
    logger.debug("spikingjelly.activation_based.cuda_kernel.spike_linear: %s", e)
    cupy = None


__all__ = [
    "bit_pack_spike_dense",
    "sparse_linear",
]


# ----------------------------------------------------------------------
# CUDA sources: bit-pack helper, low-level v3, and sparse v15
# ----------------------------------------------------------------------

_BIT_PACK_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
                } else if (dtype == 1) {
                    active = __half2float(
                        static_cast<const __half*>(S)[offset]) > 0.5f;
                } else {
                    active = __bfloat162float(
                        static_cast<const __nv_bfloat16*>(S)[offset]) > 0.5f;
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


template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(
    __nv_bfloat16 value)
{
    return __bfloat162float(value);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(
    float value)
{
    return __float2bfloat16_rn(value);
}


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

template <typename scalar_t>
__device__ __forceinline__ void spike_linear_v3_tiled(
    const unsigned char* __restrict__ S_packed,
    const scalar_t* __restrict__ W,
    scalar_t* __restrict__ Y,
    float* s_W,
    unsigned char* s_S,
    int M, int N, int K, int K_PACKED)
{
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
                w = scalar_to_float(W[n_global * K + k_global]);
            }
            s_W[n_local * BK_PACKED_V3 * 8 + kk] = w;
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
            s_S[m_local * BK_PACKED_V3 + kp_local] = s;
        }

        __syncthreads();

        #pragma unroll
        for (int kp = 0; kp < BK_PACKED_V3; kp++) {
            unsigned char s_bits[TM_V3];
            float w_vals[TN_V3][8];
            #pragma unroll
            for (int i = 0; i < TM_V3; i++) {
                s_bits[i] = s_S[(ty * TM_V3 + i) * BK_PACKED_V3 + kp];
            }
            #pragma unroll
            for (int j = 0; j < TN_V3; j++) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    w_vals[j][i] = s_W[
                        (tx * TN_V3 + j) * BK_PACKED_V3 * 8 + kp * 8 + i
                    ];
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
            Y[m_global * N + n_global] = float_to_scalar<scalar_t>(acc[i][j]);
        }
    }
}

#define DEFINE_V3_KERNEL(name, scalar_t)                                      \
extern "C" __global__ void name(                                             \
    const unsigned char* S_packed, const scalar_t* W, scalar_t* Y,            \
    int M, int N, int K, int K_PACKED)                                        \
{                                                                              \
    __shared__ float s_W[BN_V3 * BK_PACKED_V3 * 8];                           \
    __shared__ unsigned char s_S[BM_V3 * BK_PACKED_V3];                       \
    spike_linear_v3_tiled(S_packed, W, Y, s_W, s_S, M, N, K, K_PACKED);      \
}

DEFINE_V3_KERNEL(spike_linear_v3_tiled_kernel_fp32, float)
DEFINE_V3_KERNEL(spike_linear_v3_tiled_kernel_fp16, __half)
DEFINE_V3_KERNEL(spike_linear_v3_tiled_kernel_bf16, __nv_bfloat16)


// ====================================================================
// v15: per-row sparse indices + W transposed for coalesced reads.
// The fixed-capacity [M, K] index workspace avoids data-dependent host
// synchronization when allocating a compact CSR buffer.
// ====================================================================

template <typename scalar_t>
__device__ __forceinline__ void spike_to_row_indices(
    const scalar_t* __restrict__ S,
    int* __restrict__ row_counts,
    int* __restrict__ row_indices,
    int M, int K)
{
    int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0) return;

    int count = 0;
    for (int k = 0; k < K; k++) {
        if (scalar_to_float(S[m * K + k]) > 0.5f) {
            row_indices[m * K + count] = k;
            count++;
        }
    }
    row_counts[m] = count;
}

#define DEFINE_INDEX_KERNEL(name, scalar_t)                                   \
extern "C" __global__ void name(                                             \
    const scalar_t* S, int* row_counts, int* row_indices, int M, int K)       \
{                                                                              \
    spike_to_row_indices(S, row_counts, row_indices, M, K);                   \
}

DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_fp32, float)
DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_fp16, __half)
DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_bf16, __nv_bfloat16)

template <typename scalar_t>
__device__ __forceinline__ void spike_linear_v15_sparse_wT(
    const int* __restrict__ row_counts,
    const int* __restrict__ row_indices,
    const scalar_t* __restrict__ W_T,
    scalar_t* __restrict__ Y,
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
        acc += scalar_to_float(W_T[k * N + n]);
    }
    Y[m * N + n] = float_to_scalar<scalar_t>(acc);
}

#define DEFINE_V15_KERNEL(name, scalar_t)                                     \
extern "C" __global__ void name(                                             \
    const int* row_counts, const int* row_indices,                            \
    const scalar_t* W_T, scalar_t* Y, int M, int N, int K)                   \
{                                                                              \
    spike_linear_v15_sparse_wT(row_counts, row_indices, W_T, Y, M, N, K);    \
}

DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_fp32, float)
DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_fp16, __half)
DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_bf16, __nv_bfloat16)

"""
)


# ----------------------------------------------------------------------
# Module-level compile / cache
# ----------------------------------------------------------------------

_main_modules: dict[int, object] = {}
_kernel_cache: dict[tuple[int, str], object] = {}
_MAX_CUDA_ELEMENTS = 2**31 - 1
_DTYPE_KERNEL_INFO = {
    torch.float32: ("fp32", 0),
    torch.float16: ("fp16", 1),
    torch.bfloat16: ("bf16", 2),
}


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


def _dtype_kernel_info(tensor: torch.Tensor, name: str) -> tuple[str, int]:
    try:
        return _DTYPE_KERNEL_INFO[tensor.dtype]
    except KeyError as e:
        raise TypeError(f"{name} must have dtype float32, float16, or bfloat16") from e


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
    dtype: torch.dtype,
) -> None:
    _check_cuda_tensor(weight, "weight", dtype, 2)
    if weight.shape[1] != K:
        raise ValueError(f"weight.shape[1] must equal {K}, got {weight.shape[1]}")
    if weight.device != device:
        raise ValueError("spike and weight must be on the same CUDA device")
    if bias is not None:
        _check_cuda_tensor(bias, "bias", dtype, 1)
        if bias.shape[0] != weight.shape[0]:
            raise ValueError("bias.shape[0] must equal weight.shape[0]")
        if bias.device != device:
            raise ValueError("spike and bias must be on the same CUDA device")


# ----------------------------------------------------------------------
# Bit-pack helper (also used directly as a public utility)
# ----------------------------------------------------------------------


def bit_pack_spike_dense(spike: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <bit_pack_spike_dense-cn>` | :ref:`English <bit_pack_spike_dense-en>`

    ----

    .. _bit_pack_spike_dense-cn:

    * **中文**

    将二维二值 CUDA 张量按行打包为 ``uint8``。每个输出字节的第 ``i`` 位
    对应输入的第 ``b * 8 + i`` 列；末尾不足 8 位时补零。输入支持
    ``float32``、``float16`` 和 ``bfloat16``，大于 ``0.5`` 的值编码为 1。

    :param spike: 形状为 ``[M, K]`` 的连续 CUDA 张量。调用方必须保证输入满足
        二值脉冲契约
    :type spike: torch.Tensor
    :return: 形状为 ``[M, ceil(K / 8)]``、与输入位于同一设备的 ``uint8`` 张量
    :rtype: torch.Tensor
    :raises TypeError: ``spike`` 的 dtype 不受支持
    :raises ValueError: ``spike`` 不是二维、连续 CUDA 张量，或元素数超过
        CUDA kernel 的 ``int32`` 索引上限
    :raises RuntimeError: 未安装 CuPy

    ----

    .. _bit_pack_spike_dense-en:

    * **English**

    Pack a two-dimensional binary CUDA tensor into row-major ``uint8`` bytes.
    Bit ``i`` of output byte ``b`` represents input column ``b * 8 + i``;
    a trailing partial byte is zero-padded. The input may use ``float32``,
    ``float16``, or ``bfloat16``. Values greater than ``0.5`` encode as 1.

    :param spike: Contiguous CUDA tensor shaped ``[M, K]``. The caller must
        guarantee the binary-spike contract
    :type spike: torch.Tensor
    :return: A ``uint8`` tensor shaped ``[M, ceil(K / 8)]`` on the input device
    :rtype: torch.Tensor
    :raises TypeError: If ``spike`` has an unsupported dtype
    :raises ValueError: If ``spike`` is not a two-dimensional contiguous CUDA
        tensor, or exceeds the CUDA kernel's ``int32`` indexing limit
    :raises RuntimeError: If CuPy is unavailable
    """
    if spike.dim() != 2:
        raise ValueError("spike must be 2D")
    _, dtype = _dtype_kernel_info(spike, "spike")
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
    r"""
    **API Language** - :ref:`中文 <cupy_spike_linear_v3_dense_forward-cn>` | :ref:`English <cupy_spike_linear_v3_dense_forward-en>`

    ----

    .. _cupy_spike_linear_v3_dense_forward-cn:

    * **中文**

    对预先按位打包的二维脉冲执行实验性 v3 CUDA Linear kernel。低精度输入
    使用 FP32 累加，并将结果转换回权重 dtype。

    :param spike_packed: 形状为 ``[M, ceil(K / 8)]`` 的连续 ``uint8`` CUDA 张量
    :type spike_packed: torch.Tensor
    :param weight: 形状为 ``[N, K]`` 的连续 CUDA 张量，dtype 为
        ``float32``、``float16`` 或 ``bfloat16``
    :type weight: torch.Tensor
    :param bias: 可选的连续 ``[N]`` CUDA 张量，dtype 和设备必须与 ``weight`` 一致
    :type bias: Optional[torch.Tensor]
    :return: 形状为 ``[M, N]``、dtype 与 ``weight`` 相同的 CUDA 张量
    :rtype: torch.Tensor
    :raises TypeError: 输入 dtype 不满足约束
    :raises ValueError: 输入 shape、连续性、设备或元素数量不满足约束
    :raises RuntimeError: 未安装 CuPy

    ----

    .. _cupy_spike_linear_v3_dense_forward-en:

    * **English**

    Apply the experimental v3 CUDA Linear kernel to a pre-packed binary spike
    matrix. Low-precision inputs accumulate in FP32 and are converted back to
    the weight dtype.

    :param spike_packed: Contiguous ``uint8`` CUDA tensor shaped
        ``[M, ceil(K / 8)]``
    :type spike_packed: torch.Tensor
    :param weight: Contiguous ``[N, K]`` CUDA tensor with ``float32``,
        ``float16``, or ``bfloat16`` dtype
    :type weight: torch.Tensor
    :param bias: Optional contiguous ``[N]`` CUDA tensor on the same device and
        with the same dtype as ``weight``
    :type bias: Optional[torch.Tensor]
    :return: CUDA tensor shaped ``[M, N]`` with the weight dtype
    :rtype: torch.Tensor
    :raises TypeError: If an input dtype violates the contract
    :raises ValueError: If an input shape, layout, device, or element count
        violates the contract
    :raises RuntimeError: If CuPy is unavailable
    """
    if cupy is None:
        raise RuntimeError("cupy is required for CUDA SpikeLinear")
    _check_cuda_tensor(spike_packed, "spike_packed", torch.uint8, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K_PACKED = spike_packed.shape
    N, K = weight.shape
    dtype_suffix, _ = _dtype_kernel_info(weight, "weight")
    _check_weight_bias(weight, bias, K, spike_packed.device, weight.dtype)
    if K_PACKED != (K + 7) // 8:
        raise ValueError("spike_packed.shape[1] must equal ceil(weight.shape[1] / 8)")
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    Y = torch.empty((M, N), dtype=weight.dtype, device=spike_packed.device)
    if M > 0 and N > 0:
        if K == 0:
            Y.zero_()
        else:
            device = spike_packed.get_device()
            kernel = _get_kernel(f"spike_linear_v3_tiled_kernel_{dtype_suffix}", device)
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
    if spike_packed.dtype != torch.uint8:
        raise TypeError("spike_packed must have dtype uint8")
    if weight.dtype not in _DTYPE_KERNEL_INFO:
        raise TypeError("weight must have dtype float32, float16, or bfloat16")
    if weight.device != spike_packed.device:
        raise ValueError("spike_packed and weight must be on the same device")
    torch._check(spike_packed.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike_packed.shape[1] == (weight.shape[1] + 7) // 8,
        lambda: "packed width must equal ceil(K / 8)",
    )
    if bias is not None:
        if bias.dtype != weight.dtype:
            raise TypeError("bias must have the same dtype as weight")
        if bias.device != spike_packed.device:
            raise ValueError("spike_packed and bias must be on the same device")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike_packed.shape[0], weight.shape[0]),
        dtype=weight.dtype,
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
    r"""
    **API Language** - :ref:`中文 <cupy_spike_linear_sparse_forward-cn>` | :ref:`English <cupy_spike_linear_sparse_forward-en>`

    ----

    .. _cupy_spike_linear_sparse_forward-cn:

    * **中文**

    使用实验性稀疏行索引 CUDA kernel 对二维二值脉冲执行 Linear。kernel 使用
    固定容量的 ``int32 [M, K]`` 索引工作区，避免为数据相关的紧凑 CSR 缓冲区
    执行 host 同步。低精度输入使用 FP32 累加并转换回输入 dtype。

    :param spike: 形状为 ``[M, K]`` 的连续二值 CUDA 张量，dtype 为
        ``float32``、``float16`` 或 ``bfloat16``
    :type spike: torch.Tensor
    :param weight: 与 ``spike`` 同 dtype、同设备的连续 ``[N, K]`` CUDA 张量
    :type weight: torch.Tensor
    :param bias: 可选的连续 ``[N]`` CUDA 张量，dtype 和设备与 ``spike`` 一致
    :type bias: Optional[torch.Tensor]
    :return: 形状为 ``[M, N]``、dtype 与 ``spike`` 相同的 CUDA 张量
    :rtype: torch.Tensor
    :raises TypeError: 输入 dtype 不满足约束
    :raises ValueError: 输入 shape、连续性、设备或元素数量不满足约束
    :raises RuntimeError: 未安装 CuPy

    ----

    .. _cupy_spike_linear_sparse_forward-en:

    * **English**

    Apply Linear to a two-dimensional binary spike tensor with the experimental
    sparse row-index CUDA kernel. A fixed-capacity ``int32 [M, K]`` index
    workspace avoids the host synchronization required by a data-dependent
    compact CSR allocation. Low-precision inputs accumulate in FP32 and are
    converted back to the input dtype.

    :param spike: Contiguous binary ``[M, K]`` CUDA tensor with ``float32``,
        ``float16``, or ``bfloat16`` dtype
    :type spike: torch.Tensor
    :param weight: Contiguous ``[N, K]`` CUDA tensor on the same device and with
        the same dtype as ``spike``
    :type weight: torch.Tensor
    :param bias: Optional contiguous ``[N]`` CUDA tensor on the same device and
        with the same dtype as ``spike``
    :type bias: Optional[torch.Tensor]
    :return: CUDA tensor shaped ``[M, N]`` with the spike dtype
    :rtype: torch.Tensor
    :raises TypeError: If an input dtype violates the contract
    :raises ValueError: If an input shape, layout, device, or element count
        violates the contract
    :raises RuntimeError: If CuPy is unavailable
    """
    if cupy is None:
        raise RuntimeError("cupy is required for CUDA SpikeLinear")
    dtype_suffix, _ = _dtype_kernel_info(spike, "spike")
    _check_cuda_tensor(spike, "spike", spike.dtype, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K = spike.shape
    N = weight.shape[0]
    if weight.dtype != spike.dtype:
        raise TypeError("spike and weight must have the same dtype")
    _check_weight_bias(weight, bias, K, spike.device, spike.dtype)
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    Y = torch.empty((M, N), dtype=spike.dtype, device=spike.device)
    if M > 0 and N > 0:
        device = spike.get_device()
        row_counts = torch.zeros(M, dtype=torch.int32, device=spike.device)
        row_indices = torch.empty((M, K), dtype=torch.int32, device=spike.device)
        if K > 0:
            index_kernel = _get_kernel(
                f"spike_to_row_indices_kernel_{dtype_suffix}", device
            )
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
        kernel = _get_kernel(
            f"spike_linear_v15_sparse_wT_kernel_{dtype_suffix}", device
        )
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
    if spike.dtype not in _DTYPE_KERNEL_INFO:
        raise TypeError("spike must have dtype float32, float16, or bfloat16")
    if weight.dtype != spike.dtype:
        raise TypeError("spike and weight must have the same dtype")
    if weight.device != spike.device:
        raise ValueError("spike and weight must be on the same device")
    torch._check(spike.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike.shape[1] == weight.shape[1],
        lambda: "spike and weight K dimensions must match",
    )
    if bias is not None:
        if bias.dtype != spike.dtype:
            raise TypeError("bias must have the same dtype as spike")
        if bias.device != spike.device:
            raise ValueError("spike and bias must be on the same device")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike.shape[0], weight.shape[0]),
        dtype=spike.dtype,
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
    r"""
    **API Language** - :ref:`中文 <sparse_linear-cn>` | :ref:`English <sparse_linear-en>`

    ----

    .. _sparse_linear-cn:

    * **中文**

    对未按位打包的二值脉冲执行 Linear。``strategy="torch"`` 直接调用
    :func:`torch.nn.functional.linear`。``strategy="sparse"`` 使用实验性稀疏
    CUDA kernel，要求二维、同设备且同 dtype 的输入，并使用
    ``4 * M * K`` 字节的 ``int32`` 索引工作区。稀疏 kernel 以 ``> 0.5``
    判断脉冲，调用方必须保证二值输入契约，因为运行时检查数值会同步设备。

    :param spike: 输入脉冲。稀疏策略要求形状为 ``[M, K]`` 的 CUDA 张量，dtype
        为 ``float32``、``float16`` 或 ``bfloat16``
    :type spike: torch.Tensor
    :param weight: 形状为 ``[N, K]`` 的权重；稀疏策略要求与 ``spike`` 同 dtype
        和设备
    :type weight: torch.Tensor
    :param bias: 可选的 ``[N]`` 偏置；稀疏策略要求与 ``spike`` 同 dtype 和设备
    :type bias: Optional[torch.Tensor]
    :param strategy: ``"torch"`` 或 ``"sparse"``，默认 ``"torch"``
    :type strategy: Literal["torch", "sparse"]
    :return: Linear 输出；稀疏策略返回 ``[M, N]`` 且保持输入 dtype
    :rtype: torch.Tensor
    :raises ValueError: ``strategy`` 未知，或稀疏策略的 shape、设备、元素数量不满足约束
    :raises TypeError: 稀疏策略的 dtype 不满足约束
    :raises RuntimeError: 稀疏策略下未安装 CuPy

    ----

    .. _sparse_linear-en:

    * **English**

    Apply Linear to an unpacked binary spike tensor. ``strategy="torch"`` calls
    :func:`torch.nn.functional.linear` directly. ``strategy="sparse"`` uses the
    experimental sparse CUDA kernel and requires two-dimensional inputs with
    matching devices and dtypes. It allocates an ``int32`` index
    workspace of ``4 * M * K`` bytes. The sparse kernel thresholds at ``> 0.5``;
    the caller must guarantee binary values because validating them would
    synchronize the device.

    :param spike: Input spikes. The sparse strategy requires a ``[M, K]`` CUDA
        tensor with ``float32``, ``float16``, or ``bfloat16`` dtype
    :type spike: torch.Tensor
    :param weight: Weight shaped ``[N, K]``. The sparse strategy requires the
        same dtype and device as ``spike``
    :type weight: torch.Tensor
    :param bias: Optional ``[N]`` bias. The sparse strategy requires the same
        dtype and device as ``spike``
    :type bias: Optional[torch.Tensor]
    :param strategy: ``"torch"`` or ``"sparse"``. Defaults to ``"torch"``
    :type strategy: Literal["torch", "sparse"]
    :return: Linear output. The sparse strategy returns ``[M, N]`` and preserves
        the input dtype
    :rtype: torch.Tensor
    :raises ValueError: If ``strategy`` is unknown, or a sparse input shape,
        device, or element count violates the contract
    :raises TypeError: If a sparse input dtype violates the contract
    :raises RuntimeError: If CuPy is unavailable for the sparse strategy
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
