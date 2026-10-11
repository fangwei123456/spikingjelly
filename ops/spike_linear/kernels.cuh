
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void pack_kernel(const void *__restrict__ S,
                                       unsigned char *__restrict__ S_packed, int M,
                                       int K, int K_PACKED, int dtype) {
    int m = blockIdx.x;
    if (m >= M)
        return;
    for (int kp = threadIdx.x; kp < K_PACKED; kp += blockDim.x) {
        unsigned char b = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int k = kp * 8 + i;
            int offset = m * K + k;
            bool active = false;
            if (k < K) {
                if (dtype == 0) {
                    active = static_cast<const float *>(S)[offset] > 0.5f;
                } else if (dtype == 1) {
                    active =
                        __half2float(static_cast<const __half *>(S)[offset]) > 0.5f;
                } else {
                    active = __bfloat162float(
                                 static_cast<const __nv_bfloat16 *>(S)[offset]) > 0.5f;
                }
            }
            b |= ((unsigned char)active) << i;
        }
        S_packed[m * K_PACKED + kp] = b;
    }
}

#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <> __device__ __forceinline__ float scalar_to_float<float>(float value) {
    return value;
}

template <> __device__ __forceinline__ float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <> __device__ __forceinline__ float float_to_scalar<float>(float value) {
    return value;
}

template <> __device__ __forceinline__ __half float_to_scalar<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
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
__device__ __forceinline__ void
spike_linear_v3_tiled(const unsigned char *__restrict__ S_packed,
                      const scalar_t *__restrict__ W, scalar_t *__restrict__ Y,
                      float *s_W, unsigned char *s_S, int M, int N, int K,
                      int K_PACKED) {
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
                    w_vals[j][i] =
                        s_W[(tx * TN_V3 + j) * BK_PACKED_V3 * 8 + kp * 8 + i];
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
        if (m_global >= M)
            continue;
#pragma unroll
        for (int j = 0; j < TN_V3; j++) {
            int n_global = n0 + tx * TN_V3 + j;
            if (n_global >= N)
                continue;
            Y[m_global * N + n_global] = float_to_scalar<scalar_t>(acc[i][j]);
        }
    }
}

#define DEFINE_V3_KERNEL(name, scalar_t)                                               \
    extern "C" __global__ void name(const unsigned char *S_packed, const scalar_t *W,  \
                                    scalar_t *Y, int M, int N, int K, int K_PACKED) {  \
        __shared__ float s_W[BN_V3 * BK_PACKED_V3 * 8];                                \
        __shared__ unsigned char s_S[BM_V3 * BK_PACKED_V3];                            \
        spike_linear_v3_tiled(S_packed, W, Y, s_W, s_S, M, N, K, K_PACKED);            \
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
__device__ __forceinline__ void
spike_to_row_indices(const scalar_t *__restrict__ S, int *__restrict__ row_counts,
                     int *__restrict__ row_indices, int M, int K) {
    int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0)
        return;

    int count = 0;
    for (int k = 0; k < K; k++) {
        if (scalar_to_float(S[m * K + k]) > 0.5f) {
            row_indices[m * K + count] = k;
            count++;
        }
    }
    row_counts[m] = count;
}

#define DEFINE_INDEX_KERNEL(name, scalar_t)                                            \
    extern "C" __global__ void name(const scalar_t *S, int *row_counts,                \
                                    int *row_indices, int M, int K) {                  \
        spike_to_row_indices(S, row_counts, row_indices, M, K);                        \
    }

DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_fp32, float)
DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_fp16, __half)
DEFINE_INDEX_KERNEL(spike_to_row_indices_kernel_bf16, __nv_bfloat16)

template <typename scalar_t>
__device__ __forceinline__ void spike_linear_v15_sparse_wT(
    const int *__restrict__ row_counts, const int *__restrict__ row_indices,
    const scalar_t *__restrict__ W_T, scalar_t *__restrict__ Y, int M, int N, int K) {
    int blocks_n = (N + blockDim.x - 1) / blockDim.x;
    int block_index = blockIdx.x;
    int m = block_index / blocks_n;
    if (m >= M)
        return;

    int n = (block_index % blocks_n) * blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    int row_nnz = row_counts[m];
    float acc = 0.0f;
    for (int j = 0; j < row_nnz; j++) {
        int k = row_indices[m * K + j];
        acc += scalar_to_float(W_T[k * N + n]);
    }
    Y[m * N + n] = float_to_scalar<scalar_t>(acc);
}

#define DEFINE_V15_KERNEL(name, scalar_t)                                              \
    extern "C" __global__ void name(const int *row_counts, const int *row_indices,     \
                                    const scalar_t *W_T, scalar_t *Y, int M, int N,    \
                                    int K) {                                           \
        spike_linear_v15_sparse_wT(row_counts, row_indices, W_T, Y, M, N, K);          \
    }

DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_fp32, float)
DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_fp16, __half)
DEFINE_V15_KERNEL(spike_linear_v15_sparse_wT_kernel_bf16, __nv_bfloat16)
