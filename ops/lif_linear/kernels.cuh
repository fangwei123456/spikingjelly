#include "../cuda_surrogate.cuh"

#define OUTPUTS_PER_THREAD 4

template <bool Soft, bool Decay>
__device__ __forceinline__ float neuron_charge(float x, float v, float r_tau,
                                               float v_reset) {
    float temp;
    if constexpr (Soft) {
        temp = v;
    } else {
        temp = v - v_reset;
    }
    if constexpr (Decay) {
        temp = r_tau * (x - temp);
    } else {
        temp = x - r_tau * temp;
    }
    return temp + v;
}

template <bool Soft>
__device__ __forceinline__ float neuron_reset(float h, bool spike, float v_threshold,
                                              float v_reset) {
    if constexpr (Soft) {
        return h - (float)spike * v_threshold;
    } else {
        return spike ? v_reset : h;
    }
}

template <bool Soft, bool Decay>
__global__ void
lif_linear_kernel(const float *__restrict__ x_seq, const float *__restrict__ v_init,
                  const float *__restrict__ weight_t, const float *__restrict__ bias,
                  float *__restrict__ y_seq, float *__restrict__ v_out, int T, int M,
                  int K, int N, float r_tau, float v_threshold, float v_reset,
                  int has_bias) {
    int n_per_block = blockDim.x * OUTPUTS_PER_THREAD;
    int n_groups = (N - 1) / n_per_block + 1;
    int m = blockIdx.x / n_groups;
    int n_group = blockIdx.x % n_groups;
    int tid = threadIdx.x;

    // Output groups recompute neuron state to avoid materializing spikes.
    extern __shared__ unsigned char shared[];
    float *v = reinterpret_cast<float *>(shared);
    unsigned int *spike_masks =
        reinterpret_cast<unsigned int *>(shared + K * sizeof(float));

    for (int k = tid; k < K; k += blockDim.x)
        v[k] = v_init[m * K + k];
    __syncthreads();

    for (int t = 0; t < T; t++) {
        float acc[OUTPUTS_PER_THREAD];
#pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
            long long n = (long long)n_group * n_per_block + tid + q * blockDim.x;
            acc[q] = has_bias && n < N ? bias[n] : 0.0f;
        }

        for (int k0 = 0; k0 < K; k0 += blockDim.x) {
            int k = k0 + tid;
            bool spike = false;
            if (k < K) {
                float h = neuron_charge<Soft, Decay>(x_seq[(t * M + m) * K + k], v[k],
                                                     r_tau, v_reset);
                spike = h >= v_threshold;
                v[k] = neuron_reset<Soft>(h, spike, v_threshold, v_reset);
            }
            unsigned int mask = __ballot_sync(0xffffffffu, spike);
            if ((tid & 31) == 0)
                spike_masks[tid >> 5] = mask;
            __syncthreads();

            int tile_warps = (min((int)blockDim.x, K - k0) + 31) >> 5;
            for (int w = 0; w < tile_warps; w++) {
                unsigned int active = spike_masks[w];
                while (active) {
                    int k_active = k0 + (w << 5) + __ffs(active) - 1;
#pragma unroll
                    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                        long long n =
                            (long long)n_group * n_per_block + tid + q * blockDim.x;
                        if (n < N)
                            acc[q] += weight_t[k_active * N + n];
                    }
                    active &= active - 1;
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
            long long n = (long long)n_group * n_per_block + tid + q * blockDim.x;
            if (n < N)
                y_seq[(t * M + m) * N + n] = acc[q];
        }
    }

    if (n_group == 0) {
        for (int k = tid; k < K; k += blockDim.x)
            v_out[m * K + k] = v[k];
    }
}

// The same charge/reset functions and compiler options as fused forward preserve
// spike decisions at the threshold during backward rematerialization.
template <bool Soft, bool Decay>
__global__ void rematerialize(const float *x, const float *v_init, float *spikes,
                              float *charged, int T, int MK, float r_tau,
                              float threshold, float v_reset) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < MK;
         n += blockDim.x * gridDim.x) {
        float v = v_init[n];
        for (int t = 0; t < T; ++t) {
            long long i = (long long)t * MK + n;
            float h = neuron_charge<Soft, Decay>(x[i], v, r_tau, v_reset);
            bool spike = h >= threshold;
            spikes[i] = float(spike);
            charged[i] = h;
            v = neuron_reset<Soft>(h, spike, threshold, v_reset);
        }
    }
}

template <int Surrogate, bool Soft, bool Decay>
__global__ void neuron_backward(const float *grad_spikes, const float *grad_final,
                                const float *charged, const float *surrogate_grad,
                                float *grad_x, float *grad_initial, int T, int MK,
                                float r_tau, float threshold, float v_reset,
                                int detach_reset, float alpha) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < MK;
         n += blockDim.x * gridDim.x) {
        float carry = grad_final[n];
        for (int t = T - 1; t >= 0; --t) {
            long long i = (long long)t * MK + n;
            float h = charged[i];
            float sg;
            if constexpr (Surrogate == -1)
                sg = surrogate_grad[i];
            else
                sg = sj_surrogate_gradient<Surrogate>(h - threshold, alpha);
            float reset_grad;
            if constexpr (Soft) {
                reset_grad = 1.0f;
                if (!detach_reset)
                    reset_grad -= threshold * sg;
            } else {
                reset_grad = 1.0f - float(h >= threshold);
                if (!detach_reset)
                    reset_grad += (v_reset - h) * sg;
            }
            float gh = grad_spikes[i] * sg + carry * reset_grad;
            if constexpr (Decay) {
                grad_x[i] = gh * r_tau;
            } else {
                grad_x[i] = gh;
            }
            carry = gh * (1.0f - r_tau);
        }
        grad_initial[n] = carry;
    }
}
