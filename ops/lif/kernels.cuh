#include "../_cuda.cuh"

template <class scalar_t>
__global__ void lif_forward(const scalar_t *x, const float *v_init, scalar_t *spikes,
                            float *voltages, float *charged, long long T, long long N,
                            float tau, int decay_input, float threshold, float reset,
                            int soft_reset, int store_v_seq) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        float v = v_init[n];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float h = decay_input ? v + (float(x[i]) - (v - reset)) / tau
                                        : v - (v - reset) / tau + float(x[i]);
            const float spike = h >= threshold ? 1.0f : 0.0f;
            v = soft_reset ? h - spike * threshold : spike * reset + (1.0f - spike) * h;
            spikes[i] = scalar_t(spike);
            if (store_v_seq)
                voltages[i] = v;
            charged[i] = h;
        }
        if (!store_v_seq)
            voltages[n] = v;
    }
}

template <class scalar_t, int Surrogate, int Elements>
__global__ void lif_backward(const scalar_t *gs, const float *gv, const float *charged,
                             scalar_t *gx, float *gv_init, long long T, long long N,
                             float tau, int decay_input, float threshold, float reset,
                             int soft_reset, int detach_reset, float alpha,
                             int store_v_seq) {
    // Sharing the packed loop regresses soft, non-detached reset in FP16/BF16;
    // retain the original scalar loop for that path and for narrow inputs.
    if constexpr (Elements == 1) {
        for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
             n += (long long)blockDim.x * gridDim.x) {
            float carry = 0.0f;
            for (long long t = T - 1; t >= 0; --t) {
                const long long i = t * N + n;
                const float h = charged[i];
                const float sg = sj_surrogate_gradient<Surrogate>(h - threshold, alpha);
                float reset_grad = soft_reset ? 1.0f : 1.0f - float(h >= threshold);
                if (!detach_reset) {
                    reset_grad += soft_reset ? -threshold * sg : (reset - h) * sg;
                }
                const float gh =
                    float(gs[i]) * sg +
                    ((store_v_seq ? gv[i] : (t == T - 1 ? gv[n] : 0.0f)) + carry) *
                        reset_grad;
                gx[i] = scalar_t(decay_input ? gh / tau : gh);
                carry = gh - gh / tau;
            }
            gv_init[n] = carry;
        }
    } else {
        // Pack two recurrences only for wide inputs: narrow inputs need more active
        // threads to hide latency. Keep 64-bit offsets for large tensors.
        for (long long base =
                 blockIdx.x * (long long)blockDim.x * Elements + threadIdx.x;
             base < N; base += (long long)blockDim.x * gridDim.x * Elements) {
            float carry[Elements];
#pragma unroll
            for (int e = 0; e < Elements; ++e) {
                const long long n = base + (long long)e * blockDim.x;
                carry[e] = !store_v_seq && n < N ? gv[n] : 0.0f;
            }
            for (long long t = T - 1; t >= 0; --t) {
#pragma unroll
                for (int e = 0; e < Elements; ++e) {
                    const long long n = base + (long long)e * blockDim.x;
                    if (n < N) {
                        const long long i = t * N + n;
                        const float h = charged[i];
                        const float sg =
                            sj_surrogate_gradient<Surrogate>(h - threshold, alpha);
                        float reset_grad =
                            soft_reset ? 1.0f : 1.0f - float(h >= threshold);
                        if (!detach_reset) {
                            reset_grad +=
                                soft_reset ? -threshold * sg : (reset - h) * sg;
                        }
                        if (store_v_seq)
                            carry[e] = gv[i] + carry[e];
                        const float gh = float(gs[i]) * sg + carry[e] * reset_grad;
                        const float decay = gh / tau;
                        gx[i] = scalar_t(decay_input ? decay : gh);
                        carry[e] = gh - decay;
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < Elements; ++e) {
                const long long n = base + (long long)e * blockDim.x;
                if (n < N)
                    gv_init[n] = carry[e];
            }
        }
    }
}
