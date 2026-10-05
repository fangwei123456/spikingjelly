#include "../_cuda.cuh"

template <class scalar_t>
__global__ void plif_forward(const scalar_t *x, const float *v_init, const float *q_ptr,
                             scalar_t *spikes, float *voltages, float *charged,
                             long long T, long long N, int decay_input, float threshold,
                             float reset, int soft_reset, int store_v_seq) {
    const float q = q_ptr[0];
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        float v = v_init[n];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float h = decay_input ? v + (float(x[i]) - (v - reset)) * q
                                        : v - (v - reset) * q + float(x[i]);
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

template <class scalar_t, int Surrogate>
__global__ void
plif_backward(const scalar_t *gs, const float *gv, const scalar_t *x,
              const float *v_init, const float *q_ptr, const float *charged,
              scalar_t *gx, float *gv_init, float *gq_per_neuron, long long T,
              long long N, int decay_input, float threshold, float reset,
              int soft_reset, int detach_reset, float alpha, int store_v_seq) {
    const float q = q_ptr[0];
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        float carry = 0.0f;
        float gq = 0.0f;
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
            gx[i] = scalar_t(decay_input ? gh * q : gh);
            carry = gh - gh * q;
            float previous = v_init[n];
            if (t > 0) {
                const float previous_h = charged[i - N];
                const float spike = previous_h >= threshold ? 1.0f : 0.0f;
                previous = soft_reset ? previous_h - spike * threshold
                                      : spike * reset + (1.0f - spike) * previous_h;
            }
            const float factor =
                decay_input ? float(x[i]) - (previous - reset) : -(previous - reset);
            gq = gq + gh * factor;
        }
        gv_init[n] = carry;
        gq_per_neuron[n] = gq;
    }
}
