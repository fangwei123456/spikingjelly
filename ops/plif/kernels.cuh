#include "../_cuda.cuh"
#include "../native_layout.cuh"

template <class scalar_t, class Layout>
__global__ void plif_forward(const scalar_t *x, const float *v_init, const float *q_ptr,
                             scalar_t *spikes, float *voltages, float *charged,
                             long long T, long long N, int decay_input, float threshold,
                             float reset, int soft_reset, int store_v_seq,
                             Layout layout) {
    const float q = q_ptr[0];
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float v = v_init[offsets[1]];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float h =
                decay_input ? v + (float(x[layout.template index<0>(i, t, offsets)]) -
                                   (v - reset)) *
                                      q
                            : v - (v - reset) * q +
                                  float(x[layout.template index<0>(i, t, offsets)]);
            const float spike = h >= threshold ? 1.0f : 0.0f;
            v = soft_reset ? h - spike * threshold : spike * reset + (1.0f - spike) * h;
            spikes[layout.template index<2>(i, t, offsets)] = scalar_t(spike);
            if (store_v_seq)
                voltages[layout.template index<3>(i, t, offsets)] = v;
            charged[layout.template index<4>(i, t, offsets)] = h;
        }
        if (!store_v_seq)
            voltages[offsets[3]] = v;
    }
}

template <class scalar_t, int Surrogate, class Layout>
__global__ void plif_backward(const scalar_t *gs, const float *gv, const scalar_t *x,
                              const float *v_init, const float *q_ptr,
                              const float *charged, scalar_t *gx, float *gv_init,
                              float *gq_per_neuron, long long T, long long N,
                              int decay_input, float threshold, float reset,
                              int soft_reset, int detach_reset, float alpha,
                              int store_v_seq, Layout layout) {
    const float q = q_ptr[0];
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float carry = 0.0f;
        float gq = 0.0f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float h = charged[layout.template index<4>(i, t, offsets)];
            const float sg = sj_surrogate_gradient<Surrogate>(h - threshold, alpha);
            float reset_grad = soft_reset ? 1.0f : 1.0f - float(h >= threshold);
            if (!detach_reset) {
                reset_grad += soft_reset ? -threshold * sg : (reset - h) * sg;
            }
            const float gh = float(gs[layout.template index<0>(i, t, offsets)]) * sg +
                             ((store_v_seq ? gv[layout.template index<1>(i, t, offsets)]
                                           : (t == T - 1 ? gv[offsets[1]] : 0.0f)) +
                              carry) *
                                 reset_grad;
            gx[layout.template index<5>(i, t, offsets)] =
                scalar_t(decay_input ? gh * q : gh);
            carry = gh - gh * q;
            float previous = v_init[offsets[3]];
            if (t > 0) {
                const float previous_h =
                    charged[layout.template index<4>(i - N, t - 1, offsets)];
                const float spike = previous_h >= threshold ? 1.0f : 0.0f;
                previous = soft_reset ? previous_h - spike * threshold
                                      : spike * reset + (1.0f - spike) * previous_h;
            }
            const float factor =
                decay_input ? float(x[layout.template index<2>(i, t, offsets)]) -
                                  (previous - reset)
                            : -(previous - reset);
            gq = gq + gh * factor;
        }
        gv_init[offsets[6]] = carry;
        gq_per_neuron[offsets[7]] = gq;
    }
}
