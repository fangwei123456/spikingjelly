#include "../_cuda.cuh"
#include "../native_layout.cuh"

template <class scalar_t, class Layout>
__global__ void if_forward(const scalar_t *x, const float *v_init, scalar_t *spikes,
                           float *voltages, float *charged, long long T, long long N,
                           float threshold, float reset, int soft_reset,
                           int store_v_seq, Layout layout) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float v = v_init[offsets[1]];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float h = v + float(x[layout.template index<0>(i, t, offsets)]);
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
__global__ void if_backward(const scalar_t *gs, const float *gv, const float *charged,
                            scalar_t *gx, float *gv_init, long long T, long long N,
                            float threshold, float reset, int soft_reset,
                            int detach_reset, float alpha, int store_v_seq,
                            Layout layout) {
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
         n += (long long)blockDim.x * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float carry = 0.0f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float h = charged[layout.template index<2>(i, t, offsets)];
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
            gx[layout.template index<3>(i, t, offsets)] = scalar_t(gh);
            carry = gh;
        }
        gv_init[offsets[4]] = carry;
    }
}
