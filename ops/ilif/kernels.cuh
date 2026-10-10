#include "../_cuda.cuh"
#include "../native_layout.cuh"
template <typename scalar_t, class Layout>
__global__ void ilif_forward(scalar_t const *x, float const *v, scalar_t *s, float *vo,
                             float *h, long long T, long long N, float tau, float count,
                             float threshold, bool trace, Layout layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float voltage = v[offsets[1]];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float current = float(x[layout.template index<0>(i, t, offsets)]);

            const float charged = (1.f - 1.f / tau) * voltage + current;
            const float spike =
                nearbyintf(fminf(fmaxf(charged / threshold, 0.f), count));

            voltage = charged - spike * threshold;
            s[layout.template index<2>(i, t, offsets)] = scalar_t(spike);
            h[layout.template index<4>(i, t, offsets)] = charged;
            if (trace) {
                vo[layout.template index<3>(i, t, offsets)] = voltage;
            }
        }
        if (!trace) {
            vo[offsets[3]] = voltage;
        }
    }
}
template <typename scalar_t, class Layout>
__global__ void ilif_backward(scalar_t const *gs, float const *gv, float const *h,
                              scalar_t *gx, float *v0, long long T, long long N,
                              float tau, float lower, float upper, float threshold,
                              bool detach, bool trace, Layout layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float cv = 0.f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float charged = h[layout.template index<2>(i, t, offsets)];
            const float scaled = charged / threshold;
            const float sg =
                (scaled >= lower && scaled <= upper) ? 1.f / threshold : 0.f;
            float dr = detach ? 1.f : 1.f - threshold * sg;
            const float iv = cv + (trace ? gv[layout.template index<1>(i, t, offsets)]
                                         : (t == T - 1 ? gv[offsets[1]] : 0.f));

            float gh =
                float(gs[layout.template index<0>(i, t, offsets)]) * sg + iv * dr;

            const float dh = 1.f - 1.f / tau;
            gx[layout.template index<3>(i, t, offsets)] = scalar_t(gh);
            cv = gh * dh;
        }
        v0[offsets[4]] = cv;
    }
}
