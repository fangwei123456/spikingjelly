#include "../_cuda.cuh"
#include "../native_layout.cuh"
template <typename scalar_t, class Layout>
__global__ void izhikevich_forward(scalar_t const *x, float const *v, float const *w,
                                   scalar_t *s, float *vo, float *wo, float *h,
                                   float *previous, long long T, long long N, float tau,
                                   float rest, float critical, float a0, float a,
                                   float b, float tau_w, float threshold, float reset,
                                   bool soft, bool trace, Layout layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float voltage = v[offsets[1]], recovery = w[offsets[2]];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float current = float(x[layout.template index<0>(i, t, offsets)]);
            previous[layout.template index<7>(i, t, offsets)] = voltage;
            const float charged =
                voltage +
                (current + a0 * (voltage - rest) * (voltage - critical) - recovery) /
                    tau;
            const float spike = float(charged >= threshold);
            recovery = recovery + (a * (charged - rest) - recovery) / tau_w + b * spike;
            voltage = soft ? charged - spike * threshold
                           : spike * reset + (1.f - spike) * charged;
            s[layout.template index<3>(i, t, offsets)] = scalar_t(spike);
            h[layout.template index<6>(i, t, offsets)] = charged;
            if (trace) {
                vo[layout.template index<4>(i, t, offsets)] = voltage;
                wo[layout.template index<5>(i, t, offsets)] = recovery;
            }
        }
        if (!trace) {
            vo[offsets[4]] = voltage;
            wo[offsets[5]] = recovery;
        }
    }
}
template <typename scalar_t, int Surrogate, class Layout>
__global__ void
izhikevich_backward(scalar_t const *gs, float const *gv, float const *gw,
                    float const *h, float const *previous, scalar_t *gx, float *v0,
                    float *w0, long long T, long long N, float tau, float rest,
                    float critical, float a0, float a, float b, float tau_w,
                    float threshold, float reset, bool soft, bool detach, float alpha,
                    bool trace, Layout layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        float cv = 0.f, cw = 0.f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float charged = h[layout.template index<3>(i, t, offsets)],
                        p = previous[layout.template index<4>(i, t, offsets)];
            const float sg =
                sj_surrogate_gradient<Surrogate>(charged - threshold, alpha);
            float dr = soft ? 1.f : 1.f - float(charged >= threshold);
            if (!detach)
                dr += (soft ? -threshold : reset - charged) * sg;
            const float iv = cv + (trace ? gv[layout.template index<1>(i, t, offsets)]
                                         : (t == T - 1 ? gv[offsets[1]] : 0.f));
            const float iw = cw + (trace ? gw[layout.template index<2>(i, t, offsets)]
                                         : (t == T - 1 ? gw[offsets[2]] : 0.f));
            float gh =
                float(gs[layout.template index<0>(i, t, offsets)]) * sg + iv * dr;
            gh += iw * (b * sg + a / tau_w);
            const float dh = 1.f + a0 * (2.f * p - rest - critical) / tau;
            gx[layout.template index<5>(i, t, offsets)] = scalar_t(gh / tau);
            cv = gh * dh;
            cw = iw * (1.f - 1.f / tau_w) - gh / tau;
        }
        v0[offsets[6]] = cv;
        w0[offsets[7]] = cw;
    }
}
