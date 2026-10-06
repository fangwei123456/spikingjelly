#include "../_cuda.cuh"
template <typename scalar_t>
__global__ void qif_forward(scalar_t const *x, float const *v, scalar_t *s, float *vo, float *h, float *previous, long long T, long long N, float tau, float rest, float critical, float a0, float threshold, float reset, bool soft, bool trace) {
    // Match PyTorch scalar division: multiply by a rounded FP32 reciprocal.
    const float inv_tau = 1.f / tau;
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        float voltage = v[n];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float current = float(x[i]);
            previous[i] = voltage;
            const float charged = voltage + (current + a0 * (voltage - rest) * (voltage - critical)) * inv_tau;
            const float spike = float(charged >= threshold);

            voltage = soft ? charged - spike * threshold : spike * reset + (1.f - spike) * charged;
            s[i] = scalar_t(spike); h[i] = charged;
            if (trace) { vo[i] = voltage;  }
        }
        if (!trace) { vo[n] = voltage;  }
    }
}
template <typename scalar_t, int Surrogate>
__global__ void qif_backward(scalar_t const *gs, float const *gv, float const *h, float const *previous, scalar_t *gx, float *v0, long long T, long long N, float tau, float rest, float critical, float a0, float threshold, float reset, bool soft, bool detach, float alpha, bool trace) {
    const float inv_tau = 1.f / tau;
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        float cv = 0.f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float charged = h[i], p = previous[i];
            const float sg = sj_surrogate_gradient<Surrogate>(charged - threshold, alpha);
            float dr = soft ? 1.f : 1.f - float(charged >= threshold);
            if (!detach) dr += (soft ? -threshold : reset - charged) * sg;
            const float iv = cv + (trace ? gv[i] : (t == T - 1 ? gv[n] : 0.f));

            float gh = float(gs[i]) * sg + iv * dr;

            const float dh = 1.f + a0 * (2.f * p - rest - critical) * inv_tau;
            gx[i] = scalar_t(gh * inv_tau); cv = gh * dh;

        }
        v0[n] = cv;
    }
}
