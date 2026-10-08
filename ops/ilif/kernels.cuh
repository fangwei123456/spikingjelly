#include "../_cuda.cuh"
template <typename scalar_t>
__global__ void ilif_forward(scalar_t const *x, float const *v, scalar_t *s, float *vo, float *h, long long T, long long N, float tau, float count, float threshold, bool trace) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        float voltage = v[n];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float current = float(x[i]);

            const float charged = (1.f - 1.f / tau) * voltage + current;
            const float spike = nearbyintf(fminf(fmaxf(charged / threshold, 0.f), count));

            voltage = charged - spike * threshold;
            s[i] = scalar_t(spike); h[i] = charged;
            if (trace) { vo[i] = voltage;  }
        }
        if (!trace) { vo[n] = voltage;  }
    }
}
template <typename scalar_t>
__global__ void ilif_backward(scalar_t const *gs, float const *gv, float const *h, scalar_t *gx, float *v0, long long T, long long N, float tau, float lower, float upper, float threshold, bool detach, bool trace) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        float cv = 0.f;
        for (long long t = T - 1; t >= 0; --t) {
            const long long i = t * N + n;
            const float charged = h[i];
            const float scaled = charged / threshold;
            const float sg = (scaled >= lower && scaled <= upper) ? 1.f / threshold : 0.f;
            float dr = detach ? 1.f : 1.f - threshold * sg;
            const float iv = cv + (trace ? gv[i] : (t == T - 1 ? gv[n] : 0.f));

            float gh = float(gs[i]) * sg + iv * dr;

            const float dh = 1.f - 1.f / tau;
            gx[i] = scalar_t(gh); cv = gh * dh;

        }
        v0[n] = cv;
    }
}
