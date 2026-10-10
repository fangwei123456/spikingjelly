#include "../_cuda.cuh"
#include "../native_layout.cuh"
template <typename scalar_t, class Layout>
__global__ void stbif_forward(scalar_t const *x, float const *q, float const *acc_q,
                              float const *q_threshold, float const *pos_max,
                              float const *neg_min, scalar_t *out, float *vo, float *wo,
                              float *cur, long long T, long long N, Layout layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        const float th = q_threshold[0];
        float voltage = q[offsets[1]], accumulated = acc_q[offsets[2]], current = 0.f;
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            voltage = voltage + float(x[layout.template index<0>(i, t, offsets)]) / th;
            accumulated = nearbyintf(accumulated);
            const bool pos = voltage >= 1.f && accumulated < pos_max[0];
            const bool neg = voltage < 0.f && accumulated > neg_min[0];
            current = float(pos) - float(neg);
            accumulated += current;
            voltage = voltage - float(pos) + float(neg);
            out[layout.template index<3>(i, t, offsets)] = scalar_t(current * th);
        }
        vo[offsets[4]] = voltage;
        wo[offsets[5]] = accumulated;
        cur[offsets[6]] = current;
    }
}
