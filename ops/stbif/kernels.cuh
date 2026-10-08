#include "../_cuda.cuh"
template <typename scalar_t>
__global__ void stbif_forward(scalar_t const *x, float const *q, float const *acc_q, float const *q_threshold, float const *pos_max, float const *neg_min, scalar_t *out, float *vo, float *wo, float *cur, long long T, long long N) {
    for (long long n=blockIdx.x*static_cast<long long>(blockDim.x)+threadIdx.x;n<N;n+=static_cast<long long>(blockDim.x)*gridDim.x) {
        const float th=q_threshold[0];
        float voltage=q[n], accumulated=acc_q[n], current=0.f;
        for(long long t=0;t<T;++t) {
            const long long i=t*N+n;
            voltage = voltage + float(x[i]) / th;
            accumulated = nearbyintf(accumulated);
            const bool pos = voltage >= 1.f && accumulated < pos_max[0];
            const bool neg = voltage < 0.f && accumulated > neg_min[0];
            current = float(pos) - float(neg);
            accumulated += current;
            voltage = voltage - float(pos) + float(neg);
            out[i] = scalar_t(current * th);
        }
        vo[n]=voltage; wo[n]=accumulated; cur[n]=current;
    }
}
