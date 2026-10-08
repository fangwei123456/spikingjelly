#include "../_cuda.cuh"
template <typename scalar_t>
__global__ void activation_aware_if_forward(scalar_t const *x, float const *v, float const *threshold, float const *offset, scalar_t *out, float *vo, long long T, long long N, long long channels, long long inner, bool scalar_threshold, bool scalar_offset, float reset, bool soft, bool trace) {
    for (long long n=blockIdx.x*static_cast<long long>(blockDim.x)+threadIdx.x;n<N;n+=static_cast<long long>(blockDim.x)*gridDim.x) {
        const long long c=n/inner%channels; const float th=threshold[scalar_threshold ? 0:c], off=offset[scalar_offset ? 0:c];
        float voltage=v[n];
        for(long long t=0;t<T;++t) {
            const long long i=t*N+n;
            const float h = voltage + float(x[i]);
            const float spike = float(h + off >= th);
            voltage = soft ? h - spike * th : spike * reset + (1.f - spike) * h;
            out[i] = scalar_t(spike);
            if (trace) vo[i] = voltage;
        }
        if(!trace) vo[n]=voltage;
    }
}
