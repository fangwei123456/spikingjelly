#pragma once
#include <cuda_runtime.h>
#include <cstdint>

__global__ void few_spike_native_kernel(
    const float *x, const float *theta, const float *h, const float *d,
    const float *other_theta, const float *other_h, const float *other_d,
    const float *gates, float *out, int64_t n, int64_t k_count,
    int64_t regions, int route, float split, float clamp, bool has_clamp,
    bool sequence) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        float v = x[i], sign = 1.0f;
        const float *th = theta, *reset = h, *weight = d;
        if (route == 1) {
            if (has_clamp)
                v = v < -clamp ? -clamp : (v > clamp ? clamp : v);
            sign = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
            v = fabsf(v);
            if (!(v <= split)) {
                th = other_theta;
                reset = other_h;
                weight = other_d;
            }
        } else if (route == 2) {
            int64_t region = 0;
            for (int64_t boundary = 0; boundary < regions - 1; ++boundary)
                region += v > gates[boundary];
            if (isnan(v)) region = regions - 1;
            th += region * k_count;
            reset += region * k_count;
            weight += region * k_count;
        }
        float readout = 0.0f;
        for (int64_t k = 0; k < k_count; ++k) {
            const float spike = __fsub_rn(v, th[k]) >= 0.0f ? 1.0f : 0.0f;
            float value = __fmul_rn(weight[k], spike);
            if (sequence) {
                if (route == 1) value = __fmul_rn(value, sign);
                out[k * n + i] = value;
            } else {
                readout = __fadd_rn(readout, value);
            }
            v = __fsub_rn(v, __fmul_rn(reset[k], spike));
        }
        if (!sequence)
            out[i] = route == 1 ? __fmul_rn(readout, sign) : readout;
    }
}
