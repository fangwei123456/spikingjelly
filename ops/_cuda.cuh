#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// All state, surrogate and temporal-gradient arithmetic stays in FP32.
template <int Surrogate>
__device__ __forceinline__ float sj_surrogate_gradient(float x, float alpha) {
    if constexpr (Surrogate == 0) {
        const float s = 1.0f / (1.0f + expf(-alpha * x));
        return (1.0f - s) * s * alpha;
    } else if constexpr (Surrogate == 1) {
        const float z = x * (1.5707963267948966f * alpha);
        return (0.5f * alpha) / (1.0f + z * z);
    } else if constexpr (Surrogate == 2) {
        return fmaxf(0.0f, alpha - alpha * alpha * fabsf(x));
    } else if constexpr (Surrogate == 3) {
        return (0.5f * alpha) * expf(-alpha * fabsf(x));
    } else if constexpr (Surrogate == 4) {
        const float z = 1.0f / alpha + fabsf(x);
        return 1.0f / (2.0f * alpha * z * z);
    } else if constexpr (Surrogate == 5) {
        const float z = 1.0f + fabsf(x);
        return alpha / (z * z);
    } else {
        static_assert(Surrogate == 6, "Unsupported surrogate");
        const float z = alpha * x;
        return (0.5641895835477563f * alpha) * expf(-z * z);
    }
}

#ifndef __CUDACC_RTC__
#include <type_traits>
template <class Launch> void sj_dispatch_surrogate(int64_t surrogate, Launch launch) {
    switch (surrogate) {
    case 0:
        launch(std::integral_constant<int, 0>{});
        break;
    case 1:
        launch(std::integral_constant<int, 1>{});
        break;
    case 2:
        launch(std::integral_constant<int, 2>{});
        break;
    case 3:
        launch(std::integral_constant<int, 3>{});
        break;
    case 4:
        launch(std::integral_constant<int, 4>{});
        break;
    case 5:
        launch(std::integral_constant<int, 5>{});
        break;
    case 6:
        launch(std::integral_constant<int, 6>{});
        break;
    }
}
#endif
