#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_surrogate.cuh"

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
