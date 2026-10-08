#include "kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {
using at::Tensor;
int dtype_id(const Tensor &x) {
    if (x.scalar_type() == at::kFloat)
        return 0;
    if (x.scalar_type() == at::kHalf)
        return 1;
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "input must be FP32, FP16 or BF16");
    return 2;
}
void check(const Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "expected contiguous 2D CUDA tensor");
    TORCH_CHECK(x.numel() <= 2147483647, "CUDA element limit exceeded");
}
Tensor pack(const Tensor &x) {
    check(x);
    const int type = dtype_id(x);
    const int M = x.size(0), K = x.size(1), P = (K + 7) / 8;
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({M, P}, x.options().dtype(at::kByte));
    if (M && P) {
        auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
        pack_kernel<<<M, std::min(P, 256), 0, stream>>>(
            x.const_data_ptr(), out.mutable_data_ptr<unsigned char>(), M, K, P, type);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
Tensor packed(const Tensor &x, const Tensor &w) {
    check(x);
    check(w);
    const int type = dtype_id(w);
    TORCH_CHECK(x.scalar_type() == at::kByte && x.device() == w.device(),
                "packed input dtype/device mismatch");
    const int M = x.size(0), N = w.size(0), K = w.size(1), P = x.size(1);
    TORCH_CHECK(P == (K + 7) / 8 && int64_t(M) * N <= 2147483647,
                "packed shape/element limit mismatch");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({M, N}, w.options());
    if (M && N) {
        if (!K)
            return out.zero_();
        auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
        dim3 block(16, 8);
        int grid = ((N + 127) / 128) * ((M + 63) / 64);
        if (type == 0)
            spike_linear_v3_tiled_kernel_fp32<<<grid, block, 0, stream>>>(
                x.const_data_ptr<unsigned char>(), w.const_data_ptr<float>(),
                out.mutable_data_ptr<float>(), M, N, K, P);
        else if (type == 1)
            spike_linear_v3_tiled_kernel_fp16<<<grid, block, 0, stream>>>(
                x.const_data_ptr<unsigned char>(),
                static_cast<const __half *>(w.const_data_ptr()),
                static_cast<__half *>(out.mutable_data_ptr()), M, N, K, P);
        else
            spike_linear_v3_tiled_kernel_bf16<<<grid, block, 0, stream>>>(
                x.const_data_ptr<unsigned char>(),
                static_cast<const __nv_bfloat16 *>(w.const_data_ptr()),
                static_cast<__nv_bfloat16 *>(out.mutable_data_ptr()), M, N, K, P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
Tensor sparse(const Tensor &x, const Tensor &w) {
    check(x);
    check(w);
    const int type = dtype_id(x);
    TORCH_CHECK(x.scalar_type() == w.scalar_type() && x.device() == w.device(),
                "sparse input/weight dtype/device mismatch");
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(K == w.size(1) && int64_t(M) * N <= 2147483647,
                "sparse shape/element limit mismatch");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({M, N}, x.options());
    if (M && N) {
        auto counts = at::zeros({M}, x.options().dtype(at::kInt));
        auto indices = at::empty({M, K}, x.options().dtype(at::kInt));
        auto wt = w.t().contiguous();
        auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
        if (K) {
            if (type == 0)
                spike_to_row_indices_kernel_fp32<<<M, 1, 0, stream>>>(
                    x.const_data_ptr<float>(), counts.mutable_data_ptr<int>(),
                    indices.mutable_data_ptr<int>(), M, K);
            else if (type == 1)
                spike_to_row_indices_kernel_fp16<<<M, 1, 0, stream>>>(
                    static_cast<const __half *>(x.const_data_ptr()),
                    counts.mutable_data_ptr<int>(), indices.mutable_data_ptr<int>(), M,
                    K);
            else
                spike_to_row_indices_kernel_bf16<<<M, 1, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(x.const_data_ptr()),
                    counts.mutable_data_ptr<int>(), indices.mutable_data_ptr<int>(), M,
                    K);
        }
        int grid = M * ((N + 255) / 256);
        if (type == 0)
            spike_linear_v15_sparse_wT_kernel_fp32<<<grid, 256, 0, stream>>>(
                counts.const_data_ptr<int>(), indices.const_data_ptr<int>(),
                wt.const_data_ptr<float>(), out.mutable_data_ptr<float>(), M, N, K);
        else if (type == 1)
            spike_linear_v15_sparse_wT_kernel_fp16<<<grid, 256, 0, stream>>>(
                counts.const_data_ptr<int>(), indices.const_data_ptr<int>(),
                static_cast<const __half *>(wt.const_data_ptr()),
                static_cast<__half *>(out.mutable_data_ptr()), M, N, K);
        else
            spike_linear_v15_sparse_wT_kernel_bf16<<<grid, 256, 0, stream>>>(
                counts.const_data_ptr<int>(), indices.const_data_ptr<int>(),
                static_cast<const __nv_bfloat16 *>(wt.const_data_ptr()),
                static_cast<__nv_bfloat16 *>(out.mutable_data_ptr()), M, N, K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_spike_linear, m) {
    m.def("kernel_pack_rows(Tensor x) -> Tensor");
    m.def("kernel_packed(Tensor x,Tensor w) -> Tensor");
    m.def("kernel_sparse(Tensor x,Tensor w) -> Tensor");
}
TORCH_LIBRARY_IMPL(sj_spike_linear, CUDA, m) {
    m.impl("kernel_pack_rows", &pack);
    m.impl("kernel_packed", &packed);
    m.impl("kernel_sparse", &sparse);
}
