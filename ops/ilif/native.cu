#include "kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <optional>
#include <torch/library.h>
#include <tuple>
namespace {
using at::Tensor;
void check_sequence(const Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.layout() == at::kStrided && x.dim() >= 2 &&
                    x.size(0) > 0 && x.numel() > 0,
                "expected nonempty strided CUDA [T, ...]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "unsupported input dtype");
}
void check_parameters(double tau, double count, double lower, double upper,
                      double threshold, bool detach_reset, bool store_v_seq) {
    TORCH_CHECK_VALUE(std::isfinite(tau) && std::isfinite(count) &&
                          std::isfinite(lower) && std::isfinite(upper),
                      "ilif parameters must be finite");
    TORCH_CHECK_VALUE(tau > 1, "tau must exceed one");
    TORCH_CHECK_VALUE(threshold > 0 && count >= 1 && count == std::floor(count) &&
                          lower <= upper,
                      "invalid I-LIF threshold, count, or STE window");
    TORCH_CHECK_VALUE(std::isfinite(threshold), "threshold must be finite");
}
std::tuple<Tensor, Tensor, Tensor> forward(const Tensor &source_x,
                                           const Tensor &source_v, double tau,
                                           double count, double lower, double upper,
                                           double threshold, bool detach_reset,
                                           bool store_v_seq) {
    check_sequence(source_x);
    check_parameters(tau, count, lower, upper, threshold, detach_reset, store_v_seq);
    for (const auto &state : {source_v}) {
        TORCH_CHECK(state.device() == source_x.device() &&
                        state.scalar_type() == at::kFloat &&
                        state.layout() == at::kStrided &&
                        state.sizes() == source_x.sizes().slice(1),
                    "states must be FP32 with input device and x[0] shape");
    }
    const c10::cuda::CUDAGuard guard(source_x.device());
    const auto &x = source_x;
    const auto &v = source_v;
    auto s = sj_empty_like(x);
    auto vo =
        sj_empty_like(store_v_seq ? x : v, at::TensorOptions().dtype(at::kFloat));
    auto h = sj_empty_like(x, at::TensorOptions().dtype(at::kFloat));
    const int64_t T = x.size(0), N = v.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    sj_launch_layout<5>(x, {&x, &v, &s, &vo, &h}, [&](auto layout) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "sj_ilif_forward", [&] {
                ilif_forward<scalar_t><<<blocks, 256, 0, stream>>>(
                    x.const_data_ptr<scalar_t>(), v.const_data_ptr<float>(),
                    s.mutable_data_ptr<scalar_t>(), vo.mutable_data_ptr<float>(),
                    h.mutable_data_ptr<float>(), T, N, tau, count, threshold,
                    store_v_seq, layout);
            });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {s, vo, h};
}
std::tuple<Tensor, Tensor> backward(const Tensor &source_gs, const Tensor &source_gv,
                                    const Tensor &source_h, double tau, double count,
                                    double lower, double upper, double threshold,
                                    bool detach_reset, bool store_v_seq) {
    check_sequence(source_h);
    check_sequence(source_gs);
    check_parameters(tau, count, lower, upper, threshold, detach_reset, store_v_seq);
    for (const auto &tensor : {source_gs, source_gv, source_h})
        TORCH_CHECK(tensor.device() == source_h.device() &&
                        tensor.layout() == at::kStrided,
                    "gradient/workspace device and layout must match");
    const c10::cuda::CUDAGuard guard(source_h.device());
    const auto &gs = source_gs;
    const auto &gv = source_gv;
    const auto &h = source_h;
    TORCH_CHECK(h.scalar_type() == at::kFloat && gs.sizes() == h.sizes(),
                "invalid spike gradient or workspace");
    for (const auto &tensor : {gv})
        TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                    "state gradients/workspace must be FP32");
    TORCH_CHECK(gv.sizes() == (store_v_seq ? h.sizes() : h.sizes().slice(1)),
                "invalid voltage gradient shape");
    auto gx = sj_empty_like(gs);
    auto v0 = sj_empty_state_like(h);
    const int64_t T = h.size(0), N = v0.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream(h.get_device());
    sj_launch_layout<5>(h, {&gs, &gv, &h, &gx, &v0}, [&](auto layout) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, gs.scalar_type(), "sj_ilif_backward", [&] {
                ilif_backward<scalar_t><<<blocks, 256, 0, stream>>>(
                    gs.const_data_ptr<scalar_t>(), gv.const_data_ptr<float>(),
                    h.mutable_data_ptr<float>(), gx.mutable_data_ptr<scalar_t>(),
                    v0.mutable_data_ptr<float>(), T, N, tau, lower, upper, threshold,
                    detach_reset, store_v_seq, layout);
            });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, v0};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_ilif, m) {
    m.def("native_forward(Tensor x, Tensor v, float tau, float count, float "
          "lower, float upper, float threshold, bool detach_reset, bool "
          "store_v_seq) -> (Tensor, Tensor, Tensor)");
    m.def("native_backward(Tensor gs, Tensor gv, Tensor h, float tau, float "
          "count, float lower, float upper, float threshold, bool detach_reset, "
          "bool store_v_seq) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(sj_ilif, CUDA, m) {
    m.impl("native_forward", &forward);
    m.impl("native_backward", &backward);
}
