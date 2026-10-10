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
void check_parameters(double tau, double rest, double critical, double a0, double a,
                      double b, double tau_w, double threshold,
                      std::optional<double> reset, bool detach_reset, double alpha,
                      bool store_v_seq, int64_t surrogate_id) {
    TORCH_CHECK_VALUE(std::isfinite(tau) && std::isfinite(rest) &&
                          std::isfinite(critical) && std::isfinite(a0) &&
                          std::isfinite(a) && std::isfinite(b) && std::isfinite(tau_w),
                      "izhikevich parameters must be finite");
    TORCH_CHECK_VALUE(tau > 1, "tau must exceed one");
    TORCH_CHECK_VALUE(tau_w > 0, "tau_w must be positive");
    TORCH_CHECK_VALUE(!reset || std::isfinite(*reset), "reset must be finite");
    TORCH_CHECK_VALUE(std::isfinite(alpha) && alpha > 0 && surrogate_id >= 0 &&
                          surrogate_id < 7,
                      "invalid surrogate parameters");
    TORCH_CHECK_VALUE(std::isfinite(threshold), "threshold must be finite");
}
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
forward(const Tensor &source_x, const Tensor &source_v, const Tensor &source_w,
        double tau, double rest, double critical, double a0, double a, double b,
        double tau_w, double threshold, std::optional<double> reset, bool detach_reset,
        double alpha, bool store_v_seq, int64_t surrogate_id) {
    check_sequence(source_x);
    check_parameters(tau, rest, critical, a0, a, b, tau_w, threshold, reset,
                     detach_reset, alpha, store_v_seq, surrogate_id);
    for (const auto &state : {source_v, source_w}) {
        TORCH_CHECK(state.device() == source_x.device() &&
                        state.scalar_type() == at::kFloat &&
                        state.layout() == at::kStrided &&
                        state.sizes() == source_x.sizes().slice(1),
                    "states must be FP32 with input device and x[0] shape");
    }
    const c10::cuda::CUDAGuard guard(source_x.device());
    const auto &x = source_x;
    const auto &v = source_v;
    const auto &w = source_w;
    auto s = sj_empty_like(x);
    auto vo =
        sj_empty_like(store_v_seq ? x : v, at::TensorOptions().dtype(at::kFloat));
    auto wo =
        sj_empty_like(store_v_seq ? x : w, at::TensorOptions().dtype(at::kFloat));
    auto h = sj_empty_like(x, at::TensorOptions().dtype(at::kFloat));
    auto previous = sj_empty_like(x, at::TensorOptions().dtype(at::kFloat));
    const int64_t T = x.size(0), N = v.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    sj_launch_layout<8>(x, {&x, &v, &w, &s, &vo, &wo, &h, &previous}, [&](auto layout) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "sj_izhikevich_forward", [&] {
                izhikevich_forward<scalar_t><<<blocks, 256, 0, stream>>>(
                    x.const_data_ptr<scalar_t>(), v.const_data_ptr<float>(),
                    w.const_data_ptr<float>(), s.mutable_data_ptr<scalar_t>(),
                    vo.mutable_data_ptr<float>(), wo.mutable_data_ptr<float>(),
                    h.mutable_data_ptr<float>(), previous.mutable_data_ptr<float>(), T,
                    N, tau, rest, critical, a0, a, b, tau_w, threshold,
                    reset.value_or(0), !reset, store_v_seq, layout);
            });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {s, vo, wo, h, previous};
}
std::tuple<Tensor, Tensor, Tensor>
backward(const Tensor &source_gs, const Tensor &source_gv, const Tensor &source_gw,
         const Tensor &source_h, const Tensor &source_previous, double tau, double rest,
         double critical, double a0, double a, double b, double tau_w, double threshold,
         std::optional<double> reset, bool detach_reset, double alpha, bool store_v_seq,
         int64_t surrogate_id) {
    check_sequence(source_h);
    check_sequence(source_gs);
    check_parameters(tau, rest, critical, a0, a, b, tau_w, threshold, reset,
                     detach_reset, alpha, store_v_seq, surrogate_id);
    for (const auto &tensor :
         {source_gs, source_gv, source_gw, source_h, source_previous})
        TORCH_CHECK(tensor.device() == source_h.device() &&
                        tensor.layout() == at::kStrided,
                    "gradient/workspace device and layout must match");
    const c10::cuda::CUDAGuard guard(source_h.device());
    const auto &gs = source_gs;
    const auto &gv = source_gv;
    const auto &gw = source_gw;
    const auto &h = source_h;
    const auto &previous = source_previous;
    TORCH_CHECK(h.scalar_type() == at::kFloat && gs.sizes() == h.sizes(),
                "invalid spike gradient or workspace");
    for (const auto &tensor : {gv, gw, previous})
        TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                    "state gradients/workspace must be FP32");
    TORCH_CHECK(gv.sizes() == (store_v_seq ? h.sizes() : h.sizes().slice(1)),
                "invalid voltage gradient shape");
    TORCH_CHECK(previous.sizes() == h.sizes(),
                "invalid previous-voltage workspace shape");
    TORCH_CHECK(gw.sizes() == gv.sizes(), "invalid recovery gradient shape");
    auto gx = sj_empty_like(gs);
    auto v0 = sj_empty_state_like(h);
    auto w0 = sj_empty_like(v0);
    const int64_t T = h.size(0), N = v0.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream(h.get_device());
    sj_launch_layout<8>(h, {&gs, &gv, &gw, &h, &previous, &gx, &v0, &w0}, [&](auto layout) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, gs.scalar_type(), "sj_izhikevich_backward", [&] {
                sj_dispatch_surrogate(surrogate_id, [&](auto tag) {
                    izhikevich_backward<scalar_t, decltype(tag)::value>
                        <<<blocks, 256, 0, stream>>>(
                            gs.const_data_ptr<scalar_t>(), gv.const_data_ptr<float>(),
                            gw.const_data_ptr<float>(), h.mutable_data_ptr<float>(),
                            previous.mutable_data_ptr<float>(),
                            gx.mutable_data_ptr<scalar_t>(),
                            v0.mutable_data_ptr<float>(), w0.mutable_data_ptr<float>(),
                            T, N, tau, rest, critical, a0, a, b, tau_w, threshold,
                            reset.value_or(0), !reset, detach_reset, alpha, store_v_seq,
                            layout);
                });
            });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, v0, w0};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_izhikevich, m) {
    m.def("native_forward(Tensor x, Tensor v, Tensor w, float tau, float rest, "
          "float critical, float a0, float a, float b, float tau_w, float "
          "threshold, float? reset, bool detach_reset, float alpha, bool "
          "store_v_seq, int surrogate_id) -> (Tensor, Tensor, Tensor, Tensor, "
          "Tensor)");
    m.def("native_backward(Tensor gs, Tensor gv, Tensor gw, Tensor h, Tensor "
          "previous, float tau, float rest, float critical, float a0, float a, "
          "float b, float tau_w, float threshold, float? reset, bool "
          "detach_reset, float alpha, bool store_v_seq, int surrogate_id) -> "
          "(Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(sj_izhikevich, CUDA, m) {
    m.impl("native_forward", &forward);
    m.impl("native_backward", &backward);
}
