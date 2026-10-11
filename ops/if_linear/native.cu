#include "../_cuda.cuh"
#include "kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/library.h>
namespace {
using at::Tensor;
void check(const Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(),
                "expected contiguous CUDA FP32 tensor");
}
template <bool Soft>
std::tuple<Tensor, Tensor>
forward_impl(const Tensor &x, const Tensor &v, const Tensor &w,
             const std::optional<Tensor> &bias, double threshold, double reset,
             int64_t threads) {
    const int64_t T = x.size(0), M = x.size(1), K = x.size(2), N = w.size(1);
    auto y = at::empty({T, M, N}, x.options());
    auto out = at::empty_like(v);
    const int groups = (N + threads * 4 - 1) / (threads * 4);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    if_linear_kernel<Soft><<<M * groups, threads, K * 4 + threads / 8, stream>>>(
        x.const_data_ptr<float>(), v.const_data_ptr<float>(), w.const_data_ptr<float>(),
        bias ? bias->const_data_ptr<float>() : y.const_data_ptr<float>(),
        y.mutable_data_ptr<float>(), out.mutable_data_ptr<float>(), T, M, K, N,
        float(threshold), float(reset), bool(bias));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, out};
}
std::tuple<Tensor, Tensor> forward(const Tensor &x, const Tensor &v, const Tensor &w,
                                   const std::optional<Tensor> &bias, double threshold,
                                   double reset, bool soft, int64_t threads) {
    check(x);
    check(v);
    check(w);
    if (bias)
        check(*bias);
    TORCH_CHECK(x.dim() == 3 && v.dim() == 2 && w.dim() == 2,
                "invalid fused projection dimensions");
    TORCH_CHECK(v.device() == x.device() && w.device() == x.device() &&
                    (!bias || bias->device() == x.device()),
                "device mismatch");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && w.size(1) > 0,
                "empty fused projection");
    TORCH_CHECK(v.sizes() == x.sizes().slice(1) && w.size(0) == x.size(2),
                "shape mismatch");
    TORCH_CHECK(!bias || (bias->dim() == 1 && bias->size(0) == w.size(1)),
                "bias shape mismatch");
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid threads");
    TORCH_CHECK(x.numel() <= 2147483647 &&
                    x.size(0) * x.size(1) * w.size(1) <= 2147483647,
                "CUDA element limit exceeded");
    TORCH_CHECK(x.size(2) * 4 + threads / 8 <=
                    at::cuda::getDeviceProperties(x.get_device())->sharedMemPerBlock,
                "shared-memory limit exceeded");
    const c10::cuda::CUDAGuard guard(x.device());
    if (soft)
        return forward_impl<true>(x, v, w, bias, threshold, reset, threads);
    return forward_impl<false>(x, v, w, bias, threshold, reset, threads);
}
template <bool Soft>
std::tuple<Tensor, Tensor> remat_impl(const Tensor &x, const Tensor &v,
                                      double threshold, double reset) {
    auto spikes = at::empty_like(x);
    auto h = at::empty_like(x);
    const auto MK = v.numel();
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    rematerialize<Soft><<<std::min<int64_t>((MK + 255) / 256, 65535), 256, 0, stream>>>(
        x.const_data_ptr<float>(), v.const_data_ptr<float>(),
        spikes.mutable_data_ptr<float>(), h.mutable_data_ptr<float>(), x.size(0), MK,
        float(threshold), float(reset));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {spikes, h};
}
std::tuple<Tensor, Tensor> remat(const Tensor &x, const Tensor &v, double threshold,
                                 double reset, bool soft) {
    check(x);
    check(v);
    TORCH_CHECK(x.dim() == 3 && v.sizes() == x.sizes().slice(1) &&
                    x.device() == v.device(),
                "rematerialization shape/device mismatch");
    const c10::cuda::CUDAGuard guard(x.device());
    if (soft)
        return remat_impl<true>(x, v, threshold, reset);
    return remat_impl<false>(x, v, threshold, reset);
}
template <int Surrogate, bool Soft>
void backward_impl(const Tensor &gs, const Tensor &gv, const Tensor &h,
                   const Tensor &sg, Tensor &gx, Tensor &g0, double threshold,
                   double reset, bool detach, double alpha) {
    const auto MK = gv.numel();
    auto stream = at::cuda::getCurrentCUDAStream(h.get_device());
    neuron_backward<Surrogate, Soft>
        <<<std::min<int64_t>((MK + 255) / 256, 65535), 256, 0, stream>>>(
            gs.const_data_ptr<float>(), gv.const_data_ptr<float>(),
            h.const_data_ptr<float>(), sg.const_data_ptr<float>(),
            gx.mutable_data_ptr<float>(), g0.mutable_data_ptr<float>(), h.size(0), MK,
            float(threshold), float(reset), detach, float(alpha));
}
std::tuple<Tensor, Tensor> backward(const Tensor &gs, const Tensor &gv, const Tensor &h,
                                    const Tensor &sg, double threshold, double reset,
                                    bool soft, bool detach, int64_t surrogate,
                                    double alpha) {
    check(gs);
    check(gv);
    check(h);
    check(sg);
    TORCH_CHECK(gs.sizes() == h.sizes() && gv.sizes() == h.sizes().slice(1) &&
                    sg.sizes() == h.sizes(),
                "backward shape mismatch");
    TORCH_CHECK(gs.device() == h.device() && gv.device() == h.device() &&
                    sg.device() == h.device(),
                "backward device mismatch");
    const c10::cuda::CUDAGuard guard(h.device());
    auto gx = at::empty_like(h);
    auto g0 = at::empty_like(gv);
    auto launch = [&](auto tag) {
        if (soft == true)
            backward_impl<decltype(tag)::value, true>(gs, gv, h, sg, gx, g0, threshold,
                                                      reset, detach, alpha);
        if (soft == false)
            backward_impl<decltype(tag)::value, false>(gs, gv, h, sg, gx, g0, threshold,
                                                       reset, detach, alpha);
    };
    if (surrogate == -1)
        launch(std::integral_constant<int, -1>{});
    else
        sj_dispatch_surrogate(surrogate, launch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, g0};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_if_linear, m) {
    m.def("kernel_forward(Tensor x,Tensor v,Tensor w,Tensor? bias,float "
          "threshold,float reset,bool soft,int threads) -> (Tensor,Tensor)");
    m.def("rematerialize(Tensor x,Tensor v,float threshold,float reset,bool soft) -> "
          "(Tensor,Tensor)");
    m.def(
        "neuron_backward(Tensor gs,Tensor gv,Tensor h,Tensor sg,float threshold,float "
        "reset,bool soft,bool detach,int surrogate,float alpha) -> (Tensor,Tensor)");
}
TORCH_LIBRARY_IMPL(sj_if_linear, CUDA, m) {
    m.impl("kernel_forward", &forward);
    m.impl("rematerialize", &remat);
    m.impl("neuron_backward", &backward);
}
