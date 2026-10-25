#include "kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/library.h>

namespace {
using at::Tensor;
Tensor forward(const Tensor &x, const Tensor &theta, const Tensor &h,
               const Tensor &d, bool sequence,
               const std::optional<Tensor> &other_theta,
               const std::optional<Tensor> &other_h,
               const std::optional<Tensor> &other_d,
               const std::optional<Tensor> &gates, double split,
               std::optional<double> clamp) {
    TORCH_CHECK(x.is_cuda() && x.numel() > 0,
                "few-spike native inference requires nonempty CUDA input");
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "few-spike native inference requires no_grad");
    auto check = [&](const Tensor &t) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == at::kFloat &&
                        t.layout() == at::kStrided && t.is_contiguous() && !t.is_neg(),
                    "few-spike native tensors must be contiguous FP32 without lazy negative bits "
                    "on one CUDA device");
    };
    for (const auto &t : {x, theta, h, d}) check(t);
    TORCH_CHECK(theta.dim() == (gates ? 2 : 1) && theta.numel() > 0 &&
                    theta.sizes() == h.sizes() && theta.sizes() == d.sizes(),
                "incompatible few-spike table shapes");
    const bool outlier = other_theta.has_value();
    TORCH_CHECK(other_h.has_value() == outlier && other_d.has_value() == outlier &&
                    !(outlier && gates), "incompatible few-spike routing tables");
    if (outlier) {
        for (const auto &t : {*other_theta, *other_h, *other_d}) {
            check(t);
            TORCH_CHECK(t.sizes() == theta.sizes(), "outlier table shape must match");
        }
    }
    if (gates) {
        check(*gates);
        TORCH_CHECK(gates->dim() == 1 && gates->numel() == theta.size(0) - 1,
                    "HG threshold count must match regions");
    }
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t n = x.numel(), k = theta.size(-1);
    auto shape = x.sizes().vec();
    if (sequence) shape.insert(shape.begin(), k);
    auto output = at::empty(shape, x.options());
    const int blocks = std::min<int64_t>((n + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    few_spike_native_kernel<<<blocks, 256, 0, stream>>>(
        x.const_data_ptr<float>(), theta.const_data_ptr<float>(),
        h.const_data_ptr<float>(), d.const_data_ptr<float>(),
        outlier ? other_theta->const_data_ptr<float>() : nullptr,
        outlier ? other_h->const_data_ptr<float>() : nullptr,
        outlier ? other_d->const_data_ptr<float>() : nullptr,
        gates ? gates->const_data_ptr<float>() : nullptr,
        output.mutable_data_ptr<float>(), n, k, gates ? theta.size(0) : 1,
        gates ? 2 : (outlier ? 1 : 0), static_cast<float>(split),
        static_cast<float>(clamp.value_or(0.0)), clamp.has_value(), sequence);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_few_spike, m) {
    m.def("native_forward(Tensor gate, Tensor theta, Tensor h, Tensor d, bool return_sequence, "
          "Tensor? other_theta=None, Tensor? other_h=None, Tensor? other_d=None, "
          "Tensor? gates=None, float split_threshold=0., float? clamp_value=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(sj_few_spike, CUDA, m) { m.impl("native_forward", &forward); }
