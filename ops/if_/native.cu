#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>

namespace {

using at::Tensor;

void check_parameters(double threshold, std::optional<double> reset, double alpha,
                      int64_t surrogate_id) {
    TORCH_CHECK_VALUE(surrogate_id >= 0 && surrogate_id < 7,
                      "surrogate_id must be in [0, 6]");
    TORCH_CHECK_VALUE(std::isfinite(alpha) && alpha > 0,
                      "alpha must be finite and positive");
    TORCH_CHECK_VALUE(std::isfinite(threshold), "threshold must be finite");
    TORCH_CHECK_VALUE(!reset || std::isfinite(*reset), "reset must be finite");
}

void check_sequence(const Tensor &x) {
    TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
    TORCH_CHECK(x.layout() == at::kStrided, "only strided tensors are supported");
    TORCH_CHECK(x.dim() >= 2 && x.size(0) > 0, "expected [T, ...] with T >= 1");
    TORCH_CHECK(x.numel() > 0, "neuron dimensions must be nonempty");
    TORCH_CHECK((x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                 x.scalar_type() == at::kBFloat16),
                "input must be float32, float16 or bfloat16");
}

std::tuple<Tensor, Tensor, Tensor>
if_forward_cuda(const Tensor &x, const Tensor &v, double threshold,
                std::optional<double> reset, bool detach_reset, double alpha,
                bool store_v_seq, int64_t surrogate_id) {
    check_sequence(x);
    check_parameters(threshold, reset, alpha, surrogate_id);
    TORCH_CHECK(v.device() == x.device(), "input/state devices must match");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "state must have dtype float32");
    TORCH_CHECK(v.layout() == at::kStrided, "only strided tensors are supported");
    TORCH_CHECK(v.sizes() == x.sizes().slice(1), "state shape must match x[0]");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto &input = x;
    const auto &initial = v;
    auto spikes = sj_empty_like(x);
    auto voltages =
        sj_empty_like(store_v_seq ? x : v, at::TensorOptions().dtype(at::kFloat));
    // ponytail: retain full workspace even in inference; specialize if
    // measurements justify it.
    auto charged = sj_empty_like(x, at::TensorOptions().dtype(at::kFloat));
    const int64_t T = x.size(0), N = v.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    {
        sj_launch_layout<5>(
            input, {&input, &initial, &spikes, &voltages, &charged}, [&](auto layout) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                    at::kHalf, at::kBFloat16, input.scalar_type(), "sj_if_forward",
                    [&] {
                        if_forward<scalar_t><<<blocks, 256, 0, stream>>>(
                            input.const_data_ptr<scalar_t>(),
                            initial.const_data_ptr<float>(),
                            spikes.mutable_data_ptr<scalar_t>(),
                            voltages.mutable_data_ptr<float>(),
                            charged.mutable_data_ptr<float>(), T, N, float(threshold),
                            float(reset.value_or(0)), !reset.has_value(), store_v_seq,
                            layout);
                    });
            });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {spikes, voltages, charged};
}

std::tuple<Tensor, Tensor> if_backward_cuda(const Tensor &gs, const Tensor &gv,
                                            const Tensor &h, double threshold,
                                            std::optional<double> reset,
                                            bool detach_reset, double alpha,
                                            bool store_v_seq, int64_t surrogate_id) {
    check_sequence(h);
    check_sequence(gs);
    TORCH_CHECK(h.scalar_type() == at::kFloat && gv.scalar_type() == at::kFloat,
                "workspace and voltage gradient must have dtype float32");
    check_parameters(threshold, reset, alpha, surrogate_id);
    for (const auto &g : {gs, gv}) {
        TORCH_CHECK(g.device() == h.device(), "gradient devices must match workspace");
        TORCH_CHECK(g.layout() == at::kStrided, "only strided tensors are supported");
    }
    TORCH_CHECK(gs.sizes() == h.sizes(), "spike gradient shape must match workspace");
    TORCH_CHECK(gv.sizes() == (store_v_seq ? h.sizes() : h.sizes().slice(1)),
                "voltage gradient shape must match output");
    const c10::cuda::CUDAGuard guard(h.device());
    const auto &grad_s = gs;
    const auto &grad_v = gv;
    const auto &charged = h;
    auto gx = sj_empty_like(h, at::TensorOptions().dtype(grad_s.scalar_type()));
    auto gv_init = sj_empty_state_like(h);
    const int64_t T = h.size(0), N = gv_init.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    const auto stream = at::cuda::getCurrentCUDAStream(h.get_device());
    {
        sj_launch_layout<5>(
            charged, {&grad_s, &grad_v, &charged, &gx, &gv_init}, [&](auto layout) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                    at::kHalf, at::kBFloat16, grad_s.scalar_type(), "sj_if_backward",
                    [&] {
                        sj_dispatch_surrogate(surrogate_id, [&](auto tag) {
                            if_backward<scalar_t, decltype(tag)::value>
                                <<<blocks, 256, 0, stream>>>(
                                    grad_s.const_data_ptr<scalar_t>(),
                                    grad_v.const_data_ptr<float>(),
                                    charged.const_data_ptr<float>(),
                                    gx.mutable_data_ptr<scalar_t>(),
                                    gv_init.mutable_data_ptr<float>(), T, N,
                                    float(threshold), float(reset.value_or(0)),
                                    !reset.has_value(), detach_reset, float(alpha),
                                    store_v_seq, layout);
                        });
                    });
            });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, gv_init};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(sj_if, m) {
    m.def("native_forward(Tensor x, Tensor v, float threshold, float? reset, bool "
          "detach_reset, float alpha, bool store_v_seq=True, int surrogate_id=0) "
          "-> "
          "(Tensor, Tensor, Tensor)");
    m.def("native_backward(Tensor gs, Tensor gv, Tensor h, float threshold, float? "
          "reset, bool detach_reset, float alpha, bool store_v_seq=True, int "
          "surrogate_id=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sj_if, CUDA, m) {
    m.impl("native_forward", &if_forward_cuda);
    m.impl("native_backward", &if_backward_cuda);
}
