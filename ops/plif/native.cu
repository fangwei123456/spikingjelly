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

void check_inputs(const Tensor &x, const Tensor &v, const Tensor &w, double threshold,
                  std::optional<double> reset, double alpha, int64_t surrogate_id) {
    TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
    TORCH_CHECK(x.layout() == at::kStrided && v.layout() == at::kStrided &&
                    w.layout() == at::kStrided,
                "only strided tensors are supported");
    TORCH_CHECK(x.dim() >= 2 && x.size(0) > 0, "expected [T, ...] with T >= 1");
    TORCH_CHECK(x.numel() > 0, "neuron dimensions must be nonempty");
    TORCH_CHECK((x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                 x.scalar_type() == at::kBFloat16) &&
                    v.scalar_type() == at::kFloat &&
                    (w.scalar_type() == at::kFloat || w.scalar_type() == at::kHalf ||
                     w.scalar_type() == at::kBFloat16),
                "input/w must be float32, float16 or bfloat16; state must be float32");
    TORCH_CHECK(v.device() == x.device() && w.device() == x.device(),
                "input, state and w devices must match");
    TORCH_CHECK(v.sizes() == x.sizes().slice(1), "state shape must match x[0]");
    TORCH_CHECK(w.dim() == 0, "w must be a scalar tensor");
    TORCH_CHECK_VALUE(surrogate_id >= 0 && surrogate_id < 7,
                      "surrogate_id must be in [0, 6]");
    TORCH_CHECK_VALUE(std::isfinite(threshold), "threshold must be finite");
    TORCH_CHECK_VALUE(!reset || std::isfinite(*reset), "reset must be finite");
    TORCH_CHECK_VALUE(std::isfinite(alpha) && alpha > 0,
                      "alpha must be finite and positive");
}

std::tuple<Tensor, Tensor, Tensor>
forward(const Tensor &x, const Tensor &v, const Tensor &w, bool decay_input,
        double threshold, std::optional<double> reset, bool detach_reset, double alpha,
        bool store_v_seq, int64_t surrogate_id) {
    check_inputs(x, v, w, threshold, reset, alpha, surrogate_id);
    const c10::cuda::CUDAGuard guard(x.device());
    const auto &input = x;
    const auto &initial = v;
    auto q = w.to(at::kFloat).sigmoid();
    auto spikes = sj_empty_like(x);
    auto voltages =
        sj_empty_like(store_v_seq ? x : v, at::TensorOptions().dtype(at::kFloat));
    auto charged = sj_empty_like(x, at::TensorOptions().dtype(at::kFloat));
    const int64_t T = x.size(0), N = v.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    sj_launch_layout<5>(
        input, {&input, &initial, &spikes, &voltages, &charged}, [&](auto layout) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, input.scalar_type(), "sj_plif_forward", [&] {
                    plif_forward<scalar_t><<<blocks, 256, 0, stream>>>(
                        input.const_data_ptr<scalar_t>(),
                        initial.const_data_ptr<float>(), q.const_data_ptr<float>(),
                        spikes.mutable_data_ptr<scalar_t>(),
                        voltages.mutable_data_ptr<float>(),
                        charged.mutable_data_ptr<float>(), T, N, decay_input,
                        float(threshold), float(reset.value_or(0)), !reset.has_value(),
                        store_v_seq, layout);
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {spikes, voltages, charged};
}

std::tuple<Tensor, Tensor, Tensor>
backward(const Tensor &gs, const Tensor &gv, const Tensor &x, const Tensor &v,
         const Tensor &w, const Tensor &h, bool decay_input, double threshold,
         std::optional<double> reset, bool detach_reset, double alpha, bool store_v_seq,
         int64_t surrogate_id) {
    check_inputs(x, v, w, threshold, reset, alpha, surrogate_id);
    for (const auto &tensor : {gs, gv, h}) {
        TORCH_CHECK(tensor.device() == x.device(),
                    "gradient/workspace devices must match input");
        TORCH_CHECK(tensor.layout() == at::kStrided,
                    "only strided tensors are supported");
    }
    TORCH_CHECK(gs.scalar_type() == x.scalar_type() && gv.scalar_type() == at::kFloat &&
                    h.scalar_type() == at::kFloat,
                "invalid gradient/workspace dtype");
    TORCH_CHECK(gs.sizes() == x.sizes() && h.sizes() == x.sizes(),
                "spike gradient/workspace shapes must match input");
    TORCH_CHECK(gv.sizes() == (store_v_seq ? x.sizes() : v.sizes()),
                "voltage gradient shape must match output");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto &input = x;
    const auto &initial = v;
    const auto &grad_s = gs;
    const auto &grad_v = gv;
    const auto &charged = h;
    auto q = w.to(at::kFloat).sigmoid();
    auto gx = sj_empty_like(x);
    auto gv_init = sj_empty_like(v);
    auto gq = sj_empty_like(v);
    const int64_t T = x.size(0), N = v.numel();
    const int blocks = std::min<int64_t>((N + 255) / 256, 65535);
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    sj_launch_layout<8>(
        charged, {&grad_s, &grad_v, &input, &initial, &charged, &gx, &gv_init, &gq},
        [&](auto layout) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, grad_s.scalar_type(), "sj_plif_backward",
                [&] {
                    sj_dispatch_surrogate(surrogate_id, [&](auto tag) {
                        plif_backward<scalar_t, decltype(tag)::value>
                            <<<blocks, 256, 0, stream>>>(
                                grad_s.const_data_ptr<scalar_t>(),
                                grad_v.const_data_ptr<float>(),
                                input.const_data_ptr<scalar_t>(),
                                initial.const_data_ptr<float>(),
                                q.const_data_ptr<float>(),
                                charged.const_data_ptr<float>(),
                                gx.mutable_data_ptr<scalar_t>(),
                                gv_init.mutable_data_ptr<float>(),
                                gq.mutable_data_ptr<float>(), T, N, decay_input,
                                float(threshold), float(reset.value_or(0)),
                                !reset.has_value(), detach_reset, float(alpha),
                                store_v_seq, layout);
                    });
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gw = (gq.sum() * q * (1 - q)).to(w.scalar_type());
    return {gx, gv_init, gw};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(sj_plif, m) {
    m.def("native_forward(Tensor x, Tensor v, Tensor w, bool decay_input, float "
          "threshold, float? reset, bool detach_reset, float alpha, bool "
          "store_v_seq=True, int surrogate_id=0) -> (Tensor, Tensor, Tensor)");
    m.def("native_backward(Tensor gs, Tensor gv, Tensor x, Tensor v, Tensor w, "
          "Tensor "
          "h, bool decay_input, float threshold, float? reset, bool detach_reset, "
          "float alpha, bool store_v_seq=True, int surrogate_id=0) -> (Tensor, "
          "Tensor, "
          "Tensor)");
}

TORCH_LIBRARY_IMPL(sj_plif, CUDA, m) {
    m.impl("native_forward", &forward);
    m.impl("native_backward", &backward);
}
