#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include "kernels.cuh"
namespace {
using at::Tensor;
std::tuple<Tensor, Tensor> forward(const Tensor &source_x, const Tensor &source_v, const Tensor &source_threshold, const Tensor &source_offset, int64_t channels, int64_t inner, std::optional<double> reset, bool store_v_seq) {
    TORCH_CHECK(source_x.is_cuda() && source_x.layout()==at::kStrided && source_x.dim()>=2 && source_x.size(0)>0 && source_x.numel()>0,"expected nonempty strided CUDA [T,...]");
    TORCH_CHECK(source_x.scalar_type()==at::kFloat || source_x.scalar_type()==at::kHalf || source_x.scalar_type()==at::kBFloat16,"unsupported input dtype");
    for(const auto &t : {source_x, source_v, source_threshold, source_offset}) TORCH_CHECK(!t.requires_grad(), "registered inference transitions do not support autograd");
    for(const auto &t : {source_v, source_threshold, source_offset}) TORCH_CHECK(t.device()==source_x.device() && t.scalar_type()==at::kFloat && t.layout()==at::kStrided,"states/parameters must be strided FP32 on the input device");
    TORCH_CHECK(source_v.sizes()==source_x.sizes().slice(1),"invalid state shape");
    TORCH_CHECK_VALUE(channels>0 && inner>0 && (!reset || std::isfinite(*reset)),"invalid channels/inner/reset");
    TORCH_CHECK(source_v.numel()%(channels*inner)==0,"channel dimensions must divide state size");
    TORCH_CHECK((source_threshold.numel()==1 || source_threshold.numel()==channels) && (source_offset.numel()==1 || source_offset.numel()==channels),"invalid parameter size");
    const c10::cuda::CUDAGuard guard(source_x.device());
    auto x=source_x.contiguous();
    auto v=source_v.contiguous();
    auto threshold=source_threshold.contiguous();
    auto offset=source_offset.contiguous();
    auto out=at::empty(x.sizes(),x.options());
    auto vo=at::empty(store_v_seq ? x.sizes():v.sizes(),v.options());
    const int64_t T=x.size(0), N=v.numel();
    const int blocks=std::min<int64_t>((N+255)/256,65535);
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"sj_activation_aware_if_forward",[&]{ activation_aware_if_forward<scalar_t><<<blocks,256,0,stream>>>(x.const_data_ptr<scalar_t>(), v.const_data_ptr<float>(), threshold.const_data_ptr<float>(), offset.const_data_ptr<float>(), out.mutable_data_ptr<scalar_t>(), vo.mutable_data_ptr<float>(), T, N, channels, inner, threshold.numel()==1, offset.numel()==1, reset.value_or(0), !reset, store_v_seq); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, vo};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_activation_aware_if,m) { m.def("native_forward(Tensor x, Tensor v, Tensor threshold, Tensor offset, int channels, int inner, float? reset, bool store_v_seq) -> (Tensor, Tensor)"); }
TORCH_LIBRARY_IMPL(sj_activation_aware_if,CUDA,m) { m.impl("native_forward",&forward); }
