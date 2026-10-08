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
std::tuple<Tensor, Tensor, Tensor, Tensor> forward(const Tensor &source_x, const Tensor &source_q, const Tensor &source_acc_q, const Tensor &source_q_threshold, const Tensor &source_pos_max, const Tensor &source_neg_min) {
    TORCH_CHECK(source_x.is_cuda() && source_x.layout()==at::kStrided && source_x.dim()>=2 && source_x.size(0)>0 && source_x.numel()>0,"expected nonempty strided CUDA [T,...]");
    TORCH_CHECK(source_x.scalar_type()==at::kFloat || source_x.scalar_type()==at::kHalf || source_x.scalar_type()==at::kBFloat16,"unsupported input dtype");
    for(const auto &t : {source_x, source_q, source_acc_q, source_q_threshold, source_pos_max, source_neg_min}) TORCH_CHECK(!t.requires_grad(), "registered inference transitions do not support autograd");
    for(const auto &t : {source_q, source_acc_q, source_q_threshold, source_pos_max, source_neg_min}) TORCH_CHECK(t.device()==source_x.device() && t.scalar_type()==at::kFloat && t.layout()==at::kStrided,"states/parameters must be strided FP32 on the input device");
    TORCH_CHECK(source_q.sizes()==source_x.sizes().slice(1),"invalid state shape");
    TORCH_CHECK(source_acc_q.sizes()==source_q.sizes(),"invalid accumulated state shape");
    TORCH_CHECK(source_q_threshold.numel()==1 && source_pos_max.numel()==1 && source_neg_min.numel()==1,"parameters must be scalar tensors");
    const c10::cuda::CUDAGuard guard(source_x.device());
    auto x=source_x.contiguous();
    auto q=source_q.contiguous();
    auto acc_q=source_acc_q.contiguous();
    auto q_threshold=source_q_threshold.contiguous();
    auto pos_max=source_pos_max.contiguous();
    auto neg_min=source_neg_min.contiguous();
    auto out=at::empty(x.sizes(),x.options());
    auto vo=at::empty(q.sizes(),q.options()), wo=at::empty_like(vo), cur=at::empty_like(vo);
    const int64_t T=x.size(0), N=q.numel();
    const int blocks=std::min<int64_t>((N+255)/256,65535);
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"sj_stbif_forward",[&]{ stbif_forward<scalar_t><<<blocks,256,0,stream>>>(x.const_data_ptr<scalar_t>(), q.const_data_ptr<float>(), acc_q.const_data_ptr<float>(), q_threshold.const_data_ptr<float>(), pos_max.const_data_ptr<float>(), neg_min.const_data_ptr<float>(), out.mutable_data_ptr<scalar_t>(), vo.mutable_data_ptr<float>(), wo.mutable_data_ptr<float>(), cur.mutable_data_ptr<float>(), T, N); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, vo, wo, cur};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(sj_stbif,m) { m.def("native_forward(Tensor x, Tensor q, Tensor acc_q, Tensor q_threshold, Tensor pos_max, Tensor neg_min) -> (Tensor, Tensor, Tensor, Tensor)"); }
TORCH_LIBRARY_IMPL(sj_stbif,CUDA,m) { m.impl("native_forward",&forward); }
