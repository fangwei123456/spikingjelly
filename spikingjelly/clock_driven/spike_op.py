import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from torch.cuda.amp import custom_fwd, custom_bwd
from . import tensor_cache

try:
    import cupy
except BaseException as e:
    print('spikingjelly.clock_driven.spike_op:', e)
    cupy = None


try:
    print('spikingjelly.clock_driven.spike_op:', 'try to use `torch.utils.cpp_extension.load_inline` to load cudnn functions.')
    print(f'If it is hanging, pleast try to delete torch_extensions cache directory. (In most cases, the directory is {torch.utils.cpp_extension._get_build_directory("", False)})')
    cpp_wrapper = load_inline(
            name='cpp_wrapper',
            cpp_sources='using namespace at;',
            functions=[
                'cudnn_convolution_backward',
                'cudnn_convolution_backward_input',
                'cudnn_convolution_backward_weight'
            ],
            with_cuda=True
    )
except BaseException as e:
    print('spikingjelly.clock_driven.spike_op:', e)
    cpp_wrapper = None

'''
aten/src/ATen/native/cudnn/ConvPlaceholders.cpp

at::Tensor cudnn_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)

There are two overloaded C++ methods `cudnn_convolution`. So, we need to use an alternative syntax to cast the overloaded function.
Refer to https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods and https://github.com/pytorch/pytorch/issues/39518 for more details.
    
aten/src/ATen/native/cudnn/ConvShared.cpp

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)

aten/src/ATen/native/cudnn/ConvPlaceholders.cpp

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask)
  
aten/src/ATen/native/cudnn/ConvShared.cpp

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
    
aten/src/ATen/native/cudnn/ConvShared.cpp

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
'''

class spike_convolution(torch.autograd.Function):
    # Pytorch only provides cudnn_convolution without bias.
    # Refer to https://github.com/pytorch/pytorch/issues/3823 for more details.
    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias, stride, padding, dilation, groups):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)

            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(weight)

            ctx.padding = padding
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.weight_shape = weight.shape

        if spike.dim() == 3:
            return F.conv1d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 4:
            return F.conv2d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 5:
            return F.conv3d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)



    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_spike = None
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            weight = weight.to(grad_output.dtype)
            grad_spike, grad_weight = cpp_wrapper.cudnn_convolution_backward(spike, grad_output, weight, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32, (
                                                                               True,
                                                                               True))

        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(ctx.weight_shape, grad_output, spike, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32)

        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            weight = weight.to(grad_output.dtype)
            grad_spike = cpp_wrapper.cudnn_convolution_backward_input(ctx.spike_shape, grad_output, weight, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32)

        if ctx.needs_input_grad[2]:
            # grad_output.shape = [N, C, *]
            out_channels = grad_output.shape[1]
            grad_bias = grad_output.transpose(0, 1).reshape(out_channels, -1).sum(1)
        return grad_spike, grad_weight, grad_bias, None, None, None, None

class spike_linear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias=None):
        # spike.shape = [N, *, in_features]
        # weight.shape = [out_features, in_features]
        # bias.shape = [out_features]
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)
            if ctx.needs_input_grad[1]:
                ctx.save_for_backward(weight)
        return F.linear(spike, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output.shape = [N, *, out_features]
        if ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)

        grad_spike = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_spike = F.linear(grad_output, weight.t(), bias=None)
        if ctx.needs_input_grad[1]:
            in_features = spike.shape[-1]
            out_features = grad_output.shape[-1]
            # grad_output.reshape(-1, out_features).t().shape = [out_features, N*]
            # spike.reshape(-1, in_features).shape = [N*, in_features]
            grad_weight = torch.mm(grad_output.reshape(-1, out_features).t(), spike.reshape(-1, in_features).to(grad_output.dtype))
        if ctx.needs_input_grad[2]:
            out_features = grad_output.shape[-1]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        return grad_spike, grad_weight, grad_bias
