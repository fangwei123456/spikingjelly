import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from torch.cuda.amp import custom_fwd, custom_bwd
from spikingjelly import configure
from spikingjelly.clock_driven import cu_kernel_opt

try:
    import cupy
except BaseException:
    cupy = None

class DataTypeConvertCUDACode:
    float2bool = r'''
    extern "C" __global__
            void float2bool(const float* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) fs[mem_offset + i]) << i);
                    }
                }
            }
    '''

    half2bool = r'''
    extern "C" __global__
            void half2bool(const half* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) fs[mem_offset + i]) << i);
                    }
                }
            }
    '''

    bool2float = r'''
    extern "C" __global__
            void bool2float(const unsigned char* bs, float* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = (float) (compressed_v % 2);
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    '''

    bool2half = r'''
    extern "C" __global__
            void bool2half(const unsigned char* bs, half* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = (half) (compressed_v % 2);
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    '''

def float_spike_to_bool(spike: torch.Tensor):
    s_dtype = spike.dtype
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.float2bool
        kernel_name = 'float2bool'
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.half2bool
        kernel_name = 'half2bool'
    else:
        raise NotImplementedError

    s_shape = spike.shape

    spike = spike.flatten()
    s_padding = 8 - spike.numel() % 8
    if s_padding != 0:
        spike = F.pad(spike, (0, s_padding))
    device_id = spike.get_device()
    spike_b = torch.zeros([spike.numel() // 8], device=spike.device, dtype=torch.uint8)
    with cupy.cuda.Device(device_id):
        numel = spike_b.numel()
        blocks = cu_kernel_opt.cal_blocks(numel)
        numel = cupy.asarray(numel)
        spike, spike_b, numel = cu_kernel_opt.get_contiguous(spike, spike_b, numel)
        kernel_args = [spike, spike_b, numel]
        kernel = cupy.RawKernel(
            kernel_codes,
            kernel_name,
            options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend
        )
        kernel(
            (blocks,), (configure.cuda_threads,),
            cu_kernel_opt.wrap_args_to_raw_kernel(
                device_id,
                *kernel_args
            )
        )
    return spike_b, s_dtype, s_shape, s_padding

def bool_spike_to_float(spike_b: torch.Tensor, s_dtype: torch.dtype, s_shape: torch.Size, s_padding: int = 0):
    device_id = spike_b.get_device()
    spike = torch.zeros(spike_b.numel() * 8, device=spike_b.device, dtype=s_dtype)
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.bool2float
        kernel_name = 'bool2float'
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.bool2half
        kernel_name = 'bool2half'
    else:
        raise NotImplementedError
    with cupy.cuda.Device(device_id):
        numel = spike_b.numel()
        blocks = cu_kernel_opt.cal_blocks(numel)
        numel = cupy.asarray(numel)
        spike_b, spike, numel = cu_kernel_opt.get_contiguous(spike_b, spike, numel)
        kernel_args = [spike_b, spike, numel]
        kernel = cupy.RawKernel(
            kernel_codes,
            kernel_name,
            options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend
        )
        kernel(
            (blocks,), (configure.cuda_threads,),
            cu_kernel_opt.wrap_args_to_raw_kernel(
                device_id,
                *kernel_args
            )
        )
    if s_padding is not None and s_padding != 0:
        spike = spike[0: spike.numel() - s_padding]
    return spike.reshape(s_shape)

try:
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
except BaseException:
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
                spike_b, s_dtype, s_shape, s_padding = float_spike_to_bool(spike)
                ctx.s_dtype = s_dtype
                ctx.s_shape = s_shape
                ctx.s_padding = s_padding

            ctx.save_for_backward(
                spike_b if ctx.needs_input_grad[1] else None,
                weight if ctx.needs_input_grad[0] else None
            )
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
            spike, weight = ctx.saved_tensors
            spike = bool_spike_to_float(spike, ctx.s_dtype, ctx.s_shape, ctx.s_padding)
            weight = weight.to(grad_output.dtype)
            grad_spike, grad_weight = cpp_wrapper.cudnn_convolution_backward(spike, grad_output, weight, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32, (
                                                                               True,
                                                                               True))

        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            spike, _ = ctx.saved_tensors
            spike = bool_spike_to_float(spike, ctx.s_dtype, ctx.s_shape, ctx.s_padding)
            grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(ctx.weight_shape, grad_output, spike, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32)

        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            _, weight = ctx.saved_tensors
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
                spike_b, s_dtype, s_shape, s_padding = float_spike_to_bool(spike)
                ctx.s_dtype = s_dtype
                ctx.s_shape = s_shape
                ctx.s_padding = s_padding
            ctx.save_for_backward(spike_b if ctx.needs_input_grad[1] else None,
                                  weight if ctx.needs_input_grad[1] else None)
        return F.linear(spike, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output.shape = [N, *, out_features]
        spike, weight = ctx.saved_tensors
        if spike is not None:
            spike = bool_spike_to_float(spike, ctx.s_dtype, ctx.s_shape, ctx.s_padding)

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
