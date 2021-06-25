try:
    import cupy
    import torch
    from . import cu_kernel_opt
    from . import surrogate

    # IFNode----------------------------------------------------------------------------------------------------------
    IFNode_forward_hardReset_fp32 = cupy.RawKernel(r'''
    extern "C" __global__
    void IFNode_forward_hardReset_fp32(const float* x, const float* v_last, float* h, float* spike, float* v, 
    const float & v_threshold, const float & v_reset,
    const int & numel) 
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
    {
        h[index] = v_last[index] + x[index];
        if (h[index] >= v_threshold)
        {
            spike[index] = 1.0f;
            v[index] = v_reset;
        }
        else
        {
            spike[index] = 0.0f;
            v[index] = h[index];
        }
    }
    }
    ''', 'IFNode_forward_hardReset_fp32', options=cu_kernel_opt.nvcc_options)

    def create_kernel_IFNode_backward(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):
        if hard_reset:
            if dtype == 'float':
                code = r'''
                extern "C" __global__
                void IFNode_backward_hardReset_fp32(
                const float* grad_s, const float* grad_v, const float* h, const float* spike,
                float* grad_x, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & numel)
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < numel)
                {
                    const float over_th = h[index] - v_threshold;
                '''

                code += sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

                if detach_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike[index];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike[index] + (-h[index] + v_reset) * grad_s_to_h;
                    '''
                code += code_grad_v_to_h
                code += r'''
                    const float grad_h = grad_s[index] * grad_s_to_h + grad_v[index] * grad_v_to_h;
                    grad_x[index] = grad_h;
                    grad_v_last[index] = grad_h;
                }
                }
                '''
                return cupy.RawKernel(code, 'IFNode_backward_hardReset_fp32', options=cu_kernel_opt.nvcc_options)


    IFNode_forward_hardReset_fp16 = cupy.RawKernel(r'''
    #include <cuda_fp16.h>
    extern "C" __global__
    void IFNode_forward_hardReset_fp16(const half* x, const half* v_last, half* h, half* spike, half* v, 
    const half & v_threshold, const half & v_reset,
    const int & numel) 
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
    {
        h[index] = __hadd(v_last[index], x[index]);
        if (__hgeu(h[index], v_threshold))
        {
            spike[index] = __float2half(1.0f);
            v[index] = v_reset;
        }
        else
        {
            spike[index] = __float2half(0.0f);
            v[index] = h[index];
        }
    }
    }
    ''', 'IFNode_forward_hardReset_fp16', options=cu_kernel_opt.nvcc_options)

    IFNode_forward_softReset_fp32 = cupy.RawKernel(r'''
    extern "C" __global__
    void IFNode_forward_softReset_fp32(const float* x, const float* v_last, float* h, float* spike, float* v, 
    const float & v_threshold,
    const int & numel) 
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
    {
        h[index] = v_last[index] + x[index];
        if (h[index] >= v_threshold)
        {
            spike[index] = 1.0f;
            v[index] = h[index] - v_threshold;
        }
        else
        {
            spike[index] = 0.0f;
            v[index] = h[index];
        }
    }
    }
    ''', 'IFNode_forward_softReset_fp32', options=cu_kernel_opt.nvcc_options)

    IFNode_forward_softReset_fp16 = cupy.RawKernel(r'''
    #include <cuda_fp16.h>
    extern "C" __global__
    void IFNode_forward_softReset_fp16(const half* x, const half* v_last, half* h, half* spike, half* v, 
    const half & v_threshold,
    const int & numel) 
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
    {
        h[index] = __hadd(v_last[index], x[index]);
        if (__hgeu(h[index], v_threshold))
        {
            spike[index] = __float2half(1.0f);
            v[index] = __hsub(v[index], v_threshold);
        }
        else
        {
            spike[index] = __float2half(0.0f);
            v[index] = h[index];
        }
    }
    }
    ''', 'IFNode_forward_softReset_fp16', options=cu_kernel_opt.nvcc_options)


    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, v_last: torch.Tensor, v_threshold: float, v_reset: float, detach_reset: bool, sg_cuda_code_fun):
            device = x.get_device()
            if x.dtype == torch.float32:
                cp_dtype = cupy.float32
            elif x.dtype == torch.float16:
                cp_dtype = cupy.float16
            else:
                raise NotImplementedError

            v = torch.zeros_like(v_last.data)
            h = torch.zeros_like(v_last.data)
            spike = torch.zeros_like(v_last.data)

            with cupy.cuda.Device(device):
                numel = x.numel()
                threads = cu_kernel_opt.threads
                blocks = cu_kernel_opt.cal_blocks(numel)
                numel = cupy.asarray(numel)
                cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
                if v_reset is not None:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                else:
                    cp_v_reset = None

                cu_kernel_opt.check_contiguous(x, v_last, h, spike, v)
                if v_reset is None:
                    cu_kernel_opt.check_device(device, x, v_last, h, spike, v, cp_v_threshold, numel)
                    # soft reset
                    if x.dtype == torch.float32:
                        kernel = IFNode_forward_softReset_fp32
                    elif x.dtype == torch.float16:
                        kernel = IFNode_forward_softReset_fp16
                    else:
                        raise NotImplementedError
                    kernel(
                        (blocks,), (threads,),
                        (x.data_ptr(), v_last.data_ptr(), h.data_ptr(), spike.data_ptr(), v.data_ptr(),
                         cp_v_threshold,
                         numel)
                    )

                else:
                    cu_kernel_opt.check_device(device, x, v_last, h, spike, v, cp_v_threshold, cp_v_reset, numel)
                    # hard reset
                    if x.dtype == torch.float32:
                        kernel = IFNode_forward_hardReset_fp32
                    elif x.dtype == torch.float16:
                        kernel = IFNode_forward_hardReset_fp16
                    else:
                        raise NotImplementedError
                    kernel(
                        (blocks,), (threads,),
                        (x.data_ptr(), v_last.data_ptr(), h.data_ptr(), spike.data_ptr(), v.data_ptr(),
                         cp_v_threshold,
                         cp_v_reset, numel)
                    )

            if x.requires_grad or v.requires_grad:
                ctx.save_for_backward(h, spike)
                ctx.blocks = blocks
                ctx.threads = threads
                ctx.numel = numel
                ctx.cp_v_threshold = cp_v_threshold
                ctx.cp_v_reset = cp_v_reset
                ctx.detach_reset = detach_reset
                ctx.sg_cuda_code_fun = sg_cuda_code_fun



            return spike, v

        @staticmethod
        def backward(ctx, grad_spike, grad_v):
            device = grad_spike.get_device()
            h, spike = ctx.saved_tensors
            grad_x = torch.zeros_like(grad_spike)
            grad_v_last = torch.zeros_like(grad_spike)

            if ctx.cp_v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            if grad_spike.dtype == torch.float32:
                dtype = 'float'
            elif grad_spike.dtype == torch.float16:
                dtype = 'half'
            else:
                raise NotImplementedError

            kernel = create_kernel_IFNode_backward(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

            with cupy.cuda.Device(device):
                cu_kernel_opt.check_contiguous(grad_spike, grad_v, h, spike, grad_x, grad_v_last)

                if hard_reset:
                    cu_kernel_opt.check_device(device, grad_spike, grad_v, h, spike, grad_x, grad_v_last, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.numel)
                    kernel(
                        (ctx.blocks,), (ctx.threads,),
                        (grad_spike.data_ptr(), grad_v.data_ptr(), h.data_ptr(), spike.data_ptr(),
                         grad_x.data_ptr(), grad_v_last.data_ptr(),
                         ctx.cp_v_threshold, ctx.cp_v_reset,
                         ctx.numel)
                    )
                else:
                    cu_kernel_opt.check_device(device, grad_spike, grad_v, h, spike, grad_x, grad_v_last,
                                               ctx.cp_v_threshold, ctx.numel)
                    kernel(
                        (ctx.blocks,), (ctx.threads,),
                        (grad_spike.data_ptr(), grad_v.data_ptr(), h.data_ptr(), spike.data_ptr(),
                         grad_x.data_ptr(), grad_v_last.data_ptr(),
                         ctx.cp_v_threshold,
                         ctx.numel)
                    )
            return grad_x, grad_v_last, None, None, None, None














        


except ImportError:
    pass


