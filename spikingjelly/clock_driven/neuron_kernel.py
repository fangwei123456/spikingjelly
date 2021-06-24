try:
    import cupy
    import torch
    from . import cu_kernel_opt
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
        def forward(ctx, x: torch.Tensor, v_last: torch.Tensor, v_threshold: float, v_reset:float, detach_reset: bool):
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

                cu_kernel_opt.check_contiguous(x, v_last, h, spike, v)
                cu_kernel_opt.check_device(device, x, v_last, h, spike, v)
                if v_reset is None:
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
                ctx.v_threshold = v_threshold
                ctx.v_reset = v_reset
                ctx.detach_reset = detach_reset


            return h, spike, v





        


except ImportError:
    pass


