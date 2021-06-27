try:
    import cupy
    import torch
    from . import cu_kernel_opt
    from . import surrogate

    # IFNode----------------------------------------------------------------------------------------------------------
    IFNode_fptt_hardReset_fp32 = cupy.RawKernel(r'''
    extern "C" __global__
    void IFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
    const float & v_threshold, const float & v_reset,
    const int & neuron_num, const int & numel) 
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < neuron_num)
    {
        const int dt = neuron_num;
        for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
        {
            const int t = index + mem_offset;
            h_seq[t] = v_v_seq[t] + x_seq[t];
            // todo: use mul to replace if and check speed
            // spike_seq[t] = float (h_seq[t] >= v_threshold);
            // v_v_seq[t + dt] = v_reset * spike_seq[t] + (1.0f - spike_seq[t]) * h_seq[t];
            if (h_seq[t] >= v_threshold)
            {
                spike_seq[t] = 1.0f;
                v_v_seq[t + dt] = v_reset;
            }
            else
            {
                spike_seq[t] = 0.0f;
                v_v_seq[t + dt] = h_seq[t];
            }
        }
    }
    }
    ''', 'IFNode_fptt_hardReset_fp32', options=cu_kernel_opt.nvcc_options)


    def create_kernel_IFNode_bptt(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):
        if hard_reset:
            if dtype == 'float':
                code = r'''
                extern "C" __global__
                void IFNode_bptt_hardReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        
                        const float over_th = h_seq[t] - v_threshold;
                '''

                code += sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

                if detach_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (-h_seq[t] + v_reset) * grad_s_to_h;
                    '''
                code += code_grad_v_to_h
                code += r'''
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                '''
                print(code)
                return cupy.RawKernel(code, 'IFNode_bptt_hardReset_fp32', options=cu_kernel_opt.nvcc_options)
            elif dtype == 'half':
                pass
        else:
            pass

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


    class MultiStepIFNodePTT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_last: torch.Tensor, v_threshold: float, v_reset: float, detach_reset: bool, sg_cuda_code_fun):
            device = x_seq.get_device()
            if x_seq.dtype == torch.float32:
                cp_dtype = cupy.float32
            elif x_seq.dtype == torch.float16:
                cp_dtype = cupy.float16
            else:
                raise NotImplementedError

            v_seq = torch.zeros_like(x_seq.data)
            h_seq = torch.zeros_like(x_seq.data)
            spike_seq = torch.zeros_like(x_seq.data)

            v_v_seq = torch.cat((v_last.unsqueeze(0), v_seq))

            with cupy.cuda.Device(device):
                numel = x_seq.numel()
                neuron_num = numel // x_seq.shape[0]

                threads = cu_kernel_opt.threads
                blocks = cu_kernel_opt.cal_blocks(neuron_num)
                cp_numel = cupy.asarray(numel)
                cp_neuron_num = cupy.asarray(neuron_num)
                cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
                if v_reset is None:
                    cp_v_reset = None
                else:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)

                if v_reset is None:
                    # soft reset
                    if x_seq.dtype == torch.float32:
                        pass
                    elif x_seq.dtype == torch.float16:
                        pass
                    else:
                        raise NotImplementedError
                    pass


                else:
                    # hard reset

                    if x_seq.dtype == torch.float32:
                        kernel = IFNode_fptt_hardReset_fp32

                    elif x_seq.dtype == torch.float16:
                        pass
                    else:
                        raise NotImplementedError


                    kernel(
                        (blocks,), (threads,),
                        cu_kernel_opt.wrap_args_to_raw_kernel(
                            device,
                            [x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel]
                        )
                    )

            if x_seq.requires_grad or v_last.requires_grad:
                ctx.save_for_backward(h_seq, spike_seq)
                ctx.blocks = blocks
                ctx.threads = threads
                ctx.cp_numel = cp_numel
                ctx.cp_neuron_num = cp_neuron_num
                ctx.cp_v_threshold = cp_v_threshold
                ctx.cp_v_reset = cp_v_reset
                ctx.detach_reset = detach_reset
                ctx.sg_cuda_code_fun = sg_cuda_code_fun

            return spike_seq, v_v_seq[1:, ]



        @staticmethod
        def backward(ctx, grad_spike_seq, grad_v_seq):
            device = grad_spike_seq.get_device()
            h_seq, spike_seq = ctx.saved_tensors
            grad_x_seq = torch.zeros_like(grad_spike_seq)
            grad_v_last = torch.zeros_like(grad_spike_seq[0])

            if ctx.cp_v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            if grad_spike_seq.dtype == torch.float32:
                dtype = 'float'
            elif grad_spike_seq.dtype == torch.float16:
                dtype = 'half'
            else:
                raise NotImplementedError

            kernel = create_kernel_IFNode_bptt(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

            with cupy.cuda.Device(device):

                if hard_reset:
                    kernel(
                        (ctx.blocks,), (ctx.threads,),
                        cu_kernel_opt.wrap_args_to_raw_kernel(
                            device,
                            [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
                        )
                    )
                else:
                    pass

            return grad_x_seq, grad_v_last, None, None, None, None














        


except ImportError:
    pass


