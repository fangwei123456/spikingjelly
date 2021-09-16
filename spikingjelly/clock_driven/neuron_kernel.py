try:
    import cupy
    import torch
    from . import cu_kernel_opt, surrogate

    class MultiStepIFNodePTT(torch.autograd.Function):
        @staticmethod
        def create_fptt_kernel(hard_reset: bool, dtype: str):
            kernel_name = f'IFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

            if dtype == 'fp32':
                code = rf'''
                extern "C" __global__
                void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel) 
                '''

                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                '''

                if hard_reset:
                    code += r'''
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                '''

            elif dtype == 'fp16':
                code = rf'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel) 
                '''

                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd(v_v_seq[t], x_seq[t]);
                        if (__hgeu(h_seq[t], v_threshold))
                '''

                if hard_reset:
                    code += r'''
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = __hsub(h_seq[t], v_threshold);
                        }

                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = __float2half(0.0f);
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                '''
            else:
                raise TypeError

            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

            kernel_name = f'IFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

            code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

            if dtype == 'fp32':
                code = fr'''
                extern "C" __global__
                void {kernel_name}(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''

                code += r'''
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
                code += code_grad_s_to_h
                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f;
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
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

            elif dtype == 'fp16':
                code = fr'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                '''
                code += code_grad_s_to_h

                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __float2half(1.0f);
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                        '''

                code += code_grad_v_to_h
                code += r'''
                        grad_h = __hfma(__hadd(grad_v_seq[t], grad_h), grad_v_to_h, __hmul(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                '''

            else:
                raise TypeError
            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_last: torch.Tensor, v_threshold: float, v_reset: float, detach_reset: bool, sg_cuda_code_fun):
            requires_grad = x_seq.requires_grad or v_last.requires_grad
            device = x_seq.get_device()
            if x_seq.dtype == torch.float32:
                dtype = 'fp32'
                cp_dtype = cupy.float32
            elif x_seq.dtype == torch.float16:
                dtype = 'fp16'
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
                    hard_reset = False
                    x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel]
                else:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                    hard_reset = True
                    x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel]


                kernel = MultiStepIFNodePTT.create_fptt_kernel(hard_reset, dtype)


                kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            if requires_grad:
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
                dtype = 'fp32'
            elif grad_spike_seq.dtype == torch.float16:
                dtype = 'fp16'
            else:
                raise NotImplementedError

            kernel = MultiStepIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

            with cupy.cuda.Device(device):

                if hard_reset:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
                else:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

                kernel(
                    (ctx.blocks,), (ctx.threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            return grad_x_seq, grad_v_last, None, None, None, None


    class MultiStepLIFNodePTT(torch.autograd.Function):
        @staticmethod
        def create_fptt_kernel(hard_reset: bool, dtype: str):
            kernel_name = f'LIFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

            if dtype == 'fp32':
                code = rf'''
                extern "C" __global__
                void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                '''

                if hard_reset:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                '''

            elif dtype == 'fp16':
                code = rf'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel) 
                '''

                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                '''

                if hard_reset:
                    code += r'''
                        h_seq[t] = __hfma(__hadd(__hsub(x_seq[t], v_v_seq[t]), v_reset), reciprocal_tau, v_v_seq[t]); 
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        h_seq[t] = __hfma(__hsub(x_seq[t], v_v_seq[t]), reciprocal_tau, v_v_seq[t]); 
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = __hsub(h_seq[t], v_threshold);
                        }

                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = __float2half(0.0f);
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                '''
            else:
                raise TypeError

            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

            kernel_name = f'LIFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

            code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

            if dtype == 'fp32':
                code = fr'''
                extern "C" __global__
                void {kernel_name}(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''

                code += r'''
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
                code += code_grad_s_to_h
                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f;
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        '''

                code += code_grad_v_to_h
                code += r'''
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                '''

            elif dtype == 'fp16':
                code = fr'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                '''
                code += code_grad_s_to_h

                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __float2half(1.0f);
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                        '''

                code += code_grad_v_to_h
                code += r'''                        
                        grad_h = __hfma(__hfma(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]), grad_v_to_h, __hmul(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul(grad_h, reciprocal_tau);
                    }
                grad_v_last[index] = __hmul(grad_x_seq[index], one_sub_reciprocal_tau);
                }
                }
                '''

            else:
                raise TypeError
            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_last: torch.Tensor, tau: float, v_threshold: float, v_reset: float, detach_reset: bool, sg_cuda_code_fun):
            requires_grad = x_seq.requires_grad or v_last.requires_grad
            device = x_seq.get_device()
            if x_seq.dtype == torch.float32:
                dtype = 'fp32'
                cp_dtype = cupy.float32
            elif x_seq.dtype == torch.float16:
                dtype = 'fp16'
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
                cp_reciprocal_tau = cupy.asarray(1./tau, dtype=cp_dtype)
                cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1./tau, dtype=cp_dtype)

                if v_reset is None:
                    cp_v_reset = None
                    hard_reset = False
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel]
                else:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                    hard_reset = True
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel]

                kernel = MultiStepLIFNodePTT.create_fptt_kernel(hard_reset, dtype)

                kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            if requires_grad:
                ctx.save_for_backward(h_seq, spike_seq)
                ctx.blocks = blocks
                ctx.threads = threads
                ctx.cp_numel = cp_numel
                ctx.cp_neuron_num = cp_neuron_num
                ctx.cp_reciprocal_tau = cp_reciprocal_tau
                ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
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
                dtype = 'fp32'
            elif grad_spike_seq.dtype == torch.float16:
                dtype = 'fp16'
            else:
                raise NotImplementedError

            kernel = MultiStepLIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

            with cupy.cuda.Device(device):

                if hard_reset:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
                else:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]


                kernel(
                    (ctx.blocks,), (ctx.threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            return grad_x_seq, grad_v_last, None, None, None, None, None

    class MultiStepParametricLIFNodePTT(torch.autograd.Function):
        @staticmethod
        def create_fptt_kernel(hard_reset: bool, dtype: str):
            kernel_name = f'ParametricLIFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

            if dtype == 'fp32':
                code = rf'''
                extern "C" __global__
                void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                '''

                if hard_reset:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                '''

            elif dtype == 'fp16':
                code = rf'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel) 
                '''

                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                '''

                if hard_reset:
                    code += r'''
                        h_seq[t] = __hfma(__hadd(__hsub(x_seq[t], v_v_seq[t]), v_reset), reciprocal_tau, v_v_seq[t]); 
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = v_reset;
                        }
                    '''
                else:
                    code += r'''
                        h_seq[t] = __hfma(__hsub(x_seq[t], v_v_seq[t]), reciprocal_tau, v_v_seq[t]); 
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = __hsub(h_seq[t], v_threshold);
                        }

                    '''

                code += r'''
                        else
                        {
                            spike_seq[t] = __float2half(0.0f);
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                '''
            else:
                raise TypeError

            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

            kernel_name = f'ParametricLIFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

            code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

            if dtype == 'fp32':
                code = fr'''
                extern "C" __global__
                void {kernel_name}(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                '''
                code += f'__shared__ float sdata[{cu_kernel_opt.threads}];'
                code += r'''
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                '''
                code += code_grad_s_to_h
                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f;
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        '''

                code += code_grad_v_to_h
                code += r'''
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                '''

            elif dtype == 'fp16':
                code = fr'''
                #include <cuda_fp16.h>
                extern "C" __global__
                void {kernel_name}(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
                const int & neuron_num, const int & numel)
                '''
                code += r'''
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                '''
                code += f'__shared__ half sdata[{cu_kernel_opt.threads}];'
                code += r'''
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                '''
                code += code_grad_s_to_h

                if detach_reset:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __float2half(1.0f);
                        '''
                else:
                    if hard_reset:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                        '''
                    else:
                        code_grad_v_to_h = r'''
                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                        '''

                code += code_grad_v_to_h
                code += r'''                        
                        grad_h = __hfma(__hfma(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]), grad_v_to_h, __hmul(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul(grad_h, reciprocal_tau);
                        sdata[threadIdx.x] = __hadd(__hdiv(__hmul(grad_h, __hsub(h_seq[t], v_v_seq[t])), reciprocal_tau), sdata[threadIdx.x]);
                    }
                grad_v_last[index] = __hmul(grad_x_seq[index], one_sub_reciprocal_tau);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                '''

            else:
                raise TypeError

            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_last: torch.Tensor, reciprocal_tau: torch.Tensor, v_threshold: float, v_reset: float, detach_reset: bool, sg_cuda_code_fun):
            requires_grad = x_seq.requires_grad or v_last.requires_grad
            device = x_seq.get_device()
            if x_seq.dtype == torch.float32:
                dtype = 'fp32'
                cp_dtype = cupy.float32
            elif x_seq.dtype == torch.float16:
                dtype = 'fp16'
                cp_dtype = cupy.float16
            else:
                raise NotImplementedError

            v_seq = torch.zeros_like(x_seq.data)
            h_seq = torch.zeros_like(x_seq.data)
            spike_seq = torch.zeros_like(x_seq.data)

            v_v_seq = torch.cat((v_last.unsqueeze(0), v_seq))
            tau = 1. / reciprocal_tau.item()

            with cupy.cuda.Device(device):
                numel = x_seq.numel()
                neuron_num = numel // x_seq.shape[0]
                threads = cu_kernel_opt.threads
                blocks = cu_kernel_opt.cal_blocks(neuron_num)
                cp_numel = cupy.asarray(numel)
                cp_neuron_num = cupy.asarray(neuron_num)
                cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
                cp_reciprocal_tau = cupy.asarray(1./tau, dtype=cp_dtype)
                cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1./tau, dtype=cp_dtype)

                if v_reset is None:
                    cp_v_reset = None
                    hard_reset = False
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel]
                else:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                    hard_reset = True
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel)
                    kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel]

                kernel = MultiStepParametricLIFNodePTT.create_fptt_kernel(hard_reset, dtype)

                kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            if requires_grad:
                ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
                ctx.blocks = blocks
                ctx.threads = threads
                ctx.cp_numel = cp_numel
                ctx.cp_neuron_num = cp_neuron_num
                ctx.cp_reciprocal_tau = cp_reciprocal_tau
                ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
                ctx.cp_v_threshold = cp_v_threshold
                ctx.cp_v_reset = cp_v_reset
                ctx.detach_reset = detach_reset
                ctx.sg_cuda_code_fun = sg_cuda_code_fun

            return spike_seq, v_v_seq[1:, ]

        @staticmethod
        def backward(ctx, grad_spike_seq, grad_v_seq):
            device = grad_spike_seq.get_device()
            h_seq, spike_seq, v_v_seq = ctx.saved_tensors
            grad_x_seq = torch.zeros_like(grad_spike_seq)
            grad_v_last = torch.zeros_like(grad_spike_seq[0])
            grad_reciprocal_tau = torch.as_tensor(0.).to(grad_spike_seq)

            if ctx.cp_v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            if grad_spike_seq.dtype == torch.float32:
                dtype = 'fp32'
            elif grad_spike_seq.dtype == torch.float16:
                dtype = 'fp16'
            else:
                raise NotImplementedError

            kernel = MultiStepParametricLIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

            with cupy.cuda.Device(device):

                if hard_reset:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
                else:
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                    kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_last, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

                kernel(
                    (ctx.blocks,), (ctx.threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device,
                        *kernel_args
                    )
                )

            return grad_x_seq, grad_v_last, grad_reciprocal_tau, None, None, None, None


    def check_multi_step_neuron_output_and_grad(device, multi_step_neuron, *neu_args, **neu_kwargs):
        @torch.no_grad()
        def max_error(x, y):
            return (x - y).abs().max().item()

        def fbptt(m, x: torch.Tensor):
            x = x.detach()
            x.requires_grad_(True)
            m(x)
            (m.spike_seq * m.v_seq**2).sum().backward()
            ret = [m.spike_seq.detach().clone(), m.v_seq.detach().clone(), x.grad.clone()]
            for param in m.parameters():
                ret.append(param.grad.detach().clone())
                param.grad.zero_()
            x.grad.zero_()
            m.reset()
            return ret

        shape = [64, 16, 5]
        for hard_reset in [True, False]:
            for detach_reset in [False, True]:
                for dtype in ['fp32', 'fp16']:
                    if dtype == 'fp32':
                        x = torch.rand(shape, device=device)
                    elif dtype == 'fp16':
                        x = torch.rand(shape, device=device).half()
                    print(f'hard_reset={hard_reset}, detach_reset={detach_reset}, dtype={dtype}')
                    model = multi_step_neuron(v_reset=0. if hard_reset else None, detach_reset=detach_reset, *neu_args, **neu_kwargs)
                    print(model)
                    model.to(device)
                    model.backend = 'torch'
                    y_torch = fbptt(model, x)

                    model.backend = 'cupy'
                    y_cupy = fbptt(model, x)

                    for i in range(y_torch.__len__()):
                        print(max_error(y_torch[i], y_cupy[i]))





except ImportError:
    pass



