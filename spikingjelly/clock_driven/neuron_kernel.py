try:
    import cupy
    import torch
    from . import cu_kernel_opt
    from . import surrogate



    class MultiStepIFNodePTT(torch.autograd.Function):
        @staticmethod
        def create_kernel_IFNode_fptt(hard_reset: bool, dtype: str):

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
                '''

                if hard_reset:
                    code += r'''
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
                    '''
                else:
                    code += r'''
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    '''

                code += r'''
                    }
                }
                }
                '''

            elif dtype == 'fp16':
                code = rf'''
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
                '''
                if hard_reset:
                    code += r'''
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = v_reset;
                        }
                        else
                        {
                            spike_seq[t] = __float2half(0.0f);
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    '''
                else:
                    code += r'''
                        if (__hgeu(h_seq[t], v_threshold))
                        {
                            spike_seq[t] = __float2half(1.0f);
                            v_v_seq[t + dt] = __hsub(h_seq[t], v_threshold);
                        }
                        else
                        {
                            spike_seq[t] = __float2half(0.0f);
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    '''

                code += r'''
                    }
                }
                }
                '''
            else:
                raise TypeError

            return cupy.RawKernel(code, kernel_name, options=cu_kernel_opt.nvcc_options)

        @staticmethod
        def create_kernel_IFNode_bptt(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

            kernel_name = f'IFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

            code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

            if dtype == 'fp32':
                code = fr'''
                extern "C" __global__
                void {kernel_name}(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
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
                extern "C" __global__
                void {kernel_name}(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                '''
                code += r'''
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
                else:
                    cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                    hard_reset = True

                kernel = MultiStepIFNodePTT.create_kernel_IFNode_fptt(hard_reset, dtype)


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
                dtype = 'fp32'
            elif grad_spike_seq.dtype == torch.float16:
                dtype = 'fp16'
            else:
                raise NotImplementedError

            kernel = MultiStepIFNodePTT.create_kernel_IFNode_bptt(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

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


