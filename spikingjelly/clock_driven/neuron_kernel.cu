// This file is created by spikingjelly.clock_driven.neuron_kernel.save_cuda_codes.
// Note that codes in this file will not be executed This file is just created for reading.

// MultiStepIFNodePTT

// MultiStepIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

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
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
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
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

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
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
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
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

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
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
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
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt S2NN, hard_reset=True, dtype=fp32

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
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt S2NN, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt S2NN, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
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
                    }
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt S2NN, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt QPseudoSpike, hard_reset=True, dtype=fp32

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
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt QPseudoSpike, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt QPseudoSpike, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
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
                    }
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT fptt QPseudoSpike, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_h;
                }
                }
                
// MultiStepLIFNodePTT

// MultiStepLIFNodePTT fptt ATan, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt S2NN, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                 
                            grad_x_seq[t] = grad_h;
                    
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepParametricLIFNodePTT

// MultiStepParametricLIFNodePTT fptt ATan, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt S2NN, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt S2NN, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        
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
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
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
                
                            h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        
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
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                            // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                            h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        
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

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                
                            grad_x_seq[t] = grad_h;
                            sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                        
                    }
                grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
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
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputTrue_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=True, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputTrue_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                            grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                            half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                      
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT fptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_decayInputFalse_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                            // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                            h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                        
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepParametricLIFNodePTT bptt QPseudoSpike, decay_input=False, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_decayInputFalse_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  float* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)\
                // note that grad_reciprocal_tau is float to avoid overflow
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ float sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                  
                                grad_x_seq[t] = grad_h;
                                half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                                sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                          
                    }
                grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int i = threadx >> 1; i > 0; i = i >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < i)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + i];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {                
                /*
                The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

                The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
                
                The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
                
                The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
                
                The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
                */
                
                atomicAdd(grad_reciprocal_tau, sdata[0]);
                    
                }
                }
                
// MultiStepEIFNodePTT

// MultiStepEIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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

                    }
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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

                    }
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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

                    }
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt S2NN, hard_reset=True, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt S2NN, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt S2NN, hard_reset=False, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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

                    }
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt S2NN, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt S2NN, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.clock_driven.surrogate.S2NN.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt QPseudoSpike, hard_reset=True, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt QPseudoSpike, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT fptt QPseudoSpike, hard_reset=False, dtype=fp32

                extern "C" __global__
                void EIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & delta_T,
                const float & theta_rh,
                const float & v_threshold,
                const float & v_rest, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
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

                    }
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void EIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & theta_rh, const float & reciprocal_delta_T,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
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
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
                }
                }
                
// MultiStepEIFNodePTT fptt QPseudoSpike, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & delta_T,
                const half & theta_rh,
                const half & v_threshold,
                const half & v_rest, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 delta_T_half2 = __half2half2(delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                    const half2 v_rest_half2 = __half2half2(v_rest);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                
// MultiStepEIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void EIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & theta_rh, const half & reciprocal_delta_T,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                    const half2 theta_rh_half2 = __half2half2(theta_rh);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.clock_driven.surrogate.QPseudoSpike.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
                }
                }
                