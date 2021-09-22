// This file is created by spikingjelly.clock_driven.neuron_kernel.save_cuda_codes.

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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __hadd2(__halves2half2(v_v_seq[ta], v_v_seq[tb]), __halves2half2(x_seq[ta], x_seq[tb]));
                        h_seq[ta] = __low2half(h_seq_t);
                        h_seq[tb] = __high2half(h_seq_t);
                        
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                    v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                    v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);

                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(__halves2half2(grad_v_seq[ta], grad_v_seq[tb]), grad_h), grad_v_to_h, __hmul2(__halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]), grad_s_to_h));
                        grad_x_seq[ta] = __low2half(grad_h);
                        grad_x_seq[tb] = __high2half(grad_h);
                        }
                grad_v_last[index] = grad_x_seq[index];
                grad_v_last[index + stride] = grad_x_seq[index + stride];
                }
                }
                
// MultiStepLIFNodePTT

// MultiStepLIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hadd2(__hsub2(x_seq_t, v_v_seq_t), v_reset_half2), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hsub2(x_seq_t, v_v_seq_t), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hadd2(__hsub2(x_seq_t, v_v_seq_t), v_reset_half2), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hsub2(x_seq_t, v_v_seq_t), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hadd2(__hsub2(x_seq_t, v_v_seq_t), v_reset_half2), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq_t);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq_t),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq_t));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 v_v_seq_t = __halves2half2(v_v_seq[ta], v_v_seq[tb]);
                        const half2 x_seq_t = __halves2half2(x_seq[ta], x_seq[tb]);
                
                        const half2 h_seq_t = __hfma2(__hsub2(x_seq_t, v_v_seq_t), reciprocal_tau_half2, v_v_seq_t);
                        const half2 spike_seq_t = __hgeu2(h_seq_t, v_threshold_half2);
                        const half2 v_v_seq_t_next = __hadd2(__hmul2(spike_seq_t, __hsub2(h_seq_t, v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq_t), h_seq_t));
                    
                        spike_seq[ta] = __low2half(spike_seq_t);
                        spike_seq[tb] = __high2half(spike_seq_t); 
                        v_v_seq[ta + neuron_num] = __low2half(v_v_seq_t_next);
                        v_v_seq[tb + neuron_num] = __high2half(v_v_seq_t_next);
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq,
                half* grad_x_seq, half* grad_v_last,
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
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int ta = index + mem_offset;
                        const int tb = ta + stride;
                        const half2 h_seq_t = __halves2half2(h_seq[ta], h_seq[tb]);
                        const half2 spike_seq_t = __halves2half2(spike_seq[ta], spike_seq[tb]);
                        const half2 grad_spike_seq_t = __halves2half2(grad_spike_seq[ta], grad_spike_seq[tb]);
                        const half2 grad_v_seq_t = __halves2half2(grad_v_seq[ta], grad_v_seq[tb]);
                        
                        const half2 over_th = __hsub2(h_seq_t, v_threshold_half2);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq_t), grad_v_to_h, __hmul2(grad_spike_seq_t, grad_s_to_h));
                        const half2 grad_x_seq_t = __hmul2(grad_h, reciprocal_tau_half2);
                        grad_x_seq[ta] = __low2half(grad_x_seq_t);
                        grad_x_seq[tb] = __high2half(grad_x_seq_t);
                    }
                const int index_b = index + stride;
                const half2 grad_v_last_ab = __hmul2(__halves2half2(grad_x_seq[index], grad_x_seq[index_b]), one_sub_reciprocal_tau_half2);
                grad_v_last[index] = __low2half(grad_v_last_ab);
                grad_v_last[index_b] = __high2half(grad_v_last_ab);
                }
                }
                
// MultiStepParametricLIFNodePTT

// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t] + v_reset, v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hadd(__hsub(x_seq[t], v_v_seq[t]), v_reset), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                                                
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
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t], v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
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
                
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            const float grad_s_to_h = 2.0f / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hsub(x_seq[t], v_v_seq[t]), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const float grad_v_to_h = __float2half(1.0f);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha =  __float2half2_rn(2.0f);
            const half2 M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), alpha), over_th);
            const half2 grad_s_to_h = __h2div(__h2div(alpha, __float2half2_rn(2.0f)), __hfma2(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                                                
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
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t] + v_reset, v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hadd(__hsub(x_seq[t], v_v_seq[t]), v_reset), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                                                
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
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t], v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
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
                
            const float sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            const float grad_s_to_h = (1.0f - sigmoid_ax) * sigmoid_ax * 1.0f;
            
                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hsub(x_seq[t], v_v_seq[t]), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const float grad_v_to_h = __float2half(1.0f);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                
            const half2 alpha = __float2half2_rn(1.0f);
            const half2 sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(alpha, over_th))), __float2half2_rn(1.0f)));
            const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            
                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                                                
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
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t] + v_reset, v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hadd(__hsub(x_seq[t], v_v_seq[t]), v_reset), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const float grad_v_to_h = __hsub(__float2half(1.0f), spike_seq[t]);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const float grad_v_to_h = __hfma(__hsub(v_reset, h_seq[t]),  grad_s_to_h, __hsub(__float2half(1.0f), spike_seq[t]));
                                                
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
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
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
                
                        //h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        h_seq[t] = fmaf(reciprocal_tau, x_seq[t] - v_v_seq[t], v_v_seq[t]);
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        const float grad_v_to_h = 1.0f;
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
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
                const float x_abs = fabsf(over_th);
float grad_s_to_h;
if (x_abs > 1.0f)
{
grad_s_to_h = 0.01f;
}
else
{
grad_s_to_h = 1.0f;
}

                        //const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    //grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
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
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half* x_seq, half* v_v_seq, half* h_seq, half* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma(__hsub(x_seq[t], v_v_seq[t]), reciprocal_tau, v_v_seq[t]); 
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
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const float grad_v_to_h = __float2half(1.0f);
                                                
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
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half* grad_spike_seq, const half* grad_v_seq, const half* h_seq, const half* spike_seq, const half* v_v_seq,
                half* grad_x_seq, half* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                
                __shared__ half sdata[1024];
                if (index < neuron_num)
                {   
                    half grad_h = __float2half(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half(0.0f);
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const half over_th = __hsub(h_seq[t], v_threshold);
                const half2 x_abs = __habs2(over_th);
const half2 x_abs_ge_w = __hge2(x_abs, __float2half2_rn(1.0f));
half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), x_abs_ge_w), __float2half2_rn(1.0f)));

                        const float grad_v_to_h = __hsub(__float2half(1.0f), __hmul(v_threshold, grad_s_to_h));
                                                
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
                