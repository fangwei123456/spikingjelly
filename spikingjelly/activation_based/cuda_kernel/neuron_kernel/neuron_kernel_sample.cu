// This file is created by spikingjelly.activation_based.neuron_kernel.save_cuda_codes.
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.ATan.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 4.0f;
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(4.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.activation_based.surrogate.Sigmoid.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
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
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
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
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
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
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
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
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.activation_based.surrogate.PiecewiseLeakyReLU.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt S2NN, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const float sg_S2NN_sigmoid_ax = 1.0f / (1.0f + expf(- 4.0f * over_th));
            				const float sg_S2NN_mask_l = (float)(over_th < 0.0f);
            				const float grad_s_to_h = (1.0f - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * 4.0f * sg_S2NN_mask_l + 1.0f / (over_th + 1.0f) * (1.0f - sg_S2NN_mask_l);
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt S2NN, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
            				const half2 sg_S2NN_alpha = __float2half2_rn(4.0f);
            				const half2 sg_S2NN_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_S2NN_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 sg_S2NN_mask_l = __hlt2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_S2NN_sigmoid_ax), sg_S2NN_sigmoid_ax), sg_S2NN_alpha), sg_S2NN_mask_l), __hmul2(__h2div(__float2half2_rn(1.0f), __hadd2(over_th, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), sg_S2NN_mask_l)));
            
            				// end: spikingjelly.activation_based.surrogate.S2NN.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
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
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const float sg_QPseudoSpike_base = 1.0f + 2.0f / (2.0f - 1.0f) * fabsf(over_th);
            				const float grad_s_to_h = powf(sg_QPseudoSpike_base, -2.0f);
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
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
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt QPseudoSpike, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
            				const half2 sg_QPseudoSpike_alpha = __float2half2_rn(2.0f);
            				const half2 sg_QPseudoSpike_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2(over_th)), __hsub2(sg_QPseudoSpike_alpha, __float2half2_rn(1.0f))));
            				const half2 grad_s_to_h = h2exp2(__hmul2(h2log2(sg_QPseudoSpike_base), __hneg2(sg_QPseudoSpike_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            
            				// end: spikingjelly.activation_based.surrogate.QPseudoSpike.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt LeakyKReLU, hard_reset=True, dtype=fp32

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
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=True, dtype=fp32, detach_reset=True

            extern "C" __global__
            void IFNode_bptt_hardReset_detachReset_fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const float sg_LeakyKReLU_mask1 = (float) (over_th >= 0.0f);
            				const float grad_s_to_h = 0.0f * (1.0f - sg_LeakyKReLU_mask1) + 1.0f * sg_LeakyKReLU_mask1;
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const float sg_LeakyKReLU_mask1 = (float) (over_th >= 0.0f);
            				const float grad_s_to_h = 0.0f * (1.0f - sg_LeakyKReLU_mask1) + 1.0f * sg_LeakyKReLU_mask1;
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt LeakyKReLU, hard_reset=True, dtype=fp16

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
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=True, dtype=fp16, detach_reset=True

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset_detachReset_fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const half2 sg_LeakyKReLU_mask1 = __hgeu2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hfma2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1, __hmul2(__float2half2_rn(0.0f), __hsub2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1)));
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const half2 sg_LeakyKReLU_mask1 = __hgeu2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hfma2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1, __hmul2(__float2half2_rn(0.0f), __hsub2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1)));
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt LeakyKReLU, hard_reset=False, dtype=fp32

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
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=False, dtype=fp32, detach_reset=True

            extern "C" __global__
            void IFNode_bptt_softReset_detachReset_fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const float sg_LeakyKReLU_mask1 = (float) (over_th >= 0.0f);
            				const float grad_s_to_h = 0.0f * (1.0f - sg_LeakyKReLU_mask1) + 1.0f * sg_LeakyKReLU_mask1;
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const float sg_LeakyKReLU_mask1 = (float) (over_th >= 0.0f);
            				const float grad_s_to_h = 0.0f * (1.0f - sg_LeakyKReLU_mask1) + 1.0f * sg_LeakyKReLU_mask1;
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt LeakyKReLU, hard_reset=False, dtype=fp16

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
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=False, dtype=fp16, detach_reset=True

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset_detachReset_fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const half2 sg_LeakyKReLU_mask1 = __hgeu2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hfma2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1, __hmul2(__float2half2_rn(0.0f), __hsub2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1)));
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt LeakyKReLU, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
            				const half2 sg_LeakyKReLU_mask1 = __hgeu2(over_th, __float2half2_rn(0.0f));
            				const half2 grad_s_to_h = __hfma2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1, __hmul2(__float2half2_rn(0.0f), __hsub2(__float2half2_rn(1.0f), sg_LeakyKReLU_mask1)));
            
            				// end: spikingjelly.activation_based.surrogate.LeakyKReLU.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt FakeNumericalGradient, hard_reset=True, dtype=fp32

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
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=True, dtype=fp32, detach_reset=True

            extern "C" __global__
            void IFNode_bptt_hardReset_detachReset_fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const float sg_FakeNumericalGradient_sign = (float) (over_th >= 0.0f) * 2.0f - 1.0f;
            				const float grad_s_to_h = min(sg_FakeNumericalGradient_sign / over_th, 0.3f);
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=True, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_hardReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const float sg_FakeNumericalGradient_sign = (float) (over_th >= 0.0f) * 2.0f - 1.0f;
            				const float grad_s_to_h = min(sg_FakeNumericalGradient_sign / over_th, 0.3f);
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt FakeNumericalGradient, hard_reset=True, dtype=fp16

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
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=True, dtype=fp16, detach_reset=True

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset_detachReset_fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const half2 sg_FakeNumericalGradient_sign = __hfma2(__hgeu2(over_th, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            				const half2 sg_FakeNumericalGradient_grad_x = __h2div(sg_FakeNumericalGradient_sign, over_th);
            				const half2 sg_FakeNumericalGradient_grad_max = __float2half2_rn(0.3f);
            				const half2 grad_s_to_h = make_half2(sg_FakeNumericalGradient_grad_x.x <= sg_FakeNumericalGradient_grad_max.x ? sg_FakeNumericalGradient_grad_x.x : sg_FakeNumericalGradient_grad_max.x, sg_FakeNumericalGradient_grad_x.y <= sg_FakeNumericalGradient_grad_max.y ? sg_FakeNumericalGradient_grad_x.y : sg_FakeNumericalGradient_grad_max.y);
            #else
            				const half2 grad_s_to_h = __hmin2(__h2div(sg_FakeNumericalGradient_sign, over_th), __float2half2_rn(0.3f));
            #endif
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=True, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_hardReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const half2 sg_FakeNumericalGradient_sign = __hfma2(__hgeu2(over_th, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            				const half2 sg_FakeNumericalGradient_grad_x = __h2div(sg_FakeNumericalGradient_sign, over_th);
            				const half2 sg_FakeNumericalGradient_grad_max = __float2half2_rn(0.3f);
            				const half2 grad_s_to_h = make_half2(sg_FakeNumericalGradient_grad_x.x <= sg_FakeNumericalGradient_grad_max.x ? sg_FakeNumericalGradient_grad_x.x : sg_FakeNumericalGradient_grad_max.x, sg_FakeNumericalGradient_grad_x.y <= sg_FakeNumericalGradient_grad_max.y ? sg_FakeNumericalGradient_grad_x.y : sg_FakeNumericalGradient_grad_max.y);
            #else
            				const half2 grad_s_to_h = __hmin2(__h2div(sg_FakeNumericalGradient_sign, over_th), __float2half2_rn(0.3f));
            #endif
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt FakeNumericalGradient, hard_reset=False, dtype=fp32

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
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=False, dtype=fp32, detach_reset=True

            extern "C" __global__
            void IFNode_bptt_softReset_detachReset_fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const float sg_FakeNumericalGradient_sign = (float) (over_th >= 0.0f) * 2.0f - 1.0f;
            				const float grad_s_to_h = min(sg_FakeNumericalGradient_sign / over_th, 0.3f);
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const float grad_v_to_h = 1.0f;
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=False, dtype=fp32, detach_reset=False

            extern "C" __global__
            void IFNode_bptt_softReset__fp32(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const float sg_FakeNumericalGradient_sign = (float) (over_th >= 0.0f) * 2.0f - 1.0f;
            				const float grad_s_to_h = min(sg_FakeNumericalGradient_sign / over_th, 0.3f);
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT fptt FakeNumericalGradient, hard_reset=False, dtype=fp16

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
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=False, dtype=fp16, detach_reset=True

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset_detachReset_fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const half2 sg_FakeNumericalGradient_sign = __hfma2(__hgeu2(over_th, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            				const half2 sg_FakeNumericalGradient_grad_x = __h2div(sg_FakeNumericalGradient_sign, over_th);
            				const half2 sg_FakeNumericalGradient_grad_max = __float2half2_rn(0.3f);
            				const half2 grad_s_to_h = make_half2(sg_FakeNumericalGradient_grad_x.x <= sg_FakeNumericalGradient_grad_max.x ? sg_FakeNumericalGradient_grad_x.x : sg_FakeNumericalGradient_grad_max.x, sg_FakeNumericalGradient_grad_x.y <= sg_FakeNumericalGradient_grad_max.y ? sg_FakeNumericalGradient_grad_x.y : sg_FakeNumericalGradient_grad_max.y);
            #else
            				const half2 grad_s_to_h = __hmin2(__h2div(sg_FakeNumericalGradient_sign, over_th), __float2half2_rn(0.3f));
            #endif
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            
// MultiStepIFNodePTT bptt FakeNumericalGradient, hard_reset=False, dtype=fp16, detach_reset=False

            #include <cuda_fp16.h>
            extern "C" __global__
            void IFNode_bptt_softReset__fp16(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
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
            
            				// start: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
            				const half2 sg_FakeNumericalGradient_sign = __hfma2(__hgeu2(over_th, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            				const half2 sg_FakeNumericalGradient_grad_x = __h2div(sg_FakeNumericalGradient_sign, over_th);
            				const half2 sg_FakeNumericalGradient_grad_max = __float2half2_rn(0.3f);
            				const half2 grad_s_to_h = make_half2(sg_FakeNumericalGradient_grad_x.x <= sg_FakeNumericalGradient_grad_max.x ? sg_FakeNumericalGradient_grad_x.x : sg_FakeNumericalGradient_grad_max.x, sg_FakeNumericalGradient_grad_x.y <= sg_FakeNumericalGradient_grad_max.y ? sg_FakeNumericalGradient_grad_x.y : sg_FakeNumericalGradient_grad_max.y);
            #else
            				const half2 grad_s_to_h = __hmin2(__h2div(sg_FakeNumericalGradient_sign, over_th), __float2half2_rn(0.3f));
            #endif
            
            				// end: spikingjelly.activation_based.surrogate.FakeNumericalGradient.cuda_code
        
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            