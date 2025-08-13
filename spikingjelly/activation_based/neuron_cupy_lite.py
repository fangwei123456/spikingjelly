# -*- coding: utf-8 -*-
# @Time    : 2025/5/30 15:09
# @Author  : if
# @File    : neuron_cupy.py
import math
import cupy as cp
import numpy as np
from typing import Callable

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate

__all__ = ["IFNode", "LIFNode", "ParametricLIFNode", "ILIFNode"]


class DeviceEnvironment:
    def __init__(self, device: int):
        self.device = device
        self.previous_device = None

    def __enter__(self):
        current_device = torch.cuda.current_device()
        if current_device != self.device:
            torch.cuda.set_device(self.device)
            self.previous_device = current_device

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_device is not None:
            torch.cuda.set_device(self.previous_device)


class IFNodeCuPy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, forward_kernel, backward_kernel):
        inputs = inputs.contiguous()
        T, B, C, H, W = inputs.shape
        numel = inputs.numel() // T

        spike_seq = torch.empty_like(inputs, dtype=inputs.dtype)
        h_seq = torch.empty_like(inputs, dtype=inputs.dtype)

        threads = 256
        if inputs.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads

        forward_kernel(
            (blocks,), (threads,),
            (inputs.data_ptr(), spike_seq.data_ptr(), h_seq.data_ptr(), numel, T)
        )

        ctx.save_for_backward(h_seq)
        ctx.backward_kernel = backward_kernel

        return spike_seq

    @staticmethod
    def backward(ctx, grad_spike_seq):
        h_seq = ctx.saved_tensors[0]
        grad_spike_seq = grad_spike_seq.contiguous()

        grad_x_seq = torch.empty_like(h_seq, dtype=h_seq.dtype)

        T = grad_spike_seq.shape[0]
        numel = grad_spike_seq.numel() // T
        backward_kernel = ctx.backward_kernel

        threads = 256
        if h_seq.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        backward_kernel(
            (blocks,), (threads,),
            (grad_spike_seq.data_ptr(), h_seq.data_ptr(), grad_x_seq.data_ptr(), numel, T)
        )

        return grad_x_seq, None, None


class IFNode(nn.Module):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_func = surrogate_function
        self.cupy_forward_kernel = None
        self.cupy_backward_kernel = None
        self.hard_reset = True
        if self.v_reset is None:
            self.v_reset = 0.
            self.hard_reset = False

        self.alpha = 0
        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.alpha = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.alpha = surrogate_function.alpha * 2
            self.pai = math.pi * math.pi * surrogate_function.alpha * surrogate_function.alpha
        else:
            raise f"Surrogate: Sigmoid | ATan, alpha: {self.alpha}"
        self.sn_apply = IFNodeCuPy.apply

    def extra_repr(self):
        return f"IFNode CuPy: v_threshold={self.v_th}, v_reset={self.v_reset}, detach_reset={self.detach_reset}"

    def get_cupy_forward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_forward_half()
        else:
            return self.get_cupy_codes_forward_float()

    def get_cupy_codes_forward_float(self):
        kernel_code = r'''
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void IFNodeFPTTFLOATKernel(
    float* __restrict__ inputs, 
    float* __restrict__ spikes_seq, 
    float* __restrict__ h_seq, 
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + v[i];
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }
'''

        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = spikes[i] > 0 ? v_reset : v[i];
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = v[i] - spikes[i] * v_th;
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
            }
        }
    }
}
"""

        return kernel_code, "IFNodeFPTTFLOATKernel"

    def get_cupy_codes_forward_half(self):
        kernel_code = r'''
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void IFNodeFPTTHALFKernel(
    half* __restrict__ inputs, 
    half* __restrict__ spikes_seq, 
    half* __restrict__ h_seq, 
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0); 
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[i] = last_v[i] + v[i];
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }
'''

        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] * (__float2half2_rn(1.0f) - spikes[i]) + spikes[i] * __float2half2_rn(v_reset);
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] - spikes[i] * __float2half2_rn(v_th);
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            auto *ptr = (half *) &spikes[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = ptr[i];
            }
        }
    }
}
"""

        return kernel_code, "IFNodeFPTTHALFKernel"

    def get_cupy_backward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_backward_half()
        else:
            return self.get_cupy_codes_backward_float()

    def get_cupy_codes_backward_float(self):
        kernel_code = r"""
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void IFNodeBPTTFLOATKernel(
    float* __restrict__ grad_spike_seq,  
    float* __restrict__ h_seq, 
    float* __restrict__ grad_x_seq, 
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float load[4];
    float var[4], grad_v_to_h[4], grad_h[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = 0;
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = h_seq[index + i];;
            }
        }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th; // var = over_th | load = h
            grad_v_to_h[i] = var[i] < 0;
        } 
""".replace("v_th", f"{self.v_th}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th; // var = over_th | load = h
            grad_v_to_h[i] = 1.0f;
        }  
""".replace("v_th", f"{self.v_th}f")

        # surrogate_func
        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = 1.0f / (1.0f + expf(-alpha * var[i]));
            var[i] = (1.0f - var[i]) * var[i] * alpha;      // var = grad_s_to_h
        }
""".replace("alpha", f"{self.alpha}f")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = alpha / (4.0f + pai * var[i] * var[i]);  // var = grad_s_to_h
        } 
""".replace("alpha", f"{self.alpha}f").replace("pai", f"{self.pai}f")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
        }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = grad_spike_seq[index + i];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            // grad_h[i] = grad_h[i] * grad_h_next_to_v(1.0f) + grad_v[i](0);
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
            // grad_x_seq = grad_h[i] * grad_h_to_x(1.0f)
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = grad_h[i];
            }
        }
    }
}
"""

        return kernel_code, "IFNodeBPTTFLOATKernel"

    def get_cupy_codes_backward_half(self):
        kernel_code = r"""
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void IFNodeBPTTHALFKernel(
    half* __restrict__ grad_spike_seq, 
    half* __restrict__ h_seq, 
    half* __restrict__ grad_x_seq, 
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = __float2half2_rn(0);
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = h_seq[index + i];
            }
        }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - __float2half2_rn(v_th); // var = over_th
            grad_v_to_h[i] = __hle2(var[i], __float2half2_rn(0));
        } 
""".replace("v_th", f"{self.v_th}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - __float2half2_rn(v_th); // var = over_th
            grad_v_to_h[i] = __float2half2_rn(1.0f);
        }  
""".replace("v_th", f"{self.v_th}f")

        # surrogate_func
        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = __float2half2_rn(1) / (__float2half2_rn(1) + h2exp(-__float2half2_rn(alpha) * var[i]));
            var[i] = (__float2half2_rn(1) - var[i]) * var[i] * __float2half2_rn(alpha);  // grad_s_to_h
        }
""".replace("alpha", f"{self.alpha}f")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = __float2half2_rn(alpha) / (__float2half2_rn(4.0f) + __float2half2_rn(pai) * var[i] * var[i]);  // var = grad_s_to_h
        } 
""".replace("alpha", f"{self.alpha}f").replace("pai", f"{self.pai}f")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
        }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = grad_spike_seq[index + i];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            auto *ptr = (half *) &grad_h[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}
"""
        return kernel_code, "IFNodeBPTTHALFKernel"

    def get_cupy_kernel(self, dtype=torch.float32):
        forward_codes, kernel_name = self.get_cupy_forward_codes(dtype)
        self.cupy_forward_kernel = cp.RawKernel(forward_codes, kernel_name, backend="nvrtc")

        backward_codes, kernel_name = self.get_cupy_backward_codes(dtype)
        self.cupy_backward_kernel = cp.RawKernel(backward_codes, kernel_name, backend="nvrtc")

    def forward(self, x):
        if self.cupy_forward_kernel is None:
            self.get_cupy_kernel(x.dtype)

        with DeviceEnvironment(x.get_device()):
            spike_seq = self.sn_apply(x, self.cupy_forward_kernel, self.cupy_backward_kernel)

        return spike_seq


class LIFNodeCuPy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, forward_kernel, backward_kernel):
        inputs = inputs.contiguous()

        T = inputs.shape[0]
        numel = inputs.numel() // T

        spike_seq = torch.empty_like(inputs, dtype=inputs.dtype)
        h_seq = torch.empty_like(inputs, dtype=inputs.dtype)

        threads = 256
        if inputs.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        forward_kernel(
            (blocks,), (threads,),
            (inputs.data_ptr(), spike_seq.data_ptr(), h_seq.data_ptr(), numel, T)
        )

        ctx.save_for_backward(h_seq)
        ctx.backward_kernel = backward_kernel

        return spike_seq

    @staticmethod
    def backward(ctx, grad_spike_seq):
        h_seq = ctx.saved_tensors[0]
        grad_spike_seq = grad_spike_seq.contiguous()

        grad_x_seq = torch.empty_like(h_seq, dtype=h_seq.dtype)

        T = grad_spike_seq.shape[0]
        numel = grad_spike_seq.numel() // T
        backward_kernel = ctx.backward_kernel

        threads = 256
        if h_seq.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        backward_kernel(
            (blocks,), (threads,),
            (grad_spike_seq.data_ptr(), h_seq.data_ptr(), grad_x_seq.data_ptr(), numel, T)
        )

        return grad_x_seq, None, None


class LIFNode(nn.Module):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        self.decay = 1. / tau
        self.decay_input = decay_input
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_func = surrogate_function
        self.cupy_forward_kernel = None
        self.cupy_backward_kernel = None
        self.hard_reset = True
        if self.v_reset is None:
            self.v_reset = 0.
            self.hard_reset = False

        self.alpha = 0

        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.alpha = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.alpha = surrogate_function.alpha * 2
            self.pai = math.pi * math.pi * surrogate_function.alpha * surrogate_function.alpha
        else:
            raise f"Surrogate: Sigmoid | ATan, alpha: {self.alpha}"
        self.sn_apply = LIFNodeCuPy.apply

    def extra_repr(self):
        return f"LIFNode CuPy: decay={self.decay}, decay_input={self.decay_input}, v_threshold={self.v_th}," \
               f" v_reset={self.v_reset}, detach_reset={self.detach_reset}"

    def get_cupy_forward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_forward_half()
        else:
            return self.get_cupy_codes_forward_float()

    def get_cupy_codes_forward_float(self):
        kernel_code = r'''
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void LIFNodeFPTTFLOATKernel(
    float* __restrict__ inputs, 
    float* __restrict__ spikes_seq, 
    float* __restrict__ h_seq, 
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;

        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }
'''

        if self.decay_input:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + (v[i] - (last_v[i] - v_reset)) * decay;
        }     
""".replace("decay", f"{self.decay}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + v[i] - (last_v[i] - v_reset) * decay;
        }     
""".replace("decay", f"{self.decay}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }
"""

        # reset
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = spikes[i] > 0 ? v_reset : v[i];
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = v[i] - spikes[i] * v_th;
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
            }
        }
    }
}
"""

        return kernel_code, "LIFNodeFPTTFLOATKernel"

    def get_cupy_codes_forward_half(self):
        kernel_code = r'''
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void LIFNodeFPTTHALFKernel(
    half* __restrict__ inputs, 
    half* __restrict__ spikes_seq, 
    half* __restrict__ h_seq,
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0); 
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;

        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }
'''
        # neuron charge
        if self.decay_input:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + (v[i] - (last_v[i] - __float2half2_rn(v_reset))) * __float2half2_rn(decay);
        }     
""".replace("decay", f"{self.decay}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + v[i] - (last_v[i] - __float2half2_rn(v_reset)) * __float2half2_rn(decay);
        }     
""".replace("decay", f"{self.decay}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }
"""
        # reset
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] * (__float2half2_rn(1.0f) - spikes[i]) + spikes[i] * __float2half2_rn(v_reset);  // hard reset
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] - spikes[i] * __float2half2_rn(v_th);  // soft reset
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            auto *spike_ptr = (half *) &spikes[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spike_ptr[i];
            }
        }
    }
}
"""

        return kernel_code, "LIFNodeFPTTHALFKernel"

    def get_cupy_backward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_backward_half()
        else:
            return self.get_cupy_codes_backward_float()

    def get_cupy_codes_backward_float(self):
        kernel_code = r"""
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void LIFNodeBPTTFLOATKernel(
    float* __restrict__ grad_spike_seq,  
    float* __restrict__ h_seq, 
    float* __restrict__ grad_x_seq, 
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;

    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;
    
    float load[4];
    float var[4], grad_v_to_h[4], grad_h[4];
"""

        if self.decay_input:
            kernel_code += r"""
    const float grad_h_to_x = decay;
    const float grad_h_next_to_v = 1.0f - decay;  
""".replace("tau", f"{self.decay}f")
        else:
            kernel_code += r"""
    const float grad_h_to_x = 1.0f;
    const float grad_h_next_to_v = 1.0f - decay;  
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = 0;
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = h_seq[index + i];;
            }
        }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th; // var = over_th | load = h
            grad_v_to_h[i] = var[i] < 0;
        } 
""".replace("v_th", f"{self.v_th}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th; // var = over_th | load = h
            grad_v_to_h[i] = 1.0f;
        }  
""".replace("v_th", f"{self.v_th}f")

        # surrogate_func
        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = 1.0f / (1.0f + expf(-alpha * var[i]));
            var[i] = (1.0f - var[i]) * var[i] * alpha;      // var = grad_s_to_h
        }
""".replace("alpha", f"{self.alpha}f")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = alpha / (4.0f + pai * var[i] * var[i]);  // var = grad_s_to_h
        } 
""".replace("alpha", f"{self.alpha}f").replace("pai", f"{self.pai}f")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
        } 
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = grad_spike_seq[index + i];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_h[i] = grad_h[i] * grad_h_next_to_v;
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

            var[i] = grad_h[i] * grad_h_to_x;
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = var[i];
            }
        }
    }
}
"""

        return kernel_code, "LIFNodeBPTTFLOATKernel"

    def get_cupy_codes_backward_half(self):
        kernel_code = r"""
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void LIFNodeBPTTHALFKernel(
    half* __restrict__ grad_spike_seq, 
    half* __restrict__ h_seq, 
    half* __restrict__ grad_x_seq, 
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;

    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];
"""

        if self.decay_input:
            kernel_code += r"""
    const half2 grad_h_to_x = __float2half2_rn(decay);
    const half2 grad_h_next_to_v = __float2half2_rn(1.0f) - __float2half2_rn(decay);  
""".replace("decay", f"{self.decay}f")
        else:
            kernel_code += r"""
    const half2 grad_h_to_x = __float2half2_rn(1.0f);
    const half2 grad_h_next_to_v = __float2half2_rn(1.0f) - __float2half2_rn(decay);  
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = __float2half2_rn(0);
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = h_seq[index + i];
            }
        }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - __float2half2_rn(v_th); // var = over_th
            grad_v_to_h[i] = __hle2(var[i], __float2half2_rn(0));
        } 
""".replace("v_th", f"{self.v_th}")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - __float2half2_rn(v_th); // var = over_th
            grad_v_to_h[i] = __float2half2_rn(1.0f);
        }  
""".replace("v_th", f"{self.v_th}")

        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = __float2half2_rn(1) / (__float2half2_rn(1) + h2exp(-__float2half2_rn(alpha) * var[i]));
            var[i] = (__float2half2_rn(1) - var[i]) * var[i] * __float2half2_rn(alpha);  // grad_s_to_h
        }
""".replace("alpha", f"{self.alpha}")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = __float2half2_rn(alpha) / (__float2half2_rn(4.0f) + __float2half2_rn(pai) * var[i] * var[i]);  // var = grad_s_to_h
        } 
""".replace("alpha", f"{self.alpha}").replace("pai", f"{self.pai}")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
        } 
""".replace("v_reset", f"{self.v_reset}")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = grad_spike_seq[index + i];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_h[i] = grad_h[i] * grad_h_next_to_v;
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

            var[i] = grad_h[i] * grad_h_to_x;
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
        }
        else
        {
            auto *ptr = (half *) &var[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}
"""
        return kernel_code, "LIFNodeBPTTHALFKernel"

    def get_cupy_kernel(self, dtype=torch.float32):
        forward_codes, kernel_name = self.get_cupy_forward_codes(dtype)
        self.cupy_forward_kernel = cp.RawKernel(forward_codes, kernel_name, backend="nvrtc")

        backward_codes, kernel_name = self.get_cupy_backward_codes(dtype)
        self.cupy_backward_kernel = cp.RawKernel(backward_codes, kernel_name, backend="nvrtc")

    def forward(self, x):
        if self.cupy_forward_kernel is None:
            self.get_cupy_kernel(x.dtype)

        with DeviceEnvironment(x.get_device()):
            spike_seq = self.sn_apply(x, self.cupy_forward_kernel, self.cupy_backward_kernel)

        return spike_seq


class PLIFNodeCuPy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, tau, forward_kernel, backward_kernel):
        inputs = inputs.contiguous()
        T, B, C, H, W = inputs.shape
        numel = inputs.numel() // T

        decay = tau.detach().cpu().item()

        spike_seq = torch.empty_like(inputs, dtype=inputs.dtype)
        h_seq = torch.empty_like(inputs, dtype=inputs.dtype)
        v_seq = torch.empty_like(inputs, dtype=inputs.dtype)

        threads = 256
        if inputs.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        forward_kernel(
            (blocks,), (threads,),
            (inputs.data_ptr(), spike_seq.data_ptr(), h_seq.data_ptr(), v_seq.data_ptr(), np.float32(decay), numel, T)
        )

        ctx.save_for_backward(h_seq, v_seq)
        ctx.decay = decay
        ctx.backward_kernel = backward_kernel

        return spike_seq

    @staticmethod
    def backward(ctx, grad_spike_seq):
        h_seq, v_seq = ctx.saved_tensors
        grad_spike_seq = grad_spike_seq.contiguous()

        grad_x_seq = torch.empty_like(h_seq, dtype=h_seq.dtype)
        grad_tau_seq = torch.zeros([1], device=h_seq.device, dtype=torch.float32, requires_grad=True)

        T = grad_spike_seq.shape[0]
        numel = grad_spike_seq.numel() // T
        backward_kernel = ctx.backward_kernel

        threads = 256
        if h_seq.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        backward_kernel(
            (blocks,), (threads,),
            (grad_spike_seq.data_ptr(), h_seq.data_ptr(), v_seq.data_ptr(),
             grad_x_seq.data_ptr(), grad_tau_seq.data_ptr(), np.float32(ctx.decay), numel, T)
        )

        return grad_x_seq, grad_tau_seq, None, None


class ParametricLIFNode(nn.Module):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w, dtype=torch.float32), requires_grad=True)

        self.decay_input = decay_input
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.store_v_seq = store_v_seq
        self.surrogate_func = surrogate_function
        self.cupy_forward_kernel = None
        self.cupy_backward_kernel = None
        self.hard_reset = True
        if self.v_reset is None:
            self.v_reset = 0.
            self.hard_reset = False

        self.alpha = 0

        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.alpha = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.alpha = surrogate_function.alpha * 2
            self.pai = math.pi * math.pi * surrogate_function.alpha * surrogate_function.alpha
        else:
            raise f"Surrogate: Sigmoid | ATan, alpha: {self.alpha}"
        self.sn_apply = PLIFNodeCuPy.apply

    def extra_repr(self):
        return f"ParametricLIFNode CuPy: decay={self.decay}, decay_input={self.decay_input}, v_threshold={self.v_th}," \
               f" v_reset={self.v_reset}, detach_reset={self.detach_reset}, store_v_seq={self.store_v_seq}"

    def get_cupy_forward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_forward_half()
        else:
            return self.get_cupy_codes_forward_float()

    def get_cupy_codes_forward_float(self):
        kernel_code = r'''
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void PLIFNodeFPTTFLOATKernel(
    float* __restrict__ inputs, 
    float* __restrict__ spikes_seq,
    float* __restrict__ h_seq,
    float* __restrict__ v_seq,
    float decay, int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;

        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }
'''
        # neuron charge
        if self.decay_input:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + (v[i] - (last_v[i] - v_reset)) * decay;
        }     
""".replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + v[i] - (last_v[i] - v_reset) * decay;
        }     
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }
"""

        # reset
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = spikes[i] > 0 ? v_reset : v[i];
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = v[i] - spikes[i] * v_th;
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
                v_seq[index + i] = last_v[i];
            }
        }
    }
}
"""

        return kernel_code, "PLIFNodeFPTTFLOATKernel"

    def get_cupy_codes_forward_half(self):
        kernel_code = r'''
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void PLIFNodeFPTTHALFKernel(
    half* __restrict__ inputs, 
    half* __restrict__ spikes_seq,
    half* __restrict__ h_seq,
    half* __restrict__ v_seq, 
    float decay, int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0); 
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;

        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }
'''

        if self.decay_input:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + (v[i] - (last_v[i] - __float2half2_rn(v_reset))) * __float2half2_rn(decay);
        }     
""".replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = last_v[i] + v[i] - (last_v[i] - __float2half2_rn(v_reset)) * __float2half2_rn(decay);
        }     
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }
"""
        # reset
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] * (__float2half2_rn(1.0f) - spikes[i]) + spikes[i] * __float2half2_rn(v_reset);  // hard reset
        }
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")
        else:
            kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = __hge2(v[i], __float2half2_rn(v_th));
            last_v[i] = v[i] - spikes[i] * __float2half2_rn(v_th);  // soft reset
        }     
""".replace("v_th", f"{self.v_th}f").replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            auto *spike_ptr = (half *) &spikes[0];
            auto *v_ptr = (half *) &last_v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spike_ptr[i];
                v_seq[index + i] = v_ptr[i];
            }
        }
    }
}
"""

        return kernel_code, "PLIFNodeFPTTHALFKernel"

    def get_cupy_backward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_backward_half()
        else:
            return self.get_cupy_codes_backward_float()

    def get_cupy_codes_backward_float(self):
        kernel_code = r"""
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void PLIFNodeBPTTFLOATKernel(
    float* __restrict__ grad_spike_seq, 
    float* __restrict__ h_seq, 
    float* __restrict__ v_seq, 
    float* __restrict__ grad_x_seq, 
    float* __restrict__ grad_tau_seq, 
    const float decay, const int numel, const int time_step)
{ 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    __shared__ float smem[8];

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float h[4], load[4];
    float var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;
"""

        if self.decay_input:
            kernel_code += r"""
    const float grad_h_to_x = decay;
    const float grad_h_next_to_v = 1.0f - decay;  
"""
        else:
            kernel_code += r"""
    const float grad_h_to_x = 1.0f;
    const float grad_h_next_to_v = 1.0f - decay;  
"""

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = 0;
        var[i] = 0;
        grad_h[i] = 0;
    }

    if (idx < numel)
    {
        int index;
        for (int t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;

            if (isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    h[i] = h_seq[index + i];
                    load[i] = grad_spike_seq[index + i];
                }
            }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th; // var = over_th
                grad_v_to_h[i] = var[i] < 0;
            }
""".replace("v_th", f"{self.v_th}f")
        else:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th; // var = over_th
                grad_v_to_h[i] = 1.0f;
            }
""".replace("v_th", f"{self.v_th}f")

        # surrogate_func
        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = 1.0f / (1.0f + expf(-alpha * var[i]));
                var[i] = (1.0f - var[i]) * var[i] * alpha;      // var = grad_s_to_h
            }
""".replace("alpha", f"{self.alpha}f")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = alpha / (4.0f + pai * var[i] * var[i]);  // var = grad_s_to_h
            }
""".replace("alpha", f"{self.alpha}f").replace("pai", f"{self.pai}f")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_v_to_h[i] += (v_reset - h[i]) * var[i]; // var = grad_s_to_h
            }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = var[i] * load[i]; // var = grad_s_to_h * grad_spike
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_h[i] = grad_h[i] * grad_h_next_to_v;
                grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

                var[i] = grad_h[i] * grad_h_to_x;
            }

            if (isLegalIndex)
            {
                FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    grad_x_seq[index + i] = var[i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = 0;
            }
            if (t > 0)
            {
                if (isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        load[i] = v_seq[index - numel + i];
                    }
                }
            }
        """

        if self.decay_input:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                var[i] = var[i] / decay;
                grad_tau += var[i];
            }
        }
    }
"""
        else:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = (v_reset - load[i]) * grad_h[i]; // load = v
                grad_tau += var[i];
            }
        }
    }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}
"""

        return kernel_code, "PLIFNodeBPTTFLOATKernel"

    def get_cupy_codes_backward_half(self):
        kernel_code = r"""
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void PLIFNodeBPTTHALFKernel(
    half* __restrict__ grad_spike_seq, 
    half* __restrict__ h_seq, 
    half* __restrict__ v_seq, 
    half* __restrict__ grad_x_seq, 
    float* __restrict__ grad_tau_seq,
    const float decay, const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
     __shared__ float smem[8];

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 h[4], load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;
"""

        if self.decay_input:
            kernel_code += r"""
    const half2 grad_h_to_x = __float2half2_rn(decay);
    const half2 grad_h_next_to_v = __float2half2_rn(1.0f - decay);  
"""
        else:
            kernel_code += r"""
    const half2 grad_h_to_x = __float2half2_rn(1.0f);
    const half2 grad_h_next_to_v = __float2half2_rn(1.0f - decay);  
"""

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = __float2half2_rn(0);
        var[i] = __float2half2_rn(0);
        grad_h[i] = __float2half2_rn(0);
    }

    if (idx < numel)
    {
        int index;
        for (int t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;
            
            if (isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                auto *h_ptr = (half * ) & h[0];
                auto *grad_spike_ptr = (half * ) & load[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    h_ptr[i] = h_seq[index + i];
                    grad_spike_ptr[i] = grad_spike_seq[index + i];
                }
            }
"""
        # grad_v_to_h
        if self.hard_reset:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - __float2half2_rn(v_th); // var = over_th | load = h
                grad_v_to_h[i] = __hle2(var[i], __float2half2_rn(0));
            } 
""".replace("v_th", f"{self.v_th}f")
        else:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = load[i] - __float2half2_rn(v_th); // var = over_th
                grad_v_to_h[i] = __float2half2_rn(1.0f);
            }  
""".replace("v_th", f"{self.v_th}f")

        # surrogate_func
        if isinstance(self.surrogate_func, surrogate.Sigmoid):
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = __float2half2_rn(1) / (__float2half2_rn(1) + h2exp(-__float2half2_rn(alpha) * var[i]));
                var[i] = (__float2half2_rn(1) - var[i]) * var[i] * __float2half2_rn(alpha);  // grad_s_to_h
            }
""".replace("alpha", f"{self.alpha}f")
        elif isinstance(self.surrogate_func, surrogate.ATan):
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = __float2half2_rn(alpha) / (__float2half2_rn(4.0f) + __float2half2_rn(pai) * var[i] * var[i]);  // var = grad_s_to_h
            }  
""".replace("alpha", f"{self.alpha}f").replace("pai", f"{self.pai}f")
        else:
            raise f"Surrogate: [Sigmoid | ATan]"

        if not self.detach_reset:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_v_to_h[i] += (__float2half2_rn(v_reset) - h[i]) * var[i]; // var = grad_s_to_h
            }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = var[i] * load[i]; // var = grad_s_to_h * grad_spike
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_h[i] = grad_h[i] * grad_h_next_to_v;
                grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

                var[i] = grad_h[i] * grad_h_to_x;
            }

            if (isLegalIndex)
            {
                FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
            }
            else
            {
                auto *ptr = (half * ) & var[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    grad_x_seq[index + i] = ptr[i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = __float2half2_rn(0);
            }
            if (t > 0)
            {
                if (isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    auto *ptr = (half * ) & load[0];
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        ptr[i] = v_seq[index - numel + i];
                    }
                }
            }
        """

        if self.decay_input:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                var[i] = var[i] / __float2half2_rn(decay);
                grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
            }
        }
    }
"""
        else:
            kernel_code += r"""
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = (v_reset - load[i]) * grad_h[i]; // load = v
                grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
            }
        }
    }
""".replace("v_reset", f"{self.v_reset}f")

        kernel_code += r"""
    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}
"""

        return kernel_code, "PLIFNodeBPTTHALFKernel"

    def get_cupy_kernel(self, dtype=torch.float32):
        forward_codes, kernel_name = self.get_cupy_forward_codes(dtype)
        self.cupy_forward_kernel = cp.RawKernel(forward_codes, kernel_name, backend="nvrtc")

        backward_codes, kernel_name = self.get_cupy_backward_codes(dtype)
        self.cupy_backward_kernel = cp.RawKernel(backward_codes, kernel_name, backend="nvrtc")

    def forward(self, x):
        if self.cupy_forward_kernel is None:
            self.get_cupy_kernel(x.dtype)

        with DeviceEnvironment(x.get_device()):
            spike_seq = self.sn_apply(x, self.w.sigmoid(), self.cupy_forward_kernel, self.cupy_backward_kernel)

        return spike_seq


class ILIFNodeCuPy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, forward_kernel, backward_kernel):
        inputs = inputs.contiguous()

        T = inputs.shape[0]
        numel = inputs.numel() // T

        spike_seq = torch.empty_like(inputs, dtype=inputs.dtype)
        h_seq = torch.empty_like(inputs, dtype=inputs.dtype)

        threads = 256
        if inputs.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        forward_kernel(
            (blocks,), (threads,),
            (inputs.data_ptr(), spike_seq.data_ptr(), h_seq.data_ptr(), numel, T)
        )

        ctx.save_for_backward(h_seq)
        ctx.backward_kernel = backward_kernel

        return spike_seq

    @staticmethod
    def backward(ctx, grad_spike_seq):
        h_seq = ctx.saved_tensors[0]
        grad_spike_seq = grad_spike_seq.contiguous()

        grad_x_seq = torch.empty_like(h_seq, dtype=h_seq.dtype)

        T = grad_spike_seq.shape[0]
        numel = grad_spike_seq.numel() // T
        backward_kernel = ctx.backward_kernel

        threads = 256
        if h_seq.dtype == torch.float16:
            blocks = (numel // 8 + threads - 1) // threads
        else:
            blocks = (numel // 4 + threads - 1) // threads
        backward_kernel(
            (blocks,), (threads,),
            (grad_spike_seq.data_ptr(), h_seq.data_ptr(), grad_x_seq.data_ptr(), numel, T)
        )

        return grad_x_seq, None, None


class ILIFNode(nn.Module):
    def __init__(self, decay: float = 0.25, min_value: float = 0.0, max_value: float = 4.0, store_v_seq: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.decay = decay
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.store_v_seq = store_v_seq

        self.cupy_forward_kernel = None
        self.cupy_backward_kernel = None

        self.sn_apply = ILIFNodeCuPy.apply

    def extra_repr(self):
        return f"I-LIFNode CuPy: decay={self.decay}, store_v_seq={self.store_v_seq}"

    def get_cupy_forward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_forward_half()
        else:
            return self.get_cupy_codes_forward_float()

    def get_cupy_codes_forward_float(self):
        kernel_code = r'''
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void ILIFNodeFPTTFLOATKernel(
    float* __restrict__ inputs, 
    float* __restrict__ spikes_seq, 
    float* __restrict__ h_seq,
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
        spikes[i] = 0;
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;
        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }
'''
        # neuron charge
        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = (last_v[i] - spikes[i]) * decay + v[i];
            last_v[i] = v[i];
        }
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }
"""

        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {      
            v[i] = max(v[i], min_value);
            v[i] = min(v[i], max_value);
            spikes[i] = floor(v[i] + 0.5f);
        }
""".replace("min_value", f"{self.min_value}f").replace("max_value", f"{self.max_value}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
            }
        }
    }
}
"""

        return kernel_code, "ILIFNodeFPTTFLOATKernel"

    def get_cupy_codes_forward_half(self):
        kernel_code = r'''
#include <cuda_fp16.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void ILIFNodeFPTTHALFKernel(
    half* __restrict__ inputs, 
    half* __restrict__ spikes_seq, 
    half* __restrict__ h_seq,
    int numel, int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0);
        spikes[i] = __float2half2_rn(0);
    }

    int index;
    for (int t = 0; t < time_step; t++) 
    {
        index = idx + numel * t;
        if (isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }
'''
        # neuron charge
        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            v[i] = (last_v[i] - spikes[i]) * __float2half2_rn(decay) + v[i];
            last_v[i] = v[i];
        }     
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }
"""

        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++) 
        {      
            v[i] = __hmax2(v[i], __float2half2_rn(min_value));
            v[i] = __hmin2(v[i], __float2half2_rn(max_value));
            v[i] += __float2half2_rn(0.5f);
            spikes[i] = h2floor(v[i]);
        }
""".replace("min_value", f"{self.min_value}f").replace("max_value", f"{self.max_value}f")

        kernel_code += r"""
        if (isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
        }
        else
        {
            auto *spike_ptr = (half *) &spikes[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spike_ptr[i];
            }
        }
    }
}
"""

        return kernel_code, "ILIFNodeFPTTHALFKernel"

    def get_cupy_backward_codes(self, dtype=torch.float32):
        if dtype == torch.float16:
            return self.get_cupy_codes_backward_half()
        else:
            return self.get_cupy_codes_backward_float()

    def get_cupy_codes_backward_float(self):
        kernel_code = r"""
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

extern "C" __global__ void ILIFNodeBPTTFLOATKernel(
    float* __restrict__ grad_spike_seq, 
    float* __restrict__ h_seq, 
    float* __restrict__ grad_x_seq,
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;

    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int edgeIndex = numel - idx;

    float h[4], grad_spike[4], grad_h[4];
    // grad_v_to_h, grad_h_to_x == 1
"""

        kernel_code += r"""
    const float grad_h_next_to_v = decay;  
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = 0;
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
            FETCH_FLOAT4(grad_spike[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h[i] = h_seq[index + i];;
                grad_spike[i] = grad_spike_seq[index + i];
            }
        }
"""

        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {     
            if (h[i] < min_value || h[i] > max_value)
            { grad_spike[i] = 0; }
        }
""".replace("min_value", f"{self.min_value}f").replace("max_value", f"{self.max_value}f")

        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_h[i] = grad_h[i] * grad_h_next_to_v + grad_spike[i];
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = grad_h[i];
            }
        }
    }
}
"""

        return kernel_code, "ILIFNodeBPTTFLOATKernel"

    def get_cupy_codes_backward_half(self):
        kernel_code = r"""
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#include <cuda_fp16.h>
extern "C" __global__ void ILIFNodeBPTTHALFKernel(
    half* __restrict__ grad_spike_seq, 
    half* __restrict__ h_seq, 
    half* __restrict__ grad_x_seq,
    const int numel, const int time_step)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;

    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int edgeIndex = numel - idx;

    half2 h[4], grad_spike[4], grad_h[4];
"""

        kernel_code += r"""
    const half2 grad_h_next_to_v = __float2half2_rn(decay);  
""".replace("decay", f"{self.decay}f")

        kernel_code += r"""
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = __float2half2_rn(0);
    }

    int index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;
        if (isLegalIndex)
        {
            FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
            FETCH_FLOAT4(grad_spike[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            auto *h_ptr = (half *) &h[0];
            auto *s_ptr = (half *) &grad_spike[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_ptr[i] = h_seq[index + i];
                s_ptr[i] = grad_spike_seq[index + i];
            }
        }
"""

        kernel_code += r"""
        auto *h_ptr = (half *) &h[0];
        auto *s_ptr = (half *) &grad_spike[0];
#pragma unroll
        for (int i = 0; i < 8; i++)
        {   // var = grad_s_to_h
            if (h_ptr[i] < __float2half_rn(min_value) || h_ptr[i] > __float2half_rn(max_value)) 
            { s_ptr[i] = __float2half_rn(0); }
        }
""".replace("min_value", f"{self.min_value}f").replace("max_value", f"{self.max_value}f")

        kernel_code += r"""
#pragma unroll
        for (int i = 0; i < 4; i++)
        {     
            grad_h[i] = grad_h[i] * grad_h_next_to_v + grad_spike[i];
        }

        if (isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            auto *ptr = (half *) &grad_h[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}
"""
        return kernel_code, "ILIFNodeBPTTHALFKernel"

    def get_cupy_kernel(self, dtype=torch.float32):
        forward_codes, kernel_name = self.get_cupy_forward_codes(dtype)
        self.cupy_forward_kernel = cp.RawKernel(forward_codes, kernel_name, backend="nvrtc")

        backward_codes, kernel_name = self.get_cupy_backward_codes(dtype)
        self.cupy_backward_kernel = cp.RawKernel(backward_codes, kernel_name, backend="nvrtc")

    def forward(self, x):
        if self.cupy_forward_kernel is None:
            self.get_cupy_kernel(x.dtype)

        with DeviceEnvironment(x.get_device()):
            spike_seq = self.sn_apply(x, self.cupy_forward_kernel, self.cupy_backward_kernel)

        return spike_seq
