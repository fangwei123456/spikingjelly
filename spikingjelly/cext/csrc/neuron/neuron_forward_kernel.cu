#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"
__forceinline__  __device__ float grad_atan(const float & alpha, const float & x)
{
  const float M_PI_2__alpha__x = M_PI_2 * alpha * x;
  return alpha / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
}

__forceinline__  __device__ float grad_sigmoid(const float & alpha, const float & x)
{
  const float sigmoid_ax = 1.0f / (1.0f + expf(- alpha * x));
  return (1 - sigmoid_ax) * sigmoid_ax * alpha;
}

typedef float (*grad_surrogate_function) (const float &, const float &);

__device__ const grad_surrogate_function grad_surrogate_function_pointer[2] = { 
    grad_atan, 
    grad_sigmoid
    };


__global__ void LIF_hard_reset_forward_cuda_kernel(
    const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const float h = v[index] + reciprocal_tau * (x[index] - v[index] + v_reset);
    if (h >= v_th)
    {
      spike[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else
    {
      spike[index] = 0.0f;
      v_next[index] = h;
    }
  }
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* spike, float* v_next, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, spike, v_next, v_th, v_reset, size, reciprocal_tau);
}

__global__ void LIF_hard_reset_forward_with_grad_cuda_kernel(
  const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next,
  float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h,
  const float v_th, const float v_reset, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const float h = v[index] + reciprocal_tau * (x[index] - v[index] + v_reset);
    if (h >= v_th)
    {
      spike[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else
    {
      spike[index] = 0.0f;
      v_next[index] = h;
    }
    grad_s_to_h[index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
    grad_v_to_h[index] = 1 - spike[index] + (v_reset - h) * grad_s_to_h[index] * (1 - detach_reset);
  }
}

void LIF_hard_reset_forward_with_grad_cuda(
  const float* x, const float* v, float* spike, float* v_next,
  float* grad_s_to_h, float* grad_v_to_h,
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_hard_reset_forward_with_grad_cuda_kernel<<<blocks, threads>>>(
    x, v, spike, v_next, 
    grad_s_to_h, grad_v_to_h, 
    v_th, v_reset, size, 
    alpha, detach_reset, grad_surrogate_function_index,
    reciprocal_tau);
}


//fptt-------------------------------
__global__ void LIF_hard_reset_fptt_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next, 
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      const float h = v_next[index] + reciprocal_tau * (x_seq[mem_index] - v_next[index] + v_reset);
      if (h >= v_th)
      {
        spike_seq[mem_index] = 1.0f;
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = 0.0f;
        v_next[index] = h;
      }
    }
    
  }
}

void LIF_hard_reset_fptt_cuda(const float* x_seq, float* spike_seq, float* v_next, 
  const float & v_th, const float & v_reset, const int & seq_len, const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_hard_reset_fptt_cuda_kernel<<<blocks, threads>>>(x_seq, spike_seq, v_next, v_th, v_reset, neuron_num, size, reciprocal_tau);
}

__global__ void LIF_hard_reset_fptt_with_grad_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next, float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h, 
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      const float h = v_next[index] + reciprocal_tau * (x_seq[mem_index] - v_next[index] + v_reset);
      if (h >= v_th)
      {
        spike_seq[mem_index] = 1.0f;
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = 0.0f;
        v_next[index] = h;
      }
      grad_s_to_h[mem_index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
      grad_v_to_h[mem_index] = 1 - spike_seq[mem_index] + (v_reset - h) * grad_s_to_h[mem_index] * (1 - detach_reset);
    }
    
  }
}

void LIF_hard_reset_fptt_with_grad_cuda(
  const float* x_seq, float* spike_seq, float* v_next, float* grad_s_to_h, float* grad_v_to_h,
  const float & v_th, const float & v_reset, 
  const int & seq_len, const int & size, const int & gpu_id, 
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_hard_reset_fptt_with_grad_cuda_kernel<<<blocks, threads>>>(
    x_seq, spike_seq, v_next, grad_s_to_h, grad_v_to_h,
    v_th, v_reset, neuron_num, size, 
    alpha, detach_reset, grad_surrogate_function_index,
    reciprocal_tau);
}
