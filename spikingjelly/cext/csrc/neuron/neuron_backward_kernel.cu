#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"

//LIF bp----------------------------------------------------
__global__ void LIF_backward_cuda_kernel(
    float* __restrict__ grad_x, float* __restrict__ grad_v,
    const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
    const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
    grad_x[index] = grad_h * reciprocal_tau;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
  }
}

void LIF_backward_cuda(
  float* grad_x, float* grad_v,
  const float* grad_spike, const float* grad_v_next, const float* grad_s_to_h, const float* grad_v_to_h,
  const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_backward_cuda_kernel<<<blocks, threads>>>(
    grad_x, grad_v, grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h,
    size, 
    reciprocal_tau, 1 - reciprocal_tau
  );
}


//LIF bptt----------------------------------------------------

__global__ void LIF_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
  const int neuron_num, const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    float grad_h;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
      grad_x_seq[mem_index] = grad_h * reciprocal_tau;
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
    }
  }
}

void LIF_bptt_cuda(
  float* grad_x_seq, float* grad_v, 
  const float* grad_spike_seq, const float* grad_s_to_h, const float* grad_v_to_h,
  const int & seq_len, const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  LIF_bptt_cuda_kernel<<<blocks, threads>>>(
    grad_x_seq, grad_v,
    grad_spike_seq, grad_s_to_h, grad_v_to_h, 
    neuron_num, size,
    reciprocal_tau, 1 - reciprocal_tau
  );
}

//IF bp----------------------------------------------------
__global__ void IF_backward_cuda_kernel(
  float* __restrict__ grad_x, float* __restrict__ grad_v,
  const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
  const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
  grad_x[index] = grad_h;
  grad_v[index] = grad_h;
}
}

void IF_backward_cuda(
float* grad_x, float* grad_v,
const float* grad_spike, const float* grad_v_next, const float* grad_s_to_h, const float* grad_v_to_h,
const int & size, const int & gpu_id)
{
const int threads = THREADS;
const int blocks = (size + threads - 1) / threads;
CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
IF_backward_cuda_kernel<<<blocks, threads>>>(
  grad_x, grad_v, grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h,
  size);
}


//IF bptt----------------------------------------------------

__global__ void IF_bptt_cuda_kernel(
float* __restrict__ grad_x_seq, float* __restrict__ grad_v,
const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
const int neuron_num, const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < neuron_num)
{
  float grad_h;
  for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
  {
    const int mem_index = index + mem_offset;
    grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
    grad_x_seq[mem_index] = grad_h;
    grad_v[index] = grad_h;
  }
}
}

void IF_bptt_cuda(
float* grad_x_seq, float* grad_v, 
const float* grad_spike_seq, const float* grad_s_to_h, const float* grad_v_to_h,
const int & seq_len, const int & size, const int & gpu_id)
{
CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
const int threads = THREADS;
const int neuron_num = size / seq_len;
const int blocks = (neuron_num + threads - 1) / threads;
IF_bptt_cuda_kernel<<<blocks, threads>>>(
  grad_x_seq, grad_v,
  grad_spike_seq, grad_s_to_h, grad_v_to_h, 
  neuron_num, size);
}