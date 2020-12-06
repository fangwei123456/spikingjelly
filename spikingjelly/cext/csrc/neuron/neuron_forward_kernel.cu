#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"
#define HARD_RESET_FORWARD_CUDA_KERNEL(charge_function) do { \
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    h[index] = charge_function; \
    if (h[index] >= v_th) \
    { \
      spike[index] = 1.0f; \
      v_next[index] = v_reset; \
    } \
    else \
    { \
      spike[index] = 0.0f; \
      v_next[index] = h[index]; \
    } \
  } \
} while(0) \

__global__ void LIF_hard_reset_forward_cuda_kernel(
    const float* __restrict__ x, const float* __restrict__ v,  float* __restrict__ h,  float* __restrict__ spike, float* __restrict__ v_next, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  HARD_RESET_FORWARD_CUDA_KERNEL(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* h, float* spike, float* v_next, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & tau)
{
  INIT_DEVIDE_THREAD;
  const float reciprocal_tau = 1 / tau;
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, h, spike, v_next, v_th, v_reset, size, reciprocal_tau, 1 - reciprocal_tau);
}

//fptt-------------------------------
__global__ void LIF_hard_reset_fptt_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ h_seq,  float* __restrict__ spike_seq, float* __restrict__ v_next, 
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      h_seq[mem_index] = one_sub_reciprocal_tau * v_next[index] + reciprocal_tau * (x_seq[mem_index] + v_reset);

      if (h_seq[mem_index] >= v_th)
      {
        spike_seq[mem_index] = 1.0f;
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = 0.0f;
        v_next[index] = h_seq[mem_index];
      }
    }
    
  }
}

void LIF_hard_reset_fptt_cuda(const float* x_seq, float* h_seq, float* spike_seq, float* v_next, 
  const float & v_th, const float & v_reset, const int & seq_len, const int & size, const int & gpu_id, 
  const float & tau)
{
  const int threads = 1024;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  cudaError_t error = cudaSetDevice(gpu_id);
  if(error != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  const float reciprocal_tau = 1 / tau;
  LIF_hard_reset_fptt_cuda_kernel<<<blocks, threads>>>(x_seq, h_seq, spike_seq, v_next, v_th, v_reset, neuron_num, size, reciprocal_tau, 1 - reciprocal_tau);
}