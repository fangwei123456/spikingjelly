#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
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

__global__ void LIF_backward_cuda_kernel_half(
  c10::Half* __restrict__ grad_x, c10::Half* __restrict__ grad_v,
  const c10::Half* __restrict__ grad_spike, const c10::Half* __restrict__ grad_v_next, const c10::Half* __restrict__ grad_s_to_h, const c10::Half* __restrict__ grad_v_to_h,
  const int size,
  const half reciprocal_tau, const half one_sub_reciprocal_tau)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  const half grad_h = __hfma(grad_spike[index], grad_s_to_h[index], __hmul(grad_v_next[index], grad_v_to_h[index]));
  grad_x[index] = __hmul(grad_h, reciprocal_tau);
  grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
}
}
//detach x--------------
__global__ void LIF_detach_x_backward_cuda_kernel(
  float* __restrict__ grad_x, float* __restrict__ grad_v,
  const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
  const int size,
  const float one_sub_reciprocal_tau)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
  grad_x[index] = grad_h;
  grad_v[index] = grad_h * one_sub_reciprocal_tau;
}
}

__global__ void LIF_detach_x_backward_cuda_kernel_half(
c10::Half* __restrict__ grad_x, c10::Half* __restrict__ grad_v,
const c10::Half* __restrict__ grad_spike, const c10::Half* __restrict__ grad_v_next, const c10::Half* __restrict__ grad_s_to_h, const c10::Half* __restrict__ grad_v_to_h,
const int size,
const half one_sub_reciprocal_tau)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
const half grad_h = __hfma(grad_spike[index], grad_s_to_h[index], __hmul(grad_v_next[index], grad_v_to_h[index]));
grad_x[index] = grad_h;
grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
}
}

std::vector<at::Tensor> LIF_backward(
  torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h,
  const float & reciprocal_tau, const bool & detach_x)
{
  CHECK_TENSOR(grad_spike);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  auto grad_x = torch::zeros_like(grad_spike.data());
  auto grad_v = grad_x.data().clone();
  CHECK_TENSOR(grad_x);
  CHECK_TENSOR(grad_v);
  const int size = grad_spike.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike.get_device()));
  if (grad_x.scalar_type() == c10::ScalarType::Float)
  {
    if (detach_x)
    {
      LIF_detach_x_backward_cuda_kernel<<<blocks, threads>>>(
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
        size, 1.0f - reciprocal_tau);
    }
    else
    {
      LIF_backward_cuda_kernel<<<blocks, threads>>>(
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
        size, reciprocal_tau, 1.0f - reciprocal_tau);
    }

  }
  else if (grad_x.scalar_type() == c10::ScalarType::Half)
  {
    if (detach_x)
    {
      LIF_detach_x_backward_cuda_kernel_half<<<blocks, threads>>>(
        grad_x.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
        grad_spike.data_ptr<at::Half>(), grad_v_next.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
        size, __float2half(1.0f - reciprocal_tau));
    }
    else
    {
      LIF_backward_cuda_kernel_half<<<blocks, threads>>>(
        grad_x.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
        grad_spike.data_ptr<at::Half>(), grad_v_next.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
        size, __float2half(reciprocal_tau), __float2half(1.0f - reciprocal_tau));
    }

  }
  
  return {grad_x, grad_v};
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

__global__ void LIF_bptt_cuda_kernel_half(
  at::Half* __restrict__ grad_x_seq, at::Half* __restrict__ grad_v,
  const at::Half* __restrict__ grad_spike_seq, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h,
  const int neuron_num, const int size,
  const half reciprocal_tau, const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    half grad_h;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = __hfma(grad_spike_seq[mem_index], grad_s_to_h[mem_index], __hmul(grad_v[index], grad_v_to_h[mem_index]));
      grad_x_seq[mem_index] = __hmul(grad_h, reciprocal_tau);
      grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
    }
  }
}

//detach x------

__global__ void LIF_detach_x_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
  const int neuron_num, const int size,
  const float one_sub_reciprocal_tau)
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
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
    }
  }
}

__global__ void LIF_detach_x_bptt_cuda_kernel_half(
  at::Half* __restrict__ grad_x_seq, at::Half* __restrict__ grad_v,
  const at::Half* __restrict__ grad_spike_seq, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h,
  const int neuron_num, const int size,
  const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    half grad_h;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = __hfma(grad_spike_seq[mem_index], grad_s_to_h[mem_index], __hmul(grad_v[index], grad_v_to_h[mem_index]));
      grad_x_seq[mem_index] = grad_h;
      grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
    }
  }
}

std::vector<at::Tensor> LIF_bptt(
  torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
  torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h,
  const float & reciprocal_tau, const bool & detach_x)
{
  CHECK_TENSOR(grad_spike_seq);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  auto grad_x_seq = torch::zeros_like(grad_spike_seq.data());
  auto grad_v = grad_v_next.data().clone();
  CHECK_TENSOR(grad_x_seq);
  CHECK_TENSOR(grad_v);
  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike_seq.get_device()));
  const int seq_len = grad_spike_seq.size(0);
  const int size = grad_spike_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  if (grad_x_seq.scalar_type() == c10::ScalarType::Float)
  {
    if (detach_x)
    {
      LIF_detach_x_bptt_cuda_kernel<<<blocks, threads>>>(
        grad_x_seq.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_spike_seq.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
        neuron_num, size,
        1.0f - reciprocal_tau);
    }
    else
    {
      LIF_bptt_cuda_kernel<<<blocks, threads>>>(
        grad_x_seq.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_spike_seq.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
        neuron_num, size,
        reciprocal_tau, 1.0f - reciprocal_tau);
    }

  }
  else if (grad_x_seq.scalar_type() == c10::ScalarType::Half)
  {
    if (detach_x)
    {
      LIF_detach_x_bptt_cuda_kernel_half<<<blocks, threads>>>(
        grad_x_seq.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
        grad_spike_seq.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
        neuron_num, size,
        __float2half(1.0f - reciprocal_tau));
    }
    else
    {
      LIF_bptt_cuda_kernel_half<<<blocks, threads>>>(
        grad_x_seq.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
        grad_spike_seq.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
        neuron_num, size,
        __float2half(reciprocal_tau), __float2half(1.0f - reciprocal_tau));
    }

  }
  return {grad_x_seq, grad_v};
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

__global__ void IF_backward_cuda_kernel_half(
  at::Half* __restrict__ grad_x, at::Half* __restrict__ grad_v,
  const at::Half* __restrict__ grad_spike, const at::Half* __restrict__ grad_v_next, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h,
  const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const half grad_h = __hfma(grad_spike[index], grad_s_to_h[index], __hmul(grad_v_next[index], grad_v_to_h[index]));
    grad_x[index] = grad_h;
    grad_v[index] = grad_h;
  }
}

std::vector<at::Tensor> IF_backward(
  torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h)
{
  CHECK_TENSOR(grad_spike);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  auto grad_x = torch::zeros_like(grad_spike.data());
  auto grad_v = grad_x.data().clone();
  CHECK_TENSOR(grad_x);
  CHECK_TENSOR(grad_v);
  const int size = grad_spike.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike.get_device()));
  if (grad_spike.scalar_type() == c10::ScalarType::Float)
  {
    IF_backward_cuda_kernel<<<blocks, threads>>>(
      grad_x.data_ptr<float>(), grad_v.data_ptr<float>(),
      grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
      size);
  }
  else if (grad_spike.scalar_type() == c10::ScalarType::Half)
  {
    IF_backward_cuda_kernel_half<<<blocks, threads>>>(
      grad_x.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
      grad_spike.data_ptr<at::Half>(), grad_v_next.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
      size);
  }


  return {grad_x, grad_v};
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

__global__ void IF_bptt_cuda_kernel_half(
  at::Half* __restrict__ grad_x_seq, at::Half* __restrict__ grad_v,
  const at::Half* __restrict__ grad_spike_seq, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h,
  const int neuron_num, const int size)
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < neuron_num)
    {
      half grad_h;
      for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
      {
        const int mem_index = index + mem_offset;
        grad_h = __hfma(grad_spike_seq[mem_index], grad_s_to_h[mem_index], __hmul(grad_v[index], grad_v_to_h[mem_index]));
        grad_x_seq[mem_index] = grad_h;
        grad_v[index] = grad_h;
      }
    }
  }

std::vector<at::Tensor> IF_bptt(
  torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
  torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h)
{
  CHECK_TENSOR(grad_spike_seq);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  auto grad_x_seq = torch::zeros_like(grad_spike_seq.data());
  auto grad_v = grad_v_next.data().clone();
  CHECK_TENSOR(grad_x_seq);
  CHECK_TENSOR(grad_v);
  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike_seq.get_device()));
  const int seq_len = grad_spike_seq.size(0);
  const int size = grad_spike_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  if (grad_x_seq.scalar_type() == c10::ScalarType::Float)
  {
    IF_bptt_cuda_kernel<<<blocks, threads>>>(
      grad_x_seq.data_ptr<float>(), grad_v.data_ptr<float>(),
      grad_spike_seq.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
      neuron_num, size);
  }
  else if (grad_x_seq.scalar_type() == c10::ScalarType::Half)
  {
    IF_bptt_cuda_kernel_half<<<blocks, threads>>>(
      grad_x_seq.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(),
      grad_spike_seq.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(),
      neuron_num, size);
  }

  return {grad_x_seq, grad_v};
}
