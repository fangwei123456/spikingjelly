#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <cuda_fp16.h>
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

//LIF hard reset----------------------------------------------------
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

std::vector<at::Tensor> LIF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & reciprocal_tau)
{
  CHECK_TENSOR(x);
  CHECK_TENSOR(v);
  auto spike = torch::zeros_like(v.data());
  auto v_next = torch::zeros_like(v.data());
  CHECK_TENSOR(spike);
  CHECK_TENSOR(v_next);
  const int size = x.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x.get_device()));
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(),
    v_th, v_reset, size, reciprocal_tau);
  return {spike, v_next};
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

std::vector<at::Tensor> LIF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau)
{
  CHECK_TENSOR(x);
  CHECK_TENSOR(v);

  auto spike = torch::zeros_like(v.data());
  auto v_next = spike.data().clone();
  auto grad_s_to_h = spike.data().clone();
  auto grad_v_to_h = spike.data().clone();

  CHECK_TENSOR(spike);
  CHECK_TENSOR(v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  const int size = x.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x.get_device()));

  LIF_hard_reset_forward_with_grad_cuda_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(),
    grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
    v_th, v_reset, size,
    alpha, detach_reset, grad_surrogate_function_index,
    reciprocal_tau);

  return {spike, v_next, grad_s_to_h, grad_v_to_h};

}

//IF hard reset----------------------------------------------------
__global__ void IF_hard_reset_forward_cuda_kernel(
  const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next,
  const float v_th, const float v_reset, const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  const float h = v[index] + x[index] - v[index] + v_reset;
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

std::vector<at::Tensor> IF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset)
{
    CHECK_TENSOR(x);
    CHECK_TENSOR(v);
    auto spike = torch::zeros_like(v.data());
    auto v_next = torch::zeros_like(v.data());
    CHECK_TENSOR(spike);
    CHECK_TENSOR(v_next);
    const int size = x.numel();
    const int threads = THREADS;
    const int blocks = (size + threads - 1) / threads;
    CHECK_CUDA_OPERATION(cudaSetDevice(x.get_device()));
    IF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(),
      v_th, v_reset, size);
    return {spike, v_next};
}

__global__ void IF_hard_reset_forward_with_grad_cuda_kernel(
const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next,
float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h,
const float v_th, const float v_reset, const int size,
const float alpha, const bool detach_reset, const int grad_surrogate_function_index)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  const float h = v[index] + x[index];
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

std::vector<at::Tensor> IF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index)
{
  CHECK_TENSOR(x);
  CHECK_TENSOR(v);

  auto spike = torch::zeros_like(v.data());
  auto v_next = spike.data().clone();
  auto grad_s_to_h = spike.data().clone();
  auto grad_v_to_h = spike.data().clone();

  CHECK_TENSOR(spike);
  CHECK_TENSOR(v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  const int size = x.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x.get_device()));

  IF_hard_reset_forward_with_grad_cuda_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(),
    grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
    v_th, v_reset, size,
    alpha, detach_reset, grad_surrogate_function_index);
  return {spike, v_next, grad_s_to_h, grad_v_to_h};
}

//LIF hard reset fptt----------------------------------------------------
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

std::vector<at::Tensor> LIF_hard_reset_fptt(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & reciprocal_tau)
{
  CHECK_TENSOR(x_seq);
  CHECK_TENSOR(v);
  auto spike_seq = torch::zeros_like(x_seq.data());
  auto v_next = v.data().clone();
  CHECK_TENSOR(spike_seq);
  CHECK_TENSOR(v_next);
  const int seq_len = x_seq.size(0);
  const int size = x_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x_seq.get_device()));
  LIF_hard_reset_fptt_cuda_kernel<<<blocks, threads>>>(
    x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(),
    v_th, v_reset, neuron_num, size, reciprocal_tau);
  return {spike_seq, v_next};
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

std::vector<at::Tensor> LIF_hard_reset_fptt_with_grad(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau)
{
  CHECK_TENSOR(x_seq);
  CHECK_TENSOR(v);
  auto spike_seq = torch::zeros_like(x_seq.data());
  auto v_next = v.data().clone();
  auto grad_s_to_h = spike_seq.data().clone();
  auto grad_v_to_h = spike_seq.data().clone();
  CHECK_TENSOR(spike_seq);
  CHECK_TENSOR(v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  const int seq_len = x_seq.size(0);
  const int size = x_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x_seq.get_device()));
  LIF_hard_reset_fptt_with_grad_cuda_kernel<<<blocks, threads>>>(
    x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
    v_th, v_reset, neuron_num, size,
    alpha, detach_reset, grad_surrogate_function_index,
    reciprocal_tau);
  return {spike_seq, v_next, grad_s_to_h, grad_v_to_h};
}
//IF hard reset fptt----------------------------------------------------
__global__ void IF_hard_reset_fptt_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next,
  const float v_th, const float v_reset, const int neuron_num, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      const float h = v_next[index] + x_seq[mem_index];
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

std::vector<at::Tensor> IF_hard_reset_fptt(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset)
{
    CHECK_TENSOR(x_seq);
    CHECK_TENSOR(v);
    auto spike_seq = torch::zeros_like(x_seq.data());
    auto v_next = v.data().clone();
    CHECK_TENSOR(spike_seq);
    CHECK_TENSOR(v_next);
    const int seq_len = x_seq.size(0);
    const int size = x_seq.numel();
    const int threads = THREADS;
    const int neuron_num = size / seq_len;
    const int blocks = (neuron_num + threads - 1) / threads;
    CHECK_CUDA_OPERATION(cudaSetDevice(x_seq.get_device()));
    IF_hard_reset_fptt_cuda_kernel<<<blocks, threads>>>(
      x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(),
      v_th, v_reset, neuron_num, size);
    return {spike_seq, v_next};
}

__global__ void IF_hard_reset_fptt_with_grad_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next, float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h,
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      const float h = v_next[index] + x_seq[mem_index];
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

std::vector<at::Tensor> IF_hard_reset_fptt_with_grad(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index)
{
  CHECK_TENSOR(x_seq);
  CHECK_TENSOR(v);
  auto spike_seq = torch::zeros_like(x_seq.data());
  auto v_next = v.data().clone();
  auto grad_s_to_h = spike_seq.data().clone();
  auto grad_v_to_h = spike_seq.data().clone();
  CHECK_TENSOR(spike_seq);
  CHECK_TENSOR(v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  const int seq_len = x_seq.size(0);
  const int size = x_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(x_seq.get_device()));
  IF_hard_reset_fptt_with_grad_cuda_kernel<<<blocks, threads>>>(
    x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(),
    v_th, v_reset, neuron_num, size,
    alpha, detach_reset, grad_surrogate_function_index);
  return {spike_seq, v_next, grad_s_to_h, grad_v_to_h};
}