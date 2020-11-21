#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
// 在surrogate.py中通过控制use_fast_math，可以调整是否使用快速近似。默认开启此选项

void alpha_backward_cuda_base(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size, 
  void alpha_backward_cuda_kernel(const float* __restrict__, const float* __restrict__, const float, float* __restrict__, const int)
)
{
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  alpha_backward_cuda_kernel<<<blocks, threads>>>(grad_output, x, alpha, grad_x, size);
}

// atan-----------------------------------------------------------------------
__global__ void alpha_atan_backward_cuda_kernel(const float* __restrict__ grad_output, const float* __restrict__ x, const float alpha,
  float* __restrict__ grad_x, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      grad_x[index] = alpha / 2.0f / (1.0f + powf(M_PI_2 * alpha * x[index], 2)) * grad_output[index];
  }
}

void alpha_atan_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size)
{
  alpha_backward_cuda_base(grad_output, x, alpha, grad_x, size, alpha_atan_backward_cuda_kernel);
}

// sigmoid-----------------------------------------------------------------------
__global__ void alpha_sigmoid_backward_cuda_kernel(const float* __restrict__ grad_output, const float* __restrict__ x, const float alpha,
  float* __restrict__ grad_x, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      const float sigmoid_ax = 1.0f / (1.0f + expf(- alpha * x[index]));
      grad_x[index] = grad_output[index] * (1 - sigmoid_ax) * sigmoid_ax * alpha;
  }
}

void alpha_sigmoid_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size)
{
  alpha_backward_cuda_base(grad_output, x, alpha, grad_x, size, alpha_sigmoid_backward_cuda_kernel);
}
