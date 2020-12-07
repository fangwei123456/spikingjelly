#pragma once
#define CHECK_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x" must be a CUDA tensor"); if (! x.is_contiguous()){x = x.contiguous();}
#define INIT_DEVIDE_THREAD const int threads = 1024;const int blocks = (size + threads - 1) / threads;cudaError_t error = cudaSetDevice(gpu_id);if(error != cudaSuccess){printf("CUDA error: %s\n", cudaGetErrorString(error));exit(-1);}




