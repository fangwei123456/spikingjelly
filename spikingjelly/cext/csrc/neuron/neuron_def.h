#pragma once
#define CHECK_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x" must be a CUDA tensor");if (! x.is_contiguous()){x = x.contiguous();}
#define CHECK_CUDA_OPERATION(operation) if(operation != cudaSuccess){printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));exit(-1);}
#define THREADS 1024


