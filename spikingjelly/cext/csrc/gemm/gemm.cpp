#include <iostream>
#include <torch/extension.h>

void sparse_mm_dense_cusparse_backend(const int & cuda_device_id, const int & m, const int & n, const int & p, float * dA, float * dB, float * dC);

void sparse_mm_dense_cusparse(torch::Tensor & A, torch::Tensor & B, torch::Tensor & C)
{   
    // A is sparse, B is dense
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");

    if (! A.is_contiguous())
    {
        A = A.contiguous();
    }
    if (! B.is_contiguous())
    {
        B = B.contiguous();
    }
    if (! C.is_contiguous())
    {
        C = C.contiguous();
    }
    
    // A: [M, N] B:[N, P]
    // C: [M, P]
    // Mat size. In cuSparse, dense matrix is column-major format while sparse matrix is row-major.So we use csc instead of csr
    int m = A.size(0);
    int n = A.size(1);
    int p = B.size(1);
    sparse_mm_dense_cusparse_backend(B.get_device(), m, n, p, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mm_dense_cusparse", &sparse_mm_dense_cusparse);
}