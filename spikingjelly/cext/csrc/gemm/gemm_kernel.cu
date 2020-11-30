#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <assert.h>
using namespace std;

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
    fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
    exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

template<class T>
struct reCuBuffer
{
    T* data = NULL;
    int len = 0;
};

template<class T>
void resize(reCuBuffer<T>& buffer, int size)
{
    if(size > buffer.len)
    {
        if(buffer.len > 0)
            CUDA_CALL(cudaFree(buffer.data));
            
        CUDA_CALL(cudaMalloc( &(buffer.data), size));
        buffer.len = size;
    }
}

#define num_device 16

static reCuBuffer<int>   nnzPerCol_[num_device], ColInd_[num_device], RowPtr_[num_device];
static reCuBuffer<float> csrVal_[num_device], tranBuffer_[num_device];
static reCuBuffer<void>  dBuffer_[num_device];

struct cublasHandle_
{
    cublasHandle_t handle_;
    bool init = false;
};
static cublasHandle_ handle2_[num_device];

void sparse_mm_dense_cusparse_backend(const int & cuda_device_id, const int & m, const int & n, const int & p, float * dA, float * dB, float * dC)
{
    assert(cuda_device_id>=0);
    cudaSetDevice(cuda_device_id);

    reCuBuffer<int>& nnzPerCol    = nnzPerCol_[cuda_device_id];
    reCuBuffer<int>& ColInd       = ColInd_[cuda_device_id];
    reCuBuffer<int>& RowPtr       = RowPtr_[cuda_device_id];
    reCuBuffer<float>& csrVal     = csrVal_[cuda_device_id];

    cusparseHandle_t  handle;
    CUSPARSE_CALL(cusparseCreate(&handle));

    // transform dense A to csr
    cusparseMatDescr_t descrX;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));

    int total_nnz;
    resize(nnzPerCol, m * sizeof(int));

    CUSPARSE_CALL(cusparseSnnz(handle, CUSPARSE_DIRECTION_COLUMN, n, m, descrX, dA, n, nnzPerCol.data, &total_nnz));
    resize(csrVal, total_nnz * sizeof(float));
    resize(ColInd, total_nnz * sizeof(int));
    resize(RowPtr, (m+1) * sizeof(int));

    CUSPARSE_CALL(cusparseSdense2csc(handle, n, m, descrX, dA, n, nnzPerCol.data, csrVal.data, ColInd.data, RowPtr.data));

#if __CUDACC_VER_MAJOR__ == 10
    reCuBuffer<float>& tranBuffer = tranBuffer_[cuda_device_id];

    // CT = A * BT
    resize(tranBuffer, m * p * sizeof(float));

    // B * C
    cusparseMatDescr_t descrA;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CALL(cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));

    float alpha = 1.0f;
    float beta  = 0.0f;
    CUSPARSE_CALL(cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,
                  m,p,n,total_nnz,&alpha,descrA,csrVal.data,RowPtr.data, ColInd.data,dB,p,&beta,tranBuffer.data,m));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrA));

    // cublasDestroy will synchronize the device
    cublasHandle_t& handle2 = handle2_[cuda_device_id].handle_;
    if(!handle2_[cuda_device_id].init)
    {
        cublasCreate(&handle2);
        handle2_[cuda_device_id].init = true;
    }

    // C need TRANSPOSE
    cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_T, p, m, &alpha, tranBuffer.data, m, &beta, tranBuffer.data, m, dC, p);
    //cublasDestroy(handle2);
#endif

#if __CUDACC_VER_MAJOR__ == 11
    reCuBuffer<void>& dBuffer = dBuffer_[cuda_device_id];

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // Create sparse matrix A in CSR format
    CUSPARSE_CALL(cusparseCreateCsr(&matA, m, n, total_nnz, RowPtr.data, ColInd.data, csrVal.data,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense matrix B
    int ldb = p;
    CUSPARSE_CALL(cusparseCreateDnMat(&matB, n, p, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    int ldc = p;
    CUSPARSE_CALL(cusparseCreateDnMat(&matC, m, p, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate an external buffer if needed
    float alpha = 1.0f;
    float beta  = 0.0f;
    size_t bufferSize = 0;
    CUSPARSE_CALL(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    resize(dBuffer, bufferSize);

    // execute SpMM
    CUSPARSE_CALL(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer.data));

    // destroy matrix/vector descriptors
    CUSPARSE_CALL(cusparseDestroySpMat(matA));
    CUSPARSE_CALL(cusparseDestroyDnMat(matB));
    CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#endif

    CUSPARSE_CALL(cusparseDestroy(handle));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrX));
}
