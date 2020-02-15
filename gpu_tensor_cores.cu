#include "include.cuh"

extern const char *_cublasGetStatusString(cublasStatus_t status);

void GTC_Gemm (const float *A, const float *B, float *C, const int M, const int K, const int N) {
    int lda = M, ldb = K, ldc = M;
    const float alpha = 1.0;
    const float beta = 0.0;

    // create CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // set math mode to enable tensor cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH) ;

    cublasStatus_t status = cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A, lda, 
                    B, ldb, 
                    &beta, 
                    C, ldc);

    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("CUDA Error: %d\n", _cublasGetStatusString(status));

    // do not destroy the handle if more than one mmul need to be done
    cublasDestroy(handle);
}