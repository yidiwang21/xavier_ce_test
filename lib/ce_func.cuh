#include "include.cuh"

extern const char *_cublasGetStatusString(cublasStatus_t status);

// void GRC_Gemm (cublasHandle_t handle, const float *A, const float *B, float *C, const int M, const int K, const int N) {
//     // int lda = N, ldb = K, ldc = N;
//     int lda = M, ldb = K, ldc = M;
//     const float alpha = 1.0;
//     const float beta = 0.0;

//     cublasStatus_t status = cublasSgemm(handle, 
//                     CUBLAS_OP_N, CUBLAS_OP_N, 
//                     M, N, K, 
//                     &alpha, 
//                     A, lda, 
//                     B, ldb, 
//                     &beta, 
//                     C, ldc);

//     if (status != CUBLAS_STATUS_SUCCESS) 
//         printf("CUDA Error: %s\n", _cublasGetStatusString(status));
// }

// void GTC_Gemm (const float *A, const float *B, float *C, const int M, const int K, const int N) {
//     int lda = K, ldb = N, ldc = N;
//     const float alpha = 1.0;
//     const float beta = 0.0;

//     // set math mode to enable tensor cores
//     cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH) ;

//     cublasStatus_t status = cublasSgemm(handle, 
//                     CUBLAS_OP_N, CUBLAS_OP_N, 
//                     M, N, K, 
//                     &alpha, 
//                     A, lda, 
//                     B, ldb, 
//                     &beta, 
//                     C, ldc);

//     if (status != CUBLAS_STATUS_SUCCESS) 
//         printf("CUDA Error: %d\n", _cublasGetStatusString(status));
// }

struct cpu_thread_args {
    const float *A;
    const float *B; 
    float *C;
    const int M;
    const int K; 
    const int N;
};

// CPU core open mp implementation
// void OpenMP_Gemm(const float *A, const float *B, float *C, const int M, const int K, const int N) {
void *OpenMP_Gemm(void *vargp) {
    struct cpu_thread_args *myargs = (struct cpu_thread_args *)vargp;
    // width of matrices
    int lda = myargs->K, ldb = myargs->N, ldc = myargs->N;
    const float alpha = 1.0;
    const float beta = 0.0;

    int i, j, k;
    for (i = 0; i < myargs->M; i++) {
        for (j = 0; j < myargs->N; j++) {
            myargs->C[i * ldc + j] *= beta;
        }
    }
    for (i = 0; i < myargs->M; i++) {
        for (k = 0; k < myargs->K; k++) {
            for (j = 0; j < myargs->N; j++) {
                myargs->C[i * ldc + j] += alpha * myargs->A[i * lda + k] * myargs->B[k * ldb + j];
            }
        }
    }
}

// GPU regular core implementation
#define GRC_GEMM()  \
    int lda = M, ldb = K, ldc = M;  \
    cublasStatus_t status = cublasSgemm(handle, \
                    CUBLAS_OP_N, CUBLAS_OP_N, \
                    M, N, K, \
                    &alpha, \
                    A_d, lda, \
                    B_d, ldb, \
                    &beta, \
                    C_d, ldc);    \
    if (status != CUBLAS_STATUS_SUCCESS) \
        printf("CUDA Error: %s\n", _cublasGetStatusString(status));

// GPU tensor core implementation
#define GTC_GEMM()  \
    int lda = M, ldb = K, ldc = M;  \
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);   \
    cublasStatus_t status = cublasSgemm(handle, \
                    CUBLAS_OP_N, CUBLAS_OP_N, \
                    M, N, K, \
                    &alpha, \
                    A_d, lda, \
                    B_d, ldb, \
                    &beta, \
                    C_d, ldc);    \
    if (status != CUBLAS_STATUS_SUCCESS) \
        printf("CUDA Error: %s\n", _cublasGetStatusString(status));

// #define CPU_GEMM()  \
//     int lda = K, ldb = N, ldc = N;  \
//     int ii, jj, kk;    \
//     for (ii = 0; ii < M; ii++)   \
//         for (jj = 0; jj < N; jj++)  \
//             C_h[ii * ldc + jj] *= beta;   \
//     for (ii = 0; ii < M; ii++)  \
//         for (kk = 0; kk < K; kk++)  \
//             for (jj = 0; jj < N; jj++)  \
//                 C_h[ii * ldc + jj] += alpha * A_h[ii * lda + kk] * B_h[kk * ldb + jj];