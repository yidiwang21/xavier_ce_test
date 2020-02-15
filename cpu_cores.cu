#include "include.cuh"

void OpenMP_Gemm(const float *A, const float *B, float *C, const int M, const int K, const int N) {
    int lda = M, ldb = K, ldc = M;
    const float alpha = 1.0;
    const float beta = 0.0;

    int i, j, k;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            register float temp = alpha * A[i * lda + k];
            for (j = 0; j < N; j++) {
                C[i * ldc + j] += temp * B[k * ldb + j];
            }
        }
    }
}