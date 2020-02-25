#ifndef __COMMON_SNIPPET__
#define __COMMON_SNIPPET__

#include "include.cuh"

// MM means "matrix multiplication"

#define MM_INIT_PARAM() \
    int M = 1024;   \
    int K = 1024;   \
    int N = 2048;   \
    const float alpha = 1.0;    \
    const float beta = 0.0; \

#define MM_CREATE_CUDA_STREAM() \
    printf("Creating cuda stream for mm..."); fflush(stdout);   \
    cudaStream_t mm_stream;    \    
    cudaStreamCreateWithFlags(&mm_stream, cudaStreamNonBlocking);   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_CREATE_HANDLE()  \
    cublasHandle_t handle;  \
    cublasCreate(&handle);  \
    cublasSetStream(handle, mm_stream);

#define MM_DESTROY_HANDLE() \
    cublasDestroy(handle);

#define MM_ALLOC_HOST_VARS()    \
    printf("Allocating host variables..."); fflush(stdout); \
    float *A_h = (float *)malloc(M * K * sizeof(float));    \
    float *B_h = (float *)malloc(K * N * sizeof(float));    \
    float *C_h = (float *)malloc(M * N * sizeof(float));    \
    for (int i = 0; i < M * K; i++) { A_h[i] = (rand()%100)/100.00; }   \
    for (int i = 0; i < K * N; i++) { B_h[i] = (rand()%100)/100.00; }   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_ALLOC_DEVICE_VARS() \
    float *A_d, *B_d, *C_d; \
    printf("Allocating device variables..."); fflush(stdout);   \
    cudaMalloc((void **)&A_d, M * K * sizeof(float));   \
    cudaMalloc((void **)&B_d, K * N * sizeof(float));   \
    cudaMalloc((void **)&C_d, M * N * sizeof(float));   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_COPY_MEMORY_FROM_HOST_TO_DEVICE()    \
    printf("Copying data from host to device..."); fflush(stdout);  \
    cudaMemcpyAsync(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice, mm_stream);   \
    cudaMemcpyAsync(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice, mm_stream);   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_COPY_MEMORY_FROM_DEVICE_TO_HOST()    \
    printf("Copying data from device to host..."); fflush(stdout);  \
    cudaMemcpyAsync(C_d, C_h, M * N * sizeof(float), cudaMemcpyDeviceToHost, mm_stream);   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_FREE_DEVICE_VARS()   \
    printf("Freeing device variables..."); fflush(stdout);  \
    cudaFree(A_d);  \
    cudaFree(B_d);  \
    cudaFree(C_d);  \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define MM_FREE_HOST_VARS()    \
    printf("Freeing host variables..."); fflush(stdout);  \
    free(A_h);  \
    free(B_h);  \
    free(C_h);  \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));






#endif __COMMON_SNIPPET__