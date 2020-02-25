#include "include.cuh"
#include "ce_func.cuh"
#include "support.cuh"
#include "sm_alloc.cuh"
#include "common_snippet.cuh"
#include "cuda_profiler_api.h"
#include <thread>

struct gemm_thread_struct {
    cublasHandle_t hdl;
    int m;
    int k;
    int n;
    const float a;
    const float b;
    float *A_d;
    float *B_d;
    float *C_d;
};

extern __global__ void dummy_kernel();

std::mutex mtx;

// for dummy_kernel launch
void thread_dummy(cudaStream_t s, Timer timer, int nb) {
    mtx.lock();
    stopTime(&timer); printf("[1] Launching dummy kernel at: %f s\n", elapsedTime(timer));
    mtx.unlock();

    dummy_kernel <<< MAX_SM * nb, 1, 0, s >>> ();
    cudaStreamSynchronize(s);
    stopTime(&timer); printf("[1] dummy kernel completee at: %f s\n", elapsedTime(timer));
}

// for gemm on tensor core
void thread_gemm_tc(Timer timer, cudaStream_t mm_stream, cublasHandle_t handle, int M, int K, int N, const float alpha, const float beta, float *A_d, float *B_d, float * C_d) {
    mtx.lock();


    stopTime(&timer); printf("[2] Launching GEMM on tensor cores at: %f s\n", elapsedTime(timer));
    mtx.unlock();

    GRC_GEMM();
    cudaStreamSynchronize(mm_stream);
    stopTime(&timer); printf("[2] GEMM completee at: %f s\n", elapsedTime(timer));
}

int main(int argc, char *argv[]) {
    int opt;
    enum exec_mode EM = INDIVIDUAL;
    int reg_num_block_per_sm = 32;
    while ((opt = getopt (argc, argv, "m:l: ")) != EOF) {
        switch (opt) {
            case 'm':
                if (atoi(optarg) == 0) EM = INDIVIDUAL;
                else if (atoi(optarg) == 1) EM = CONCURRENT;
                break;
            case 'l':
                reg_num_block_per_sm = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "# Usage: \n -e <computing_elem> -s <number of SMs> -c <number of CPU cores>");
            default:
                printf("\n"); abort();
        }
    }


    // create stream for dummy_kernel
    cudaStream_t dummy_stream;
    cudaStreamCreateWithFlags(&dummy_stream, cudaStreamNonBlocking);

    cudaError_t cuda_ret;

    float time1, time2;
    Timer timer;
    startTime(&timer);

    MM_INIT_PARAM();
    MM_CREATE_CUDA_STREAM();
    MM_CREATE_HANDLE();
    MM_ALLOC_HOST_VARS();
    MM_ALLOC_DEVICE_VARS();
    MM_COPY_MEMORY_FROM_HOST_TO_DEVICE();

    cudaDeviceSynchronize();

    std::thread mythread1(thread_dummy, dummy_stream, timer, reg_num_block_per_sm);
    std::thread mythread2(thread_gemm_tc, timer, mm_stream, handle, M, K, N, alpha, beta, A_d, B_d, C_d);


    // printf("Launching dummy_kernel..."); fflush(stdout);
    // dummy_kernel <<< MAX_SM * reg_num_block_per_sm, 1, 0, dummy_stream >>> ();
    // if (EM == INDIVIDUAL) {
    //     cuda_ret = cudaDeviceSynchronize();
    //     stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    //     if (cuda_ret != cudaSuccess) {
    //         printf("Failed to launch dummy_kernel.\n");
    //         exit(1);
    //     }
    // }
    // printf("Launching GEMM on tensor cores..."); fflush(stdout);
    // stopTime(&timer);
    // time1 = elapsedTime(timer);
    // GTC_GEMM();

    // cudaStreamSynchronize(mm_stream);
    // cudaStreamSynchronize(mm_stream);
    mythread1.join();
    mythread2.join();


    MM_COPY_MEMORY_FROM_DEVICE_TO_HOST();
    cudaDeviceSynchronize();

    MM_FREE_DEVICE_VARS();
    MM_FREE_HOST_VARS();

    MM_DESTROY_HANDLE();

    cudaDeviceSynchronize();
    if (EM == INDIVIDUAL) {
        printf("GEMM compute time = %f ms\n", (time2 - time1) * 1000);
    }else {
        printf("GEMM compute time = %f ms\n", (time2 - time1 - 1) * 1000);
    }
    
    cudaProfilerStop();
    return 0;

}