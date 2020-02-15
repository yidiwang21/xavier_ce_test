#include "include.cuh"
#include "support.cuh"

#define _GNU_SOURCE

// gpu regular core implementation
extern void GRC_Gemm (const float *A, const float *B, float *C, const int M, const int K, const int N);
// gpu tensor core implementation
extern void GTC_Gemm (const float *A, const float *B, float *C, const int M, const int K, const int N);
// cpu open mp implementation
extern void OpenMP_Gemm(const float *A, const float *B, float *C, const int M, const int K, const int N);

// TODO: fix or not fix freq

int main(int argc, char *argv[]) {
    // reserve 2 cpus for this process
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    CPU_SET(1, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        printf("sched_setaffinity for the main process failed.\n");
        exit(1);
    }

    // main program starts
    int opt;
    enum computing_elem CE = GPU_REG_CORES;   // default computing element is GPU regular cores
    while ((opt = getopt (argc, argv, "c: ")) != EOF) {
        switch (opt) {
            case 'c':       // computing element
                if (atoi(optarg) == 0) CE = CPU_CORES; 
                else if (atoi(optarg) == 1) CE = GPU_REG_CORES;
                else if (atoi(optarg) == 2) CE = GPU_TENSOR_CORES;
                break;
            case '?':
                fprintf(stderr, "# Usage: \n -c <computing_elem>");
            default:
                printf("\n"); abort();
        }
    }

    switch (CE) {
        case CPU_CORES:
            printf("Computing element: CPU cores\n"); fflush(stdout);
            break;
        case GPU_REG_CORES:
            printf("Computing element: GPU regular cores\n"); fflush(stdout);
            break;
        case GPU_TENSOR_CORES:
            printf("Computing element: GPU tensor cores\n"); fflush(stdout);
            break;
    }
    printf("---------------------------------------------------------\n");

    srand(time(NULL));

    Timer timer;
    startTime(&timer);

    int M = 1024;
    int K = 1024;
    int N = 1024;       // going to be in a range

    printf("Allocating host variables..."); fflush(stdout);
    float *A_h = (float *)malloc(M * K * sizeof(float));
    float *B_h = (float *)malloc(K * N * sizeof(float));
    float *C_h = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) { A_h[i] = (rand()%100)/100.00; }
    for (int i = 0; i < K * N; i++) { B_h[i] = (rand()%100)/100.00; }
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    if (CE == CPU_CORES) {
        printf("Starting computing on CPU...");
        OpenMP_Gemm(A_h, B_h, C_h, M, K, N);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }else if (CE == GPU_REG_CORES || CE == GPU_TENSOR_CORES) {
        // allocating device variables
        float *A_d, *B_d, *C_d;
        printf("Allocating device variables..."); fflush(stdout);
        cudaMalloc((void **)&A_d, M * K * sizeof(float));
        cudaMalloc((void **)&B_d, K * N * sizeof(float));
        cudaMalloc((void **)&C_d, M * N * sizeof(float));
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy from host to device
        printf("Copying data from host to device..."); fflush(stdout);
        cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // TODO: SM filling-retreating to be added here

        printf("Launching GEMM...");
        if (CE == GPU_REG_CORES) {  // compute on GPU reg cores
            GRC_Gemm(A_h, B_h, C_h, M, K, N);
        }else {                     // compute on GPU tensor cores
            GTC_Gemm(A_h, B_h, C_h, M, K, N);
        }
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy from device to host
        printf("Copying data from device to host..."); fflush(stdout);
        cudaMemcpy(C_d, C_h, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        // free gpu memory
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }

    // free cpu memory
    free(A_h);
    free(B_h);
    free(C_h);
    
}