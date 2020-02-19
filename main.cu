#include "include.cuh"
#include "support.cuh"
#include "sm_alloc.cuh"
#include "ce_func.cuh"

#define LOOP_NUM    2048

// #define _GNU_SOURCE

// gpu regular core implementation
// extern void GRC_Gemm (cublasHandle_t handle, const float *A, const float *B, float *C, const int M, const int K, const int N);
// // gpu tensor core implementation
// extern void GTC_Gemm (const float *A, const float *B, float *C, const int M, const int K, const int N);
// // cpu open mp implementation
// extern void OpenMP_Gemm(const float *A, const float *B, float *C, const int M, const int K, const int N);

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
    std::string outfile;
    int opt;
    enum computing_elem CE = GPU_REG_CORES;   // default computing element is GPU regular cores
    int sm = MAX_SM;
    int n_cpu_cores = 0;
    while ((opt = getopt (argc, argv, "e:s:c: ")) != EOF) {
        switch (opt) {
            case 'e':       // computing element
                if (atoi(optarg) == 0) CE = CPU_CORES; 
                else if (atoi(optarg) == 1) CE = GPU_REG_CORES;
                else if (atoi(optarg) == 2) CE = GPU_TENSOR_CORES;
                break;
            case 's':       // number of SM
                sm = atoi(optarg);
                break;
            case 'c': // number of cpu cores
                n_cpu_cores = atoi(optarg);
                if (get_nprocs() - 2 < n_cpu_cores) {
                    printf("Error: only %d cores are available.\n", get_nprocs());
                    exit(1);
                }
                break;
            case '?':
                fprintf(stderr, "# Usage: \n -e <computing_elem> -s <number of SMs> -c <number of CPU cores>");
            default:
                printf("\n"); abort();
        }
    }

    printf("##################################################\n");
    switch (CE) {
        case CPU_CORES:
            printf("Computing element: CPU cores\n"); fflush(stdout);
            outfile = "log_cpu_" + std::to_string(n_cpu_cores) + ".txt";
            break;
        case GPU_REG_CORES:
            printf("Computing element: GPU regular cores\n"); fflush(stdout);
            printf("Online SM: %d\n", sm);
            outfile = "log_gpu_reg_SM_" + std::to_string(sm) + ".txt";
            break;
        case GPU_TENSOR_CORES:
            printf("Computing element: GPU tensor cores\n"); fflush(stdout);
            printf("Online SM: %d\n", sm);
            outfile = "log_gpu_tensor_SM_" + std::to_string(sm) + ".txt";
            break;
    }
    const char *cstr = outfile.c_str();
    std::ifstream ifile(cstr);
    if (ifile) remove(cstr);
    std::ofstream ofile(outfile, std::ofstream::out);
    printf("Log file saved to %s\n", outfile.c_str()); fflush(stdout);
    printf("##################################################\n");

    srand(time(NULL));

    // create CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    // create stream for the above CUBLAS handle
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    cpu_set_t gemm_cpu_mask;
    CPU_ZERO(&gemm_cpu_mask);
    for (int i = 2; i < n_cpu_cores; i++)
        CPU_SET(i, &gemm_cpu_mask);

    // if (CE == GPU_REG_CORES || CE == GPU_TENSOR_CORES) {
    //     if (sm == 4) {
    //         SM_MAPPING_INIT(0, 0, 0, 0, 1, 1, 1, 1);
    //         #ifdef SM_OCCUPATION
    //         SM_KERNEL_LAUNCH();
    //         #endif
    //     }else if (sm == 8) {
    //         // SM_MAPPING_INIT(0, \
    //             1, 1, 1, 1, \
    //             1, 1, 1, 1);
    //         #undef SM_OCCUPATION
    //     }
    // }

    Timer timer;
    startTime(&timer);  // us
    float time_stamp_1, time_stamp_2;
    float total_time;

    for (int n = 1; n <= LOOP_NUM; n++) {
        printf("Looping number # %d\n", n);

        int M = 1024;
        int K = 1024;
        int N = n;       // dflt 1024
        const float alpha = 1.0;
        const float beta = 0.0;

        printf("Allocating host variables..."); fflush(stdout);
        float *A_h = (float *)malloc(M * K * sizeof(float));
        float *B_h = (float *)malloc(K * N * sizeof(float));
        float *C_h = (float *)malloc(M * N * sizeof(float));

        for (int i = 0; i < M * K; i++) { A_h[i] = (rand()%100)/100.00; }
        for (int i = 0; i < K * N; i++) { B_h[i] = (rand()%100)/100.00; }
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        if (CE == CPU_CORES) {
            struct cpu_thread_args in_args = {A_h, B_h, C_h, M, K, N};
            printf("Starting computing on CPU...");
            time_stamp_1 = elapsedTime(timer) * 1000;   // convert to ms

            pthread_t mythread;
            pthread_create(&mythread, NULL, *OpenMP_Gemm, (void *)&in_args); 
            pthread_setaffinity_np(mythread, sizeof(cpu_set_t), &gemm_cpu_mask);
            pthread_join(mythread, NULL);
            
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));
            time_stamp_2 = elapsedTime(timer) * 1000;
        }else if (CE == GPU_REG_CORES || CE == GPU_TENSOR_CORES) {
            // SM filling-retreating added here
            // sync stream instead of device after implementing this
            // if a smid is desired for GEMM, then it's undesired for this kernel
            // pthread_t sm_thread;
            // int ret1;
            // if (sm == 4) {
            //     pthread_create(&sm_thread, NULL, *use_sm_residents, NULL);
            // }else if (sm == 8) {
            //     #undef SM_OCCUPATION
            // }
            SM_MAPPING_INIT(0, 0, 0, 0, 1, 1, 1, 1);
            if (sm == 4) {
                #ifdef SM_OCCUPATION
                SM_KERNEL_LAUNCH();
                #endif            
            }else if (sm == 8) {
                #undef SM_OCCUPATION
            }
            

            // #ifdef SM_OCCUPATION

            //     SM_KERNEL_LAUNCH();
            // #endif  /* SM_OCCUPATION */
            usleep(100);


            // allocating device variables
            float *A_d, *B_d, *C_d;
            printf("Allocating device variables..."); fflush(stdout);
            cudaMalloc((void **)&A_d, M * K * sizeof(float));
            cudaMalloc((void **)&B_d, K * N * sizeof(float));
            cudaMalloc((void **)&C_d, M * N * sizeof(float));
            // cudaDeviceSynchronize();
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));

            // Copy from host to device
            printf("Copying data from host to device..."); fflush(stdout);
            cudaMemcpyAsync(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
            // cudaDeviceSynchronize();
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));

            printf("Launching GEMM...");
            time_stamp_1 = elapsedTime(timer) * 1000;   // convert to ms
            if (CE == GPU_REG_CORES) {  // compute on GPU reg cores
                // GRC_Gemm(handle, A_d, B_d, C_h, M, K, N);
                GRC_GEMM();
            }else {                     // compute on GPU tensor cores
                // GTC_Gemm(A_d, B_d, C_h, M, K, N);
                GTC_GEMM();
            }
            cudaStreamSynchronize(stream);  // 0?
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));
            time_stamp_2 = elapsedTime(timer) * 1000;   // convert to ms

            // Copy from device to host
            printf("Copying data from device to host..."); fflush(stdout);
            cudaMemcpyAsync(C_d, C_h, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));

            //
            // int status = pthread_kill(sm_thread, SIGUSR1); 
            // if (status < 0) {
            //     perror("pthread_kill failed\n");
            // }
            // status = pthread_join(sm_thread, NULL);
            // if (status < 0) {
            //     perror("pthread_join failed\n");
            // }

            SM_STOP_KERNEL_RESIDENTS();
                                       
            // pthread_cancel(sm_thread);

            // free gpu memory
            cudaFree(A_d);
            cudaFree(B_d);
            cudaFree(C_d);
        }

        // free cpu memory
        free(A_h);
        free(B_h);
        free(C_h);

        // write time stamps into log file
        ofile << "N = " << N << ": " << std::to_string(time_stamp_2 - time_stamp_1) << std::endl;
        printf("---------------------------------------------------------\n");
    }
    // do not destroy the handle if more than one mmul need to be done
    cublasDestroy(handle);    

    // cudaDeviceReset();
    stopTime(&timer); printf("Total time: %f s\n", elapsedTime(timer));
    
    cudaDeviceSynchronize();
    return 0;
}