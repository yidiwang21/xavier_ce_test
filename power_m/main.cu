#include "include.cuh"
#include "ce_func.cuh"
#include "support.cuh"
#include "sm_alloc.cuh"
#include "common_snippet.cuh"
#include "cuda_profiler_api.h"
#include <thread>

int gflag = 0;
std::mutex mtx;

int main(int argc, char *argv[]) {
    int opt;
    enum exec_mode EM = INDIVIDUAL;
    int loop_num = 1;
    int sm = 8;
    enum computing_elem CE = GPU_REG_CORES;
    while ((opt = getopt (argc, argv, "e:l:s: ")) != EOF) {
        switch (opt) {
            case 'e':
                if (atoi(optarg) == 0) CE = CPU_CORES; 
                else if (atoi(optarg) == 1) CE = GPU_REG_CORES;
                else if (atoi(optarg) == 2) CE = GPU_TENSOR_CORES;
                else if (atoi(optarg) == -1) CE = TEST_OUTPUT;
                break;
            case 'l':
                loop_num = atoi(optarg);
                break;
            case 's':
                sm = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "# Usage: \n -e <computing_elem> -l <loop number>");
            default:
                printf("\n"); abort();
        }
    }

    std::string outfile;
    switch (CE) {
        case GPU_REG_CORES:
            printf("Computing element: GPU regular cores\n"); fflush(stdout);
            outfile = "power_gpu_reg_SM_" + std::to_string(sm) + ".txt";
            break;
        case GPU_TENSOR_CORES:
            printf("Computing element: GPU tensor cores\n"); fflush(stdout);
            outfile = "power_gpu_tensor_SM_" + std::to_string(sm) + ".txt";
            break;
        case TEST_OUTPUT:
            printf("Testing resident kernels\n"); fflush(stdout);
            outfile = "power_residents_SM_" + std::to_string(sm) + ".txt";
            break;
    }

    const char *cstr = outfile.c_str();
    std::ifstream ifile(cstr);
    if (ifile) remove(cstr);

    cudaError_t cuda_ret;

    float time1, time2;
    Timer timer;
    startTime(&timer);

    // NOTE: GEMM configuration starts here
    // GEMM init
    // MM_INIT_PARAM();
    // MM_CREATE_CUDA_STREAM();
    // MM_CREATE_HANDLE();
    // MM_ALLOC_HOST_VARS();
    // MM_ALLOC_DEVICE_VARS();
    // MM_COPY_MEMORY_FROM_HOST_TO_DEVICE();

    // cudaDeviceSynchronize();

    int fd = open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input", O_RDONLY | O_NONBLOCK);
    printf("Creating thread for power reading...");
    std::thread power_thread(get_data_from_sensor, fd, outfile, 0);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("##################################################\n");


    stopTime(&timer);
    time1 = elapsedTime(timer);
    
    SM_VARS_INIT();
    // SM_MAPPING_INIT(1, 1, 1, 1, 1, 1, 1, 1);
    if (sm == 0) {
        SM_MAPPING_INIT(1, 1, 1, 1, 1, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 1) {
        SM_MAPPING_INIT(0, 1, 1, 1, 1, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 2) {
        SM_MAPPING_INIT(0, 0, 1, 1, 1, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 3) {
        SM_MAPPING_INIT(0, 0, 0, 1, 1, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 4) {
        SM_MAPPING_INIT(0, 0, 0, 0, 1, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    }else if (sm == 5) {
        SM_MAPPING_INIT(0, 0, 0, 0, 0, 1, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 6) {
        SM_MAPPING_INIT(0, 0, 0, 0, 0, 0, 1, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 7) {
        SM_MAPPING_INIT(0, 0, 0, 0, 0, 0, 0, 1);
        SM_KERNEL_LAUNCH();
    } else if (sm == 8) {
        SM_MAPPING_INIT(0, 0, 0, 0, 0, 0, 0, 0);
        SM_KERNEL_LAUNCH();
    }


    // NOTE: GEMM launched here
    // printf("Run GEMM for %d times...", loop_num);
    // if (CE == GPU_REG_CORES) {
    //     for (int i = 0; i < loop_num; i++) { 
    //         GRC_GEMM();
    //         cudaDeviceSynchronize();
    //     }
    // }else if (CE == GPU_TENSOR_CORES) {
    //     for (int i = 0; i < loop_num; i++) {
    //         GTC_GEMM();
    //         cudaDeviceSynchronize();
    //     }
    // }

    // TODO: measure permanent kernels
    // sleep(10);
    // SM_STOP_KERNEL_RESIDENTS();
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    time2 = elapsedTime(timer);


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

    mtx.lock();
    gflag = 1;
    mtx.unlock();
    printf("Joining power thread...\n");
    power_thread.join();

    std::ofstream out;
    out.open(outfile, std::ios::app);
    out << "time1: " << 0 << std::endl;
    out << "time2: " << time2 << std::endl;

    // NOTE: GEMM ends here
    // MM_COPY_MEMORY_FROM_DEVICE_TO_HOST();
    // cudaDeviceSynchronize();

    // MM_FREE_DEVICE_VARS();
    // MM_FREE_HOST_VARS();

    // MM_DESTROY_HANDLE();

    cudaDeviceSynchronize();

    
    cudaProfilerStop();
    return 0;

}