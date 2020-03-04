#ifndef __SUPPORT_CU__
#define __SUPPORT_CU__

#include "support.cuh"
#include "sm_alloc.cuh"

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

#ifdef CUBLAS_API_H_
// cuBLAS API errors
const char *_cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}
#endif

// SM filling-retreating tools
__global__ void resident_kernel(int *mapping, int *stop_flag, int *block_smids) {
    KERNEL_PROLOGUE();

    KERNEL_PERMANENT_RESIDENTS();

    KERNEL_EPILOGUE();
}

__global__ void dummy_kernel() {
    uint64_t spin_duration = 1000 * 1000 * 1000;
    uint64_t start_time = _get_global_time();
    // if (threadIdx.x == 0) {
    //     block_times[blockIdx.x * 2] = start_time;
    // }
    __syncthreads();
    while ((_get_global_time() - start_time) < spin_duration) {
        continue;
    }
    // if (threadIdx.x == 0) {
    //     block_times[blockIdx.x * 2 + 1] = __get_global_time();
    // }
    return;
}

// void *use_sm_residents(void *vargp) {
//     sigset_t set;
//     sigemptyset(&set);

//     SM_VARS_INIT();
//     SM_MAPPING_INIT(0, 0, 0, 0, 1, 1, 1, 1);
//     #ifdef SM_OCCUPATION
//     SM_KERNEL_LAUNCH();
//     #endif

//     pthread_exit((void *)0);
// }
extern int gflag;

void get_data_from_sensor(int src_file, std::string dest_file, int freq_div) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    // CPU_SET(0, &mask);
    CPU_SET(1, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        printf("sched_setaffinity for the power query thread failed.\n");
        exit(1);
    }

    if (src_file < 0) {
        fprintf(stderr, "Error opening file.");
        exit(1);
    }
    std::ofstream out;
    out.open(dest_file, std::ios::app);
    
    char buf[10];
    int n = 0;
    while (gflag == 0) {
        memset( buf, '\0', sizeof(char) * 10 );
        lseek(src_file, 0, 0);
        n = read(src_file, buf, 10);
        out << "GPU " << buf;
    }
}


#endif