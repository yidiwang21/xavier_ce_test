#ifndef __SUPPORT_CU__
#define __SUPPORT_CU__

#include "support.cuh"

int gflag = 0;

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

void get_data_from_sensor(int src_file, std::string dest_file, int freq_div) {
    if (src_file < 0) {
        fprintf(stderr, "Error opening file.");
        exit(1);
    }

    std::ofstream out;
    out.open(dest_file, std::ios::app);
    
    while (gflag == 0) {
        char buf[31];
        lseek(src_file, 0, 0);
        int n = read(src_file, buf, 32);
        out << "GPU " << buf;
    }
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

#endif