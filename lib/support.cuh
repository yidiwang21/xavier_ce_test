#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include "include.cuh"

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

const char *_cublasGetStatusString(cublasStatus_t status);

__global__ void resident_kernel(int *mapping, bool *stop_flag);

__device__ inline uint64_t GlobalTimer64(void);

__device__ inline long long seconds_to_gpu_cycles(int sec, int freq);

// void *use_sm_residents(void *vargp);

void get_data_from_sensor(int src_file, std::string dest_file, int freq_div);


#endif