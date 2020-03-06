#ifndef __SM_ALLOC__
#define __SM_ALLOC__

#include "include.cuh"

#define SM_OCCUPATION

// TODO: used to be 1
#define SM_OCCUPIED_BLOCKSIZE 32    /* blocksize of the blocks that are used to occupy the certain  number of SMs */
#define SM_BLOCKS_PER   32
#define SM_OCCUPIED_GRIDSIZE MAX_SM * SM_BLOCKS_PER

#define NUMARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))

extern __global__ void resident_kernel(int *mapping, int *stop_flag, int *block_smids);

extern __device__ inline uint64_t GlobalTimer64(void);
extern __device__ inline long long seconds_to_gpu_cycles(int sec, int freq);

#define _get_smid() ({  \
    uint ret;   \
    asm("mov.u32 %0, %smid;" : "=r"(ret) ); \
    ret; })

#define SM_VARS_INIT()   \
    printf("    * initializing variables...\n");    \
    int device = -1;    \
    int peak_clk = 1;   \
    int *stop_flag_h = new int;   \
    *stop_flag_h = 0;    \
    int *stop_flag_d;  \
    cudaMalloc((void**)&stop_flag_d, sizeof(int));  \
    int *block_smids_h = new int[SM_OCCUPIED_GRIDSIZE]; \
    memset(block_smids_h, -100, SM_OCCUPIED_GRIDSIZE * sizeof(int)); \
    int *block_smids_d; \
    cudaMalloc((void**)&block_smids_d, sizeof(int) * SM_OCCUPIED_GRIDSIZE); \
    cudaStream_t occupied_stream;   \
    cudaStream_t backup_stream; \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define SM_MAPPING_INIT(...)    \
    printf("    * Initialize SM mapping...\n");   \
    int *mapping_d; \
    int mapping_h[MAX_SM+1] = {MAX_SM, __VA_ARGS__};    /*  1: enabled, 0: disabled */ \
    if (NUMARGS(__VA_ARGS__) != MAX_SM) {   \
        printf("STGM_INIT: wrong arguments, ("#__VA_ARGS__")\n");   \
        exit(-1);   \
    }   \
    int active_sm = 0;  \
    for (int i = 1; i <= MAX_SM; i++)   \
        if (mapping_h[i] > 0) active_sm++;  \
    mapping_h[0] = active_sm;   \
    int mapping_size = (MAX_SM + 1) * sizeof(int);  \
    cudaMalloc((void**)&mapping_d, mapping_size);   \
    /* Alloc unified memory  */ \
    /* int *stop_flag; \
    cudaMallocManaged((void**)&stop_flag, sizeof(int)); \
    *stop_flag = 0; \ */\
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define SM_CREATE_STREAM()  \
    printf("    * Creating stream for resident kernels...\n"); \
    cudaStreamCreateWithFlags(&occupied_stream, cudaStreamNonBlocking); \
    cudaStreamCreateWithFlags(&backup_stream, cudaStreamNonBlocking);   \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define SM_COPY_MAPPING()   \
    printf("    * Copying SM mapping to device...\n"); \
    cudaMemcpyAsync(mapping_d, mapping_h, mapping_size, cudaMemcpyHostToDevice, occupied_stream); \
    cudaMemcpyAsync(block_smids_d, block_smids_h, sizeof(int) * SM_OCCUPIED_GRIDSIZE, cudaMemcpyHostToDevice, occupied_stream); \
    cudaGetDevice(&device); \
    /* cudaMemPrefetchAsync(stop_flag, sizeof(int), device, backup_stream); */\
    cudaMemcpyAsync(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice, backup_stream);  \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define SM_KERNEL_LAUNCH() \
    SM_CREATE_STREAM(); \
    SM_COPY_MAPPING(); \
    printf("    * Launching resident_kernel...\n"); \
    resident_kernel <<< SM_OCCUPIED_GRIDSIZE, SM_OCCUPIED_BLOCKSIZE, 0, occupied_stream >>> (mapping_d, stop_flag_d, block_smids_d);    \
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#define SM_STOP_KERNEL_RESIDENTS() \
    printf("    * Stopping kernel residents...\n"); \
    /* *stop_flag = 1; \
    cudaMemPrefetchAsync(stop_flag, sizeof(int), device, backup_stream); \ */\
    *stop_flag_h = 1; \
    cudaMemcpyAsync(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice, backup_stream);  \
    printf("    * Copying smids from device to host..."); \
    cudaMemcpyAsync(block_smids_h, block_smids_d, sizeof(int) * SM_OCCUPIED_GRIDSIZE, cudaMemcpyDeviceToHost, occupied_stream); \
    /* for (int i = 0; i < SM_OCCUPIED_GRIDSIZE; i++) { \
        printf("block smid: %d\n", block_smids_h[i]); \
    }   \ */\
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


// to add some workload to the permanent kernels to check SM power consumption
#define EXTRA_BEFORE_LOOP() \
    float r1, r2, r3;   \
    unsigned int seed = threadIdx.x;    \
    curandState_t curand_state; \
    curand_init(seed, 0, 0, &curand_state);    \
    r1 = curand_uniform(&curand_state); \
    r2 = curand_uniform(&curand_state); \
    r3 = curand_uniform(&curand_state);

#define EXTRA_WORKLOADS_1()  \
    int temp = 0;   \
    temp = threadIdx.x + blockIdx.x * blockDim.x;   \
    temp %= _get_smid();    \

#define EXTRA_WORKLOADS_2() \
    r3 = r1 + r2 ;  \
    r2 = r3 + r1 ;  \
    r1 = r2 + r3 ;  \
    r3 = r1 + r2 ;  \
    r2 = r3 + r1 ;  \
    r1 = r2 + r3 ;


#define KERNEL_PROLOGUE() \
    uint64_t start_time = GlobalTimer64();   \
    long long start_clock = clock64();  \
    int smid = _get_smid(); \
    __syncthreads(); \
    if (mapping[smid + 1] == 0) {   \
        return; \
    }

/* permanent residents in GPU, spin "forever" */
#define KERNEL_PERMANENT_RESIDENTS()  \
    if (threadIdx.x == 0) { \
        block_smids[blockIdx.x] = _get_smid();  \
    }   \
    __syncthreads();    \
    EXTRA_BEFORE_LOOP();    \
    while (clock64() < start_clock + seconds_to_gpu_cycles(10, 675750000)) {    \
        /* if (*stop_flag == 1) return; \ */\
        EXTRA_WORKLOADS_2(); \
        continue; \
    }   \
    return; \

#define KERNEL_EPILOGUE()


#endif /* __SM_ALLOC__ */


// _get_global_time() - start_time < 1000 * 1000 * 1000 * 100