#ifndef __SM_ALLOC__
#define __SM_ALLOC__

#include "include.cuh"

#define SM_OCCUPATION

#define MAX_SM  8
#define SM_OCCUPIED_BLOCKSIZE 1    /* blocksize of the blocks that are used to occupy the certain  number of SMs */
#define SM_BLOCKS_PER   32
#define SM_OCCUPIED_GRIDSIZE MAX_SM * SM_BLOCKS_PER

#define NUMARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))

extern __global__ void resident_kernel(int *mapping, int *stop_flag);

#define _get_smid() ({  \
    uint ret;   \
    asm("mov.u32 %0, %smid;" : "=r"(ret) ); \
    ret; })

#define _get_global_time() ({   \
    uint64_t reading; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading)); \
    reading;    })



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
    int *stop_flag_h = new int;   \
    *stop_flag_h = false;    \
    int *stop_flag_d;  \
    cudaMalloc((void**)&stop_flag_d, sizeof(int));

#define SM_CREATE_STREAM()  \
    printf("    * Creating stream for resident kernels...\n"); \
    cudaStream_t occupied_stream;   \
    cudaStreamCreateWithFlags(&occupied_stream, cudaStreamNonBlocking);

#define SM_COPY_MAPPING()   \
    printf("    * Copying SM mapping to device...\n"); \
    cudaMemcpy(mapping_d, mapping_h, mapping_size, cudaMemcpyHostToDevice); \
    cudaMemcpy(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice);

#define SM_STOP_KERNEL_RESIDENTS() \
    printf("    * Stopping kernel residents...\n"); \
    *stop_flag_h = true; \
    cudaMemcpy(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice);


#define SM_KERNEL_LAUNCH() \
    SM_CREATE_STREAM(); \
    SM_COPY_MAPPING(); \
    printf("    * Launching resident_kernel...\n"); \
    resident_kernel <<< SM_OCCUPIED_GRIDSIZE, SM_OCCUPIED_BLOCKSIZE, 0, occupied_stream >>> (mapping_d, stop_flag_d); \

/* FIXME: check any other code needed here */
#define KERNEL_PROLOGUE() \
    int smid = _get_smid(); \
    if (mapping[smid + 1] == 0) return; /* if smid is not desired  */

/* permanent residents in GPU, spin "forever" */
#define KERNEL_PERMANENT_RESIDENTS()  \
    uint64_t curr_time; \
    while (stop_flag == false) curr_time = _get_global_time();

#define KERNEL_EPILOGUE()


#endif /* __SM_ALLOC__ */