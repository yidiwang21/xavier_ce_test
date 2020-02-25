#ifndef __SM_ALLOC__
#define __SM_ALLOC__

#include "include.cuh"

#define SM_OCCUPATION

#define SM_OCCUPIED_BLOCKSIZE 1    /* blocksize of the blocks that are used to occupy the certain  number of SMs */
#define SM_BLOCKS_PER   32
#define SM_OCCUPIED_GRIDSIZE MAX_SM * SM_BLOCKS_PER

#define NUMARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))

extern __global__ void resident_kernel(int *mapping, int *stop_flag, int *block_smids);

#define _get_smid() ({  \
    uint ret;   \
    asm("mov.u32 %0, %smid;" : "=r"(ret) ); \
    ret; })

#define _get_global_time() ({   \
    uint64_t reading; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading)); \
    reading;    })

#define SM_VARS_INIT()   \
    printf("    * initializing variables...\n");    \
    int device = -1;    \
    int *stop_flag_h = new int;   \
    *stop_flag_h = 0;    \
    int *stop_flag_d;  \
    cudaMalloc((void**)&stop_flag_d, sizeof(int));  \
    int *block_smids_h = new int[SM_OCCUPIED_GRIDSIZE]; \
    memset(block_smids_h, -100, SM_OCCUPIED_GRIDSIZE * sizeof(int)); \
    int *block_smids_d; \
    cudaMalloc((void**)&block_smids_d, sizeof(int) * SM_OCCUPIED_GRIDSIZE); \
    cudaStream_t occupied_stream;   \
    cudaStream_t backup_stream;

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

#define SM_CREATE_STREAM()  \
    printf("    * Creating stream for resident kernels...\n"); \
    cudaStreamCreateWithFlags(&occupied_stream, cudaStreamNonBlocking); \
    cudaStreamCreateWithFlags(&backup_stream, cudaStreamNonBlocking);

#define SM_COPY_MAPPING()   \
    printf("    * Copying SM mapping to device...\n"); \
    cudaMemcpyAsync(mapping_d, mapping_h, mapping_size, cudaMemcpyHostToDevice, occupied_stream); \
    cudaMemcpyAsync(block_smids_d, block_smids_h, sizeof(int) * SM_OCCUPIED_GRIDSIZE, cudaMemcpyHostToDevice, occupied_stream); \
    cudaGetDevice(&device); \
    /* cudaMemPrefetchAsync(stop_flag, sizeof(int), device, backup_stream); */\
    cudaMemcpyAsync(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice, backup_stream);

#define SM_KERNEL_LAUNCH() \
    SM_CREATE_STREAM(); \
    SM_COPY_MAPPING(); \
    printf("    * Launching resident_kernel...\n"); \
    resident_kernel <<< SM_OCCUPIED_GRIDSIZE, SM_OCCUPIED_BLOCKSIZE, 0, occupied_stream >>> (mapping_d, stop_flag_d, block_smids_d);

#define SM_STOP_KERNEL_RESIDENTS() \
    printf("    * Stopping kernel residents...\n"); \
    /* *stop_flag = 1; \
    cudaMemPrefetchAsync(stop_flag, sizeof(int), device, backup_stream); \ */\
    *stop_flag_h = 1; \
    cudaMemcpyAsync(stop_flag_d, stop_flag_h, sizeof(int), cudaMemcpyHostToDevice, backup_stream);  \
    printf("    * Copying smids from device to host...\n"); \
    cudaMemcpyAsync(block_smids_h, block_smids_d, sizeof(int) * SM_OCCUPIED_GRIDSIZE, cudaMemcpyDeviceToHost, occupied_stream); \
    /* for (int i = 0; i < SM_OCCUPIED_GRIDSIZE; i++) { \
        printf("block smid: %d\n", block_smids_h[i]); \
    }   \ */
    

#define KERNEL_PROLOGUE() \
    uint64_t start_time = _get_global_time();   \
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
    while (_get_global_time() - start_time < 1000 * 1000 * 1000) {    \
        if (*stop_flag == 1)    \
            break; \
        continue; \
    }   \
    return; \

#define KERNEL_EPILOGUE()


#endif /* __SM_ALLOC__ */