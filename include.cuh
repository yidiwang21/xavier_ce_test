#ifndef __INCLUDE_CUH__
#define __INCLUDE_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <string>
#include <mutex>
#include <signal.h>


enum computing_elem {CPU_CORES, GPU_REG_CORES, GPU_TENSOR_CORES};

#endif