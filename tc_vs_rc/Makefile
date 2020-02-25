NVCC        	= nvcc #~-std=c++11
NVCC_FLAGS  	= -O3 -I/usr/local/cuda/include
LD_FLAGS    	= -lcudart -L/usr/local/cuda/lib64 -lcublas
EXE	        = main
# OBJ	        = main.o cpu_cores.o gpu_reg_cores.o gpu_tensor_cores.o support.o
OBJ	        = main.o support.o
ARCH		= -arch=sm_52
CJSON		= -l cjson
LIB			= lib/

default: $(EXE)

support.o: $(LIB)support.cu $(LIB)support.cuh $(LIB)sm_alloc.cuh
	$(NVCC) -c -o $@ $(LIB)support.cu $(NVCC_FLAGS)

# cpu_cores.o: cpu_cores.cu
# 	$(NVCC) -c -o $@ cpu_cores.cu $(NVCC_FLAGS)

# gpu_reg_cores.o: gpu_reg_cores.cu
# 	$(NVCC) -c -o $@ gpu_reg_cores.cu $(NVCC_FLAGS)

# gpu_tensor_cores.o: gpu_tensor_cores.cu
# 	$(NVCC) -c -o $@ gpu_tensor_cores.cu $(NVCC_FLAGS)

main.o: main.cu $(LIB)support.cuh $(LIB)sm_alloc.cuh
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(CJSON) $(ARCH) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)