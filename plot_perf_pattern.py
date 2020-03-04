import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

def createList(r1, r2): 
    return np.arange(r1, r2+1, 1) 

def column(matrix, i):
    return [row[i] for row in matrix]

# N = int(sys.argv[1]) - 1

tensor_file_list = []
reg_file_list = []
size_list = createList(1, 2048)
# time_list.append([])

sm_list = []

# assume that all the file exist
for i in range(2, 9):
    tensor_file_list.append('log/log_gpu_tensor_SM_' + str(i) + '.txt')
    reg_file_list.append('log/log_gpu_reg_SM_' + str(i) + '.txt')
    sm_list.append(i)

# time_list = [[0] * 19] * len(tensor_file_list)
tensor_time_list = []
reg_time_list = []
temp_list = []

for i in range(len(tensor_file_list)):
    with open(tensor_file_list[i]) as f:
        strline = f.read()
    data = strline.split()

    temp_list = []
    cnt = 0
    for index, line in enumerate(data):
        if 'N' == line:
            # time_list[i][cnt] = float(data[index+3])
            # print(time_list[i][cnt])
            temp_list.append(float(data[index+3]))
            cnt += 1

    tensor_time_list.append(temp_list)
    f.close()


for i in range(len(reg_file_list)):
    with open(reg_file_list[i]) as f:
        strline = f.read()
    data = strline.split()

    temp_list = []
    cnt = 0
    for index, line in enumerate(data):
        if 'N' == line:
            # time_list[i][cnt] = float(data[index+3])
            # print(time_list[i][cnt])
            temp_list.append(float(data[index+3]))
            cnt += 1

    reg_time_list.append(temp_list)
    f.close()

# plot performance pattern
# plt.figure(1)
# plt.title('GEMM performance pattern')
# plt.plot(sm_list, column(tensor_time_list, N), label='Tensor core')
# plt.plot(sm_list, column(reg_time_list, N), label='Regular core')
# plt.xlabel('# of SM')
# plt.ylabel('Time(ms)')
# plt.legend()
# plt.show()

nrow = 2
ncol = 4
fig, axs = plt.subplots(nrow, ncol)
for i in range(nrow):
    for j in range(ncol):
        n = i * ncol + j    # 2**n
        N = 2**(n+4) - 1
        axs[i,j].plot(sm_list, column(tensor_time_list, N), label='Tensor core')
        axs[i,j].plot(sm_list, column(reg_time_list, N), label='Reg core')
        axs[i,j].set_xlabel('# of SMs')
        axs[i,j].set_ylabel('Time(ms)')
        axs[i,j].set_title('N = ' + str(N+1))
        axs[i,j].legend()

plt.show()
