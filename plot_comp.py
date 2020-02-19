import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

point_split = 20

## SM = 8
file1 = 'log_gpu_reg_SM_8.txt'
file2 = 'log_gpu_tensor_SM_8.txt'

file1_time_list = []
file2_time_list = []

size_list = []

with open(file1) as f:
    strline1 = f.read()
data1 = strline1.split()

for index, line in enumerate(data1):
    if 'N' == line:
        a = data1[index+2].replace(':', '')
        size_list.append(int(a))
        file1_time_list.append(float(data1[index+3]))

with open(file2) as f:
    strline2 = f.read()
data2 = strline2.split()

for index, line in enumerate(data2):
    if 'N' == line:
        a = data2[index+2].replace(':', '')
        file2_time_list.append(float(data2[index+3]))

fig, axs = plt.subplots(2)
fig.suptitle('GPU regular cores vs. tensor cores')
axs[0].plot(size_list, file2_time_list, label='Tensor cores')
axs[0].plot(size_list, file1_time_list, label='Regular cores')
axs[0].set_xlabel('Size N')
axs[0].set_ylabel('Time(ms)')
axs[0].set_xscale('log', basex=2)
axs[0].legend()

axs[1].plot(size_list[:point_split], file2_time_list[:point_split], label='Tensor cores')
axs[1].plot(size_list[:point_split], file1_time_list[:point_split], label='Regular cores')
axs[1].set_xticks(np.arange(0, point_split+1, step=1))
axs[1].set_xlabel('Size N')
axs[1].set_ylabel('Time(ms)')
axs[1].legend()



plt.show()