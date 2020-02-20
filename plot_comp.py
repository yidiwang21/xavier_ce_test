import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

def compute_avg(lst):
    return sum(lst) / len(lst)

point_split = 32

## SM = 8
file0 = 'log_cpu_2.txt'
file1 = 'log_gpu_reg_SM_8.txt'
file2 = 'log_gpu_tensor_SM_8.txt'
file3 = 'log_gpu_reg_SM_4.txt'
file4 = 'log_gpu_tensor_SM_4.txt'

file0_time_list = []
file1_time_list = []
file2_time_list = []
file3_time_list = []
file4_time_list = []

size_list = []

with open(file0) as f:
    strline0 = f.read()
data0 = strline0.split()

for index, line in enumerate(data0):
    if 'N' == line:
        a = data0[index+2].replace(':', '')
        size_list.append(int(a))
        file0_time_list.append(float(data0[index+3]))


with open(file1) as f:
    strline1 = f.read()
data1 = strline1.split()

for index, line in enumerate(data1):
    if 'N' == line:
        a = data1[index+2].replace(':', '')
        file1_time_list.append(float(data1[index+3]))

with open(file2) as f:
    strline2 = f.read()
data2 = strline2.split()

for index, line in enumerate(data2):
    if 'N' == line:
        a = data2[index+2].replace(':', '')
        file2_time_list.append(float(data2[index+3]))

with open(file3) as f:
    strline3 = f.read()
data3 = strline3.split()

for index, line in enumerate(data3):
    if 'N' == line:
        a = data3[index+2].replace(':', '')
        file3_time_list.append(float(data3[index+3]))

with open(file4) as f:
    strline4 = f.read()
data4 = strline4.split()

for index, line in enumerate(data4):
    if 'N' == line:
        a = data4[index+2].replace(':', '')
        file4_time_list.append(float(data4[index+3]))

# figure 1 shows the reg core and tensor core comparison
fig, axs = plt.subplots(2, 2)
fig.suptitle('GPU regular cores vs. tensor cores')
axs[0,0].plot(size_list, file2_time_list, label='Tensor cores')
axs[0,0].plot(size_list, file1_time_list, label='Regular cores')
axs[0,0].set_xlabel('Size N')
axs[0,0].set_ylabel('Time(ms)')
axs[0,0].set_xscale('log', basex=2)
axs[0,0].set_title('SM = 8, N ranging from 1 to 2^11')
axs[0,0].legend()

axs[0,1].plot(size_list[:point_split], file2_time_list[:point_split], label='Tensor cores')
axs[0,1].plot(size_list[:point_split], file1_time_list[:point_split], label='Regular cores')
axs[0,1].set_xticks(np.arange(0, point_split+1, step=1))
axs[0,1].set_xlabel('Size N')
axs[0,1].set_ylabel('Time(ms)')
axs[0,1].set_title('SM = 8, N ranging from 1 to 2^5')
axs[0,1].legend()

axs[1,0].plot(size_list, file3_time_list, label='Regular cores SM = 4')
axs[1,0].plot(size_list, file1_time_list, label='Regular cores SM = 8')
axs[1,0].set_xlabel('Size N')
axs[1,0].set_ylabel('Time(ms)')
axs[1,0].set_xscale('log', basex=2)
axs[1,0].legend()

axs[1,1].plot(size_list, file4_time_list, label='Tensor cores SM = 4')
axs[1,1].plot(size_list, file2_time_list, label='Tensor cores SM = 8')
axs[1,1].set_xlabel('Size N')
axs[1,1].set_ylabel('Time(ms)')
axs[1,1].set_xscale('log', basex=2)
axs[1,1].legend()

# figure 2 
# compare the average? max? computation time
# average computation time of the different matrix sizes

bary = [file0_time_list[int(len(size_list)/2)-1], file1_time_list[int(len(size_list)/2)-1], file3_time_list[int(len(size_list)/2)-1], file2_time_list[int(len(size_list)/2)-1], file4_time_list[int(len(size_list)/2)-1]]

plt.figure()
plt.title('Average computation time @ M = N = K = 1024')
# plt.bar(0, file0_time_list[int(len(size_list)/2)-1])
plt.bar(1, file1_time_list[int(len(size_list)/2)-1])
plt.bar(2, file3_time_list[int(len(size_list)/2)-1])
plt.bar(3, file2_time_list[int(len(size_list)/2)-1])
plt.bar(4, file4_time_list[int(len(size_list)/2)-1])
for i in range(1,5):
    plt.text(x = i-0.3, y = bary[i] + 0.1, s = bary[i])
plt.xticks(np.arange(5), ('', 'gpu reg sm=8', 'gpu reg sm=4', 'gpu tensor sm=8', 'gpu tensor sm=4'))


plt.show()