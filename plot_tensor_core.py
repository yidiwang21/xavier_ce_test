import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

def createList(r1, r2): 
    return np.arange(r1, r2+1, 1) 

tensor_file_list = []
size_list = createList(1, 2048)
# time_list.append([])

sm_list = []

for i in range(2, 9):
    tensor_file_list.append('log/log_gpu_tensor_SM_' + str(i) + '.txt')
    if os.path.isfile(tensor_file_list[-1]) is False:
        tensor_file_list.pop()
    else:
        sm_list.append(i)

# time_list = [[0] * 19] * len(tensor_file_list)
time_list = []
temp_list = []

print(tensor_file_list)

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

    time_list.append(temp_list)
    f.close()

item_list = ['','']
for i in range(len(tensor_file_list)):
    item_list.append('SM=' + str(i+2))

plt.figure(1)
plt.title('Average computation time @ M = N = K = 2048')
for i in range(len(tensor_file_list)):
    plt.bar(i+2, time_list[i][int(len(size_list))-1], 0.5)
    plt.text(x = i+1.8, y = time_list[i][int(len(size_list))-1] + 0.1, s = time_list[i][int(len(size_list)/2)-1])
plt.xticks(np.arange(len(tensor_file_list)+2), item_list)
plt.ylabel('Time(ms)')

plt.figure(2)
plt.title('N ranging from 1 to 2048')
for i in range(len(tensor_file_list)):
    plt.plot(size_list, time_list[i], label='Tensor cores SM = ' + str(sm_list[i]))
plt.xlabel('Size N')
plt.ylabel('Time(ms)')
# plt.xscale('log', basex=2)
plt.legend()

plt.show()