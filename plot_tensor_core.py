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

for i in range(1, 9):
    tensor_file_list.append('log/log_gpu_tensor_SM_' + str(i) + '.txt')
    if os.path.isfile(tensor_file_list[-1]) is False:
        tensor_file_list.pop()
    else:
        sm_list.append(i)

time_list = [[0] * 2048] * len(tensor_file_list)

print(tensor_file_list)
print(len(tensor_file_list))

for i in range(len(tensor_file_list)):
    with open(tensor_file_list[i]) as f:
        strline = f.read()
    data = strline.split()

    cnt = 0
    for index, line in enumerate(data):
        if 'N' == line:
            a = data[index+2].replace(':', '')
            time_list[i][cnt] = float(data[index+3])
            cnt += 1

    f.close()
    print(i, len(time_list[i]))

item_list = ['']
for i in range(len(tensor_file_list)):
    item_list.append('SM=' + str(i))

fig, axs = plt.subplots(2)
for i in range(len(tensor_file_list)):
    axs[0].bar(i+1, time_list[i][int(len(size_list)/2)-1], 0.5)
    axs[0].text(x = i+0.8, y = time_list[i][int(len(size_list)/2)-1] + 0.1, s = time_list[i][int(len(size_list)/2)-1])
axs[0].set_xticks(np.arange(len(tensor_file_list)), str(item_list))
axs[0].set_ylabel('Time(ms)')

for i in range(len(tensor_file_list)):
    axs[1].plot(size_list, time_list[i], label='Tensor cores SM = ' + str(sm_list[i]))
axs[1].set_xlabel('Size N')
axs[1].set_ylabel('Time(ms)')
axs[1].set_xscale('log', basex=2)
axs[1].legend()

plt.show()