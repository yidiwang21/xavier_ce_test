import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

def closest(lst, K): 
    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return lst[idx] 

filename = sys.argv[1]

curr_power_list = []
avgr_power_list = []
time_list = []

util_list = []

time_list.append(float(0))

with open(filename) as f:
    str = f.read()

data = str.split()
end_time = []
end_time_index = [] # corresponding index in time list

# extract gpu power consumption and execution time
for index, line in enumerate(data):
    if 'GPU' == line:
        curr_power_list.append(float(data[index+1]))
    elif 'time' in line:
        end_time.append(float(data[index+1]))

if not end_time:
    print("Log time error!")
    exit()
else:
    range_time = int(end_time[-1])
    increment = range_time / len(curr_power_list)

for x in range(1, len(curr_power_list)):
    time_list.append(float(x * increment))

for t in end_time:
    t = closest(time_list, t)
    end_time_index.append(time_list.index(t))

# time1: after copy from host to device
# time2: before copy from device to host
# end_time: exe end time
# stop_time: stop logging time (after cooling down)




fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(time_list, curr_power_list)
axs[0].set_title('Power')
axs[0].set_ylabel("Power Consumption (mWatt)")
axs[0].grid(color='k', linestyle=':', linewidth=1)


fig.suptitle(filename)
plt.show()