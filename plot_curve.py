import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math

filename = sys.argv[1]

size_list = []
elapsed_time_list = []

with open(filename) as f:
    strline = f.read()
data = strline.split()

for index, line in enumerate(data):
    if 'N' == line:
        a = data[index+2].replace(':', '')
        size_list.append(int(a))
        elapsed_time_list.append(float(data[index+3]))

plt.figure();
plt.plot(size_list, elapsed_time_list)
plt.show()