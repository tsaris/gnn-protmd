import os, re
import numpy as np

input_dir = '/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/KRAS_r9_sup'
cnt = 0
freq = 10

filenames = os.listdir(input_dir)
num_list_on = []
num_list_off = []

for fil in filenames:

    if re.search("_ras_on.npz", fil) is not None:
        searchObj = re.finditer("_ras_on.npz", fil, re.M | re.I)
        for match in searchObj:
            start = match.span()[0]
            end = match.span()[1]
        num_list_on.append(int(fil[:start]))

    if re.search("_ras_off.npz", fil) is not None:
        searchObj = re.finditer("_ras_off.npz", fil, re.M | re.I)
        for match in searchObj:
            start = match.span()[0]
            end = match.span()[1]
        num_list_off.append(int(fil[:start]))



num_list_on = np.array(num_list_on)
num_list_off = np.array(num_list_off)
num_list_on = np.sort(num_list_on)
num_list_off = np.sort(num_list_off)

for i in range(0, num_list_on.shape[0]):
    if (cnt%freq) == 0:
        print('%s/%d_ras_on.npz'%(input_dir, num_list_on[i]))
        print('%s/%d_ras_off.npz'%(input_dir, num_list_off[i]))
    cnt+=1
