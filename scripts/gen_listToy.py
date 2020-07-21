import os, re
import numpy as np

input_dir = '/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/toy-protmd/graphs'


filenames = os.listdir(input_dir)
num_list_on = []
num_list_off = []

for fil in filenames:

    if re.search("_mdToy.npz", fil) is not None:
        searchObj = re.finditer("_mdToy.npz", fil, re.M | re.I)
        for match in searchObj:
            start = match.span()[0]
            end = match.span()[1]
        num_list_on.append(int(fil[:start]))


num_list_on = np.array(num_list_on)
num_list_on = np.sort(num_list_on)


# First parse the name files
path = "/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/toy-protmd/tmp.txt"
lines = []
with open(path, 'r') as f:
    line = f.readline()
    line = line.split()
    lines.append(line[1])

    while line:
        line = f.readline()
        line = line.split()
        if len(line) == 0: break
        lines.append(line[1])


if num_list_on.shape[0] != len(lines):
    print("Something is wrong"); exit(-1)


for i in range(0, num_list_on.shape[0]):
    filename = '%s/%d_mdToy.npz'%(input_dir, num_list_on[i])
    if (lines[i] == "Turn"):
        new_filename = '%s/%d_mdToy_on.npz'%(input_dir, num_list_on[i])
    else: new_filename = '%s/%d_mdToy_off.npz'%(input_dir, num_list_on[i])
    os.rename(filename, new_filename)

