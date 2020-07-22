import os, re
import numpy as np

def rename_files(dir):

    input_dir = dir + '/graphs'

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
    path = dir + "/state.txt"
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


#rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run00/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run01/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run02/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run03/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run04/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run05/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run06/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run07/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run08/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run09/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run10/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run11/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run12/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run13/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run14/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run15/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run16/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run17/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run18/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run19/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run20/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run21/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run22/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run23/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run24/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run25/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run26/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run27/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run28/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run29/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run30/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run31/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run32/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run33/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run34/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run35/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run36/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run37/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run38/toy-protmd/')
rename_files('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run39/toy-protmd/')
