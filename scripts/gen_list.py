import os, re
import numpy as np

def makelist(dir):

    input_dir = dir + '/graphs'
    cnt = 0
    freq = 1

    filenames = os.listdir(input_dir)
    num_list_on = []
    num_list_off = []

    for fil in filenames:

        if re.search("_mdToy_on.npz", fil) is not None:
            searchObj = re.finditer("_mdToy_on.npz", fil, re.M | re.I)
            for match in searchObj:
                start = match.span()[0]
                end = match.span()[1]
            num_list_on.append(int(fil[:start]))

        if re.search("_mdToy_off.npz", fil) is not None:
            searchObj = re.finditer("_mdToy_off.npz", fil, re.M | re.I)
            for match in searchObj:
                start = match.span()[0]
                end = match.span()[1]
            num_list_off.append(int(fil[:start]))


    num_list_on = np.array(num_list_on)
    num_list_off = np.array(num_list_off)
    num_list_on = np.sort(num_list_on)
    num_list_off = np.sort(num_list_off)

    mini = min(num_list_on.shape[0], num_list_off.shape[0])
        
    for i in range(0, mini):
        if (cnt%freq) == 0:
            filelist = input_dir + '/filelist.csv'
            with open(filelist, 'a+') as f:
                print('%s/%d_mdToy_on.npz'%(input_dir, num_list_on[i]), file=f)
                print('%s/%d_mdToy_off.npz'%(input_dir, num_list_off[i]), file=f)
        cnt+=1


makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run00/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run01/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run02/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run03/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run04/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run05/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run06/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run07/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run08/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run09/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run10/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run11/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run12/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run13/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run14/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run15/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run16/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run17/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run18/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run19/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run20/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run21/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run22/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run23/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run24/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run25/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run26/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run27/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run28/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run29/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run30/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run31/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run32/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run33/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run34/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run35/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run36/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run37/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run38/toy-protmd/')
makelist('/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/run39/toy-protmd/')
