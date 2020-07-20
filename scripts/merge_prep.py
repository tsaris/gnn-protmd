import os, re
import numpy as np

input_dir = '/gpfs/alpine/stf011/world-shared/atsaris/datagnn/datagnn_ras_2020/KRAS_r9_sup'
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

list_on, list_off = [], []

for i in range(0, num_list_on.shape[0]):
    if (cnt%freq) == 0:
        on = '%s/%d_ras_on.npz'%(input_dir, num_list_on[i])
        off = '%s/%d_ras_off.npz'%(input_dir, num_list_off[i])
        list_on.append(on)
        list_off.append(off)
    cnt+=1

# Make the pairs (every 3 this time)
list_on_pair = list(zip(list_on, list_on[1:], list_on[2:]))
list_on_pair = list_on_pair[::3]
list_off_pair = list(zip(list_off, list_off[1:], list_off[2:]))
list_off_pair = list_off_pair[::3]

dest = "/gpfs/alpine/stf011/world-shared/atsaris/datagnn/datagnn_ras_2020/KRAS_r9_sup_newgraphs"

cnt=0
for pair in list_on_pair:
    with np.load(pair[0]) as npzfile: edge_np_0 = npzfile['edgelist']; dist3_0 = npzfile['distlist']; nd_labels_0 = npzfile['nodefeat']
    with np.load(pair[1]) as npzfile: edge_np_1 = npzfile['edgelist']; dist3_1 = npzfile['distlist']; nd_labels_1 = npzfile['nodefeat']
    with np.load(pair[2]) as npzfile: edge_np_2 = npzfile['edgelist']; dist3_2 = npzfile['distlist']; nd_labels_2 = npzfile['nodefeat']
    newFileName = "%s/%d_ras_on.npz"%(dest, cnt)
    cnt+=1
    np.savez(newFileName, edgelist_0=edge_np_0, distlist_0=dist3_0, nodefeat_0=nd_labels_0, 
             edgelist_1=edge_np_1, distlist_1=dist3_1, nodefeat_1=nd_labels_1,
             edgelist_2=edge_np_2, distlist_2=dist3_2, nodefeat_2=nd_labels_2)

cnt=0
for pair in list_off_pair:
    with np.load(pair[0]) as npzfile: edge_np_0 = npzfile['edgelist']; dist3_0 = npzfile['distlist']; nd_labels_0 = npzfile['nodefeat']
    with np.load(pair[1]) as npzfile: edge_np_1 = npzfile['edgelist']; dist3_1 = npzfile['distlist']; nd_labels_1 = npzfile['nodefeat']
    with np.load(pair[2]) as npzfile: edge_np_2 = npzfile['edgelist']; dist3_2 = npzfile['distlist']; nd_labels_2 = npzfile['nodefeat']
    newFileName = "%s/%d_ras_off.npz"%(dest, cnt)
    cnt+=1
    np.savez(newFileName, edgelist_0=edge_np_0, distlist_0=dist3_0, nodefeat_0=nd_labels_0, 
             edgelist_1=edge_np_1, distlist_1=dist3_1, nodefeat_1=nd_labels_1,
             edgelist_2=edge_np_2, distlist_2=dist3_2, nodefeat_2=nd_labels_2)
