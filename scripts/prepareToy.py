import numpy as np
from  scipy.spatial.distance import euclidean
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist
from itertools import combinations
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

residues = ['1ALA', '2GLU', '3ASP', '4VAL', '5GLY', '6SER', 
            '7ASN', '8LYSH', '9GLY', '10ALA']

dist_cut = 5
edge_np_lst_on, nd_labels_lst_on, distXYZ_lst_on, dist3_lst_on, dist3C_lst_on = [], [], [], [], []
edge_np_lst_off, nd_labels_lst_off, distXYZ_lst_off, dist3_lst_off, dist3C_lst_off = [], [], [], [], []

def make_files(path_dir, listSim, cnt, cnt_on, cnt_off, temporal_fq, label):
    npSim = np.asarray(listSim, dtype=np.float32)

    # Make all the combinations
    edge_np = combinations(np.arange(npSim.shape[0]), 2)
    edge_np = np.array(list(edge_np))
    edge_np_f = np.flip(edge_np)
    edge_np = np.concatenate((edge_np, edge_np_f[::-1]), axis=0)

    # Edge features
    distX = np.array(pdist(npSim[:,[1]].astype('float')))
    distX2 = np.concatenate((distX, distX), axis=0)
    distX3 = distX2.reshape(distX2.shape[0], 1)
    
    distY = np.array(pdist(npSim[:,[2]].astype('float')))
    distY2 = np.concatenate((distY, distY), axis=0)
    distY3 = distY2.reshape(distY2.shape[0], 1)

    distZ = np.array(pdist(npSim[:,[3]].astype('float')))
    distZ2 = np.concatenate((distZ, distZ), axis=0)
    distZ3 = distZ2.reshape(distZ2.shape[0], 1)
    
    # Euclidean Distance
    dist = np.array(pdist(npSim[:,-3:].astype('float')))
    dist2 = np.concatenate((dist, dist), axis=0)
    dist3 = dist2.reshape(dist2.shape[0], 1)
    
    # Remove the edges and edge features with distance > X A
    list_ = []
    for i in range(0, len(dist3)):
        if (dist3[i]>dist_cut): list_.append(i)
    edge_np = np.delete(edge_np, list_, axis=0)
    dist3 = np.delete(dist3, list_, axis=0)
    distX3 = np.delete(distX3, list_, axis=0)
    distY3 = np.delete(distY3, list_, axis=0)
    distZ3 = np.delete(distZ3, list_, axis=0)
    distXYZ = np.hstack((distX3, distY3, distZ3))
    
    # Another normilization
    dist3C = dist3/dist_cut # Do later on so I can use absoture number for the cut
    
    # Normalization
    scaler = MinMaxScaler()
    distXYZ = scaler.fit_transform(distXYZ)
    scaler = MinMaxScaler()
    dist3 = scaler.fit_transform(dist3)
    
    # Make the node type
    nd_labels = tf.keras.utils.to_categorical(npSim[:,0], num_classes=24)

    global edge_np_lst_on, nd_labels_lst_on, distXYZ_lst_on, dist3_lst_on, dist3C_lst_on
    global edge_np_lst_off, nd_labels_lst_off, distXYZ_lst_off, dist3_lst_off, dist3C_lst_off
        
    # Save the file
    ## Think what happens the first time thqta cnt_on = 0 and cnt_off = 0  ? ###############################
    if (cnt_on!=0 and label=='on'):
        edge_np_lst_on.append(edge_np)
        nd_labels_lst_on.append(nd_labels)
        distXYZ_lst_on.append(distXYZ)
        dist3_lst_on.append(dist3)
        dist3C_lst_on.append(dist3C)

    if (cnt_off!=0 and label=='off'):
        edge_np_lst_off.append(edge_np)
        nd_labels_lst_off.append(nd_labels)
        distXYZ_lst_off.append(distXYZ)
        dist3_lst_off.append(dist3)
        dist3C_lst_off.append(dist3C)

    if (cnt_on%temporal_fq==0 and cnt_on!=0 and label=='on'):
        file_name = "%s/%d_ras_%s.npz"%(path_dir, cnt, label)
        np.savez(file_name, edgelist=np.asarray(edge_np_lst_on), 
                 nodefeat=np.asarray(nd_labels_lst_on), 
                 distlist=np.asarray(distXYZ_lst_on), 
                 dist3list=np.asarray(dist3_lst_on), 
                 dist3Clist=np.asarray(dist3C_lst_on))
        edge_np_lst_on, nd_labels_lst_on, distXYZ_lst_on, dist3_lst_on, dist3C_lst_on = [], [], [], [], []

    if (cnt_off%temporal_fq==0 and cnt_off!=0 and label=='off'):
        file_name = "%s/%d_ras_%s.npz"%(path_dir, cnt, label)
        np.savez(file_name, edgelist=np.asarray(edge_np_lst_off), 
                 nodefeat=np.asarray(nd_labels_lst_off), 
                 distlist=np.asarray(distXYZ_lst_off), 
                 dist3list=np.asarray(dist3_lst_off), 
                 dist3Clist=np.asarray(dist3C_lst_off))
        edge_np_lst_off, nd_labels_lst_off, distXYZ_lst_off, dist3_lst_off, dist3C_lst_off = [], [], [], [], []



def parse_pdb(pdb_file, label_file, path_dir, temporal_fq):

    f = open(label_file, "r")
    labels = f.readlines()

    listSim = []
    cnt_on = 0
    cnt_off = 0
    cnt = 0

    # Parse the pdb file
    with open(pdb_file, 'r') as f:
        line = f.readline()

        tmp_line = line.split()
        step = tmp_line[6]

        while line:
            line = f.readline()
            line = line.split()

            # Make sure itsn't the EOF
            if len(line) == 0: break
            
            if line[0] in residues:
                tmp = []
                res = residues.index(line[0])
                tmp.append(str(res))
                pos = line[3:6]
                pos = tmp + pos
                listSim.append(pos)

            if line[0] == 'Protein' and int(line[6]) != step:
                if(labels[cnt].split()[1] == "Turn"): label = 'on'
                else: label = 'off'
                make_files(path_dir, listSim, cnt, cnt_on, cnt_off, temporal_fq, label)

            if line[0] == 'Protein':
                listSim = []
                step = line[6]
                cnt+=1
                if (label=='on'): cnt_on+=1
                if (label=='off'): cnt_off+=1

    #make_files(listSim, cnt, cnt_on, cnt_off, temporal_fq, label)



parse_pdb("/gpfs/alpine/proj-shared/stf011/atsaris/results/toy-protmd/WT/md.gro", 
          "/gpfs/alpine/proj-shared/stf011/atsaris/results/toy-protmd/WT/state.txt", 
          "/gpfs/alpine/proj-shared/stf011/atsaris/results/toy-protmd/WT_graphs/",
          temporal_fq=1)
