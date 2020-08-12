import numpy as np
from  scipy.spatial.distance import euclidean
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist
from itertools import combinations
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL', 'HSD']

dist_cut = 5

import os
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE']) # hvd.size()
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK']) # hvd.rank()
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) # hvd.local_rank()

def parse_pdb(path, label, sample_fq=1):

    listSim = []
    cnt = 0

    # Parse the pdb file
    with open(path, 'r') as f:
        line = f.readline()

        while line:
            line = f.readline()
            line = line.split()

            # Make sure itsn't the EOF
            if len(line) == 0: break

            if (line[0] == 'MODEL') and (cnt%sample_fq) == 0:
                listSim = []

            if line[0] == 'ATOM' and line[2] == 'CA' and (cnt%sample_fq) == 0:
                tmp = []
                res = residues.index(line[3])
                tmp.append(str(res))
                pos = line[6:9]
                pos = tmp + pos
                listSim.append(pos)

            if line[0] == 'ENDMDL' and (cnt%sample_fq) == 0:
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

                # Save the file
                file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_August2020/pdbs/tmp/%d_ras_%d_%s.npz"%(cnt, local_rank, label)
                np.savez(file_name, edgelist=edge_np, nodefeat=nd_labels, distlist=distXYZ, dist3list=dist3, dist3Clist=dist3C)

            if line[0] == 'ENDMDL': 
                cnt+=1


print(world_size, world_rank, local_rank)
file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_August2020/pdbs/tmp/x%d"%(local_rank)
print(file_name)
parse_pdb(file_name, "on", sample_fq=1)
