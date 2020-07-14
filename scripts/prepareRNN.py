import numpy as np
from  scipy.spatial.distance import euclidean
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist
from itertools import combinations
import tensorflow as tf

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL', 'HSD']

dist_cut = 5

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
                pos = line[5:8]
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
                distX3 = distX3/dist_cut

                distY = np.array(pdist(npSim[:,[2]].astype('float')))
                distY2 = np.concatenate((distY, distY), axis=0)
                distY3 = distY2.reshape(distY2.shape[0], 1)
                distY3 = distY3/dist_cut

                distZ = np.array(pdist(npSim[:,[3]].astype('float')))
                distZ2 = np.concatenate((distZ, distZ), axis=0)
                distZ3 = distZ2.reshape(distZ2.shape[0], 1)
                distZ3 = distZ3/dist_cut

                # Euclidean Distance
                dist = np.array(pdist(npSim[:,-3:].astype('float')))
                dist2 = np.concatenate((dist, dist), axis=0)
                dist3 = dist2.reshape(dist2.shape[0], 1)

                # Remove the edges and edge features with distance > X A
                list_ = []
                for i in range(0, len(dist3)):
                    if (dist3[i]>dist_cut): list_.append(i)
                edge_np = np.delete(edge_np, list_, axis=0)
                distX3 = np.delete(dist3, list_, axis=0)
                distY3 = np.delete(dist3, list_, axis=0)
                distZ3 = np.delete(dist3, list_, axis=0)
                distXYZ = np.hstack((distX3, distY3, distZ3))

                # Make the node type
                nd_labels = tf.keras.utils.to_categorical(npSim[:,0], num_classes=24)

                # Save the file
                file_name = "/gpfs/alpine/stf011/world-shared/atsaris/datagnn/datagnn_ras_2020/pdb_cnn/graphs/%d_ras_%s.npz"%(cnt, label)
                np.savez(file_name, edgelist=edge_np, distlist=distXYZ, nodefeat=nd_labels)

            if line[0] == 'ENDMDL': 
                cnt+=1


parse_pdb("/gpfs/alpine/stf011/world-shared/atsaris/datagnn/datagnn_ras_2020/pdb_cnn/tmp_small.pdb", "on", sample_fq=1)
