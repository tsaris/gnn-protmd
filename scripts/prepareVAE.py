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



def parse_pdb(path):

    listSim = []
    listSimAll = []
    listSim_loc = []
    listSim_vel = []

    cnt = 0

    # Parse the pdb file
    with open(path, 'r') as f:
        line = f.readline()

        while line:
            line = f.readline()
            line = line.split()

            # Make sure itsn't the EOF
            if len(line) == 0: break

            if (line[0] == 'MODEL'):
                listSim = []

            if line[0] == 'ATOM' and line[2] == 'CA':
                tmp = []
                res = residues.index(line[3])
                tmp.append(str(res))
                pos = line[6:9]
                pos = tmp + pos
                listSim.append(pos)

            if line[0] == 'ENDMDL':
                listSimAll.append(listSim)
                npSim = np.asarray(listSimAll, dtype=np.float32)
                npSim = npSim[:,:,-3:]

                if cnt!=0:

                    loc = npSim[cnt-1]
                    vel = npSim[cnt-1] - npSim[cnt]

                    scaler = MinMaxScaler(feature_range=(-1,1))
                    loc = scaler.fit_transform(loc)
                    loc = loc.reshape(loc.shape[1], loc.shape[0])
                    listSim_loc.append(loc)

                    scaler = MinMaxScaler(feature_range=(-1,1))
                    vel = scaler.fit_transform(vel)
                    vel = vel.reshape(vel.shape[1], vel.shape[0])
                    listSim_vel.append(vel)
                
                # Save the file
                #file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/graphs/%d_ras_%s.npz"%(cnt, label)
                #np.savez(file_name, edgelist=edge_np, nodefeat=nd_labels, distlist=distXYZ, dist3list=dist3, dist3Clist=dist3C)

            if line[0] == 'ENDMDL': 
                cnt+=1

    listSim_loc = np.asarray(listSim_loc)
    listSim_vel = np.asarray(listSim_vel)
    return listSim_loc, listSim_vel


loc_0, vel_0 = parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/tmp.pdb")
loc_1, vel_1 = parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/tmp.pdb")
loc_2, vel_2 = parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/tmp.pdb")

loc_0 = loc_0.reshape(1, loc_0.shape[0], loc_0.shape[1], loc_0.shape[2])
loc_1 = loc_1.reshape(1, loc_1.shape[0], loc_1.shape[1], loc_1.shape[2])
loc_2 = loc_2.reshape(1, loc_2.shape[0], loc_2.shape[1], loc_2.shape[2])

loc = np.vstack((loc_0, loc_1, loc_2))

file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/graphs/ras.npy"
np.save(file_name, loc)
