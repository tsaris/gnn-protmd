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

def parse_pdb(path):

    listSim = []
    listSimAll = []
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
                #npSim = npSim.reshape(npSim.shape[0], npSim.shape[2], npSim.shape[1])

                if cnt!=0:

                    print(npSim[cnt])
                    print(npSim[cnt-1])

                    loc = npSim[cnt-1]
                    vel = npSim[cnt-1] - npSim[cnt]
                    print(tmp)

                    scaler = MinMaxScaler()
                    distXYZ = scaler.fit_transform(distXYZ)
                    
                    #print(npSim)
                    exit(-1)
            
                
                # Save the file
                #file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/graphs/%d_ras_%s.npz"%(cnt, label)
                #np.savez(file_name, edgelist=edge_np, nodefeat=nd_labels, distlist=distXYZ, dist3list=dist3, dist3Clist=dist3C)

            if line[0] == 'ENDMDL': 
                cnt+=1

parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb_test/tmp.pdb")
