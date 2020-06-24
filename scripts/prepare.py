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

            if line[0] == 'MODEL':
                listSim = []

            if line[0] == 'ATOM' and line[2] == 'CA':
                tmp = []
                res = residues.index(line[3])
                tmp.append(str(res))
                pos = line[5:8]
                pos = tmp + pos
                listSim.append(pos)

            if line[0] == 'ENDMDL':
                npSim = np.asarray(listSim, dtype=np.float32)                
                cent = np.mean(npSim[:,1:4], axis=0)
                res_depth = np.array([euclidean(cent, c) for c in npSim[:,1:4]])
                res_depth_perc = [1- perc(res_depth, d)/100.0 for d in res_depth]
                res_depth_perc = np.array(res_depth_perc, dtype=np.float32)
                npSim = np.hstack((res_depth_perc.reshape(res_depth_perc.shape[0],1), npSim))

                # Make all the combinations
                edge_np = combinations(np.arange(npSim.shape[0]), 2)
                edge_np = np.array(list(edge_np))
                edge_np_f = np.flip(edge_np)
                edge_np = np.concatenate((edge_np, edge_np_f[::-1]), axis=0)

                # Edge features
                dist = np.array(pdist(npSim[:,-3:].astype('float')))
                dist2 = np.concatenate((dist, dist), axis=0)
                dist3 = dist2.reshape(dist2.shape[0], 1)

                # Remove the edges and edge features with distance > X A
                list_ = []
                for i in range(0, len(dist3)):
                    if (dist3[i]>dist_cut): list_.append(i)
                edge_np = np.delete(edge_np, list_, axis=0)
                dist3 = np.delete(dist3, list_, axis=0)
                dist3 = dist3/dist_cut

                # Make the node type
                nd_labels = tf.keras.utils.to_categorical(npSim[:,1], num_classes=24)
                # Add node feature of relative position
                nd_labels = np.hstack((nd_labels, npSim[:,[0]]))

                # Save the file
                file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/graph_full_kRas_/%d_ras_%s.npz"%(cnt, label)
                np.savez(file_name, edgelist=edge_np, distlist=dist3, nodefeat=nd_labels)
                cnt+=1
                



parse_pdb("/gpfs/alpine/world-shared/bif112/AllosteryLearning_GraphNeuralNet/whole_traj/TTR/0-100ns_3NEX_holo.pdb", "on")
parse_pdb("/gpfs/alpine/world-shared/bif112/AllosteryLearning_GraphNeuralNet/whole_traj/TTR/0-100ns_3NEX_apo.pdb", "off")

#parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_on.pdb", "on")
#parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_off.pdb", "off")
#parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_on_tmp.pdb", "on")
