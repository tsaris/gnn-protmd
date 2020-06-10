import numpy as np
from  scipy.spatial.distance import euclidean
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist
from itertools import combinations
import tensorflow as tf

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL']

def parse_pdb(path, label):
    # Parse residue, atom type and atomic coordinates
    protein_data = []
    protein_data_all = []
    residue_depth_percentile = []
    res_ = None
    res_i = None
    res_c = None

    # Parse the pdb file
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in lines:

            if row[:4] == 'ATOM':
                res_i = row[7:13].strip()
                res_ = row[17:20]
                res_c = [row[30:38].strip(), row[38:46].strip(), row[47:54].strip()]
                if (res_ == 'HSD'): continue
                ress = residues.index(res_)
                res_data = [res_i, ress] + res_c
                protein_data.append(res_data)

            if row[:6] == 'ENDMDL':
                protein_data_all.append(np.asarray(protein_data))
                protein_data = []

    # Calculate the features
    for protein_data in protein_data_all:
        chain_c = protein_data[:,2:5].astype('float')
        chain_centroid = np.mean(chain_c,axis=0)
        residue_depth = np.array([euclidean(chain_centroid, c) for c in chain_c])
        residue_depth_percentile.append([1- perc(residue_depth, d)/100.0 for d in residue_depth])
    
    # Add the feature to data
    protein_data_all = np.asarray(protein_data_all, dtype=np.float32)
    residue_depth_percentile = np.asarray(residue_depth_percentile, dtype=np.float32)

    for x in range(0, protein_data_all.shape[0]):
        res_depth_perc = residue_depth_percentile[x].reshape(residue_depth_percentile[x].shape[0], 1)
        protein  = np.hstack((res_depth_perc, protein_data_all[x,:,1:5]))

        # Make all the combinations
        edge_np = combinations(np.arange(protein.shape[0]), 2)
        edge_np = np.array(list(edge_np))
        edge_np_f = np.flip(edge_np)
        edge_np = np.concatenate((edge_np, edge_np_f[::-1]), axis=0)

        # Edge features
        dist = np.array(pdist(protein[:,-3:].astype('float')))
        dist2 = np.concatenate((dist, dist), axis=0)
        dist3 = dist2.reshape(dist2.shape[0], 1)

        # Remove the edges and edge features with distance > 10 A (find an optimized way of this)
        list_ = []
        for i in range(0, len(dist3)):
            if (dist3[i]>6): list_.append(i)
        edge_np = np.delete(edge_np, list_, axis=0)
        dist3 = np.delete(dist3, list_, axis=0)

        # Make the node type
        nd_labels = tf.keras.utils.to_categorical(protein[:,1], num_classes=23)
        # Add node feature of relative position
        nd_labels = np.hstack((nd_labels, protein[:,[0]]))

        # Save the file
        file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/graph/%d_ras_%s.npz"%(x, label)
        np.savez(file_name, edgelist=edge_np, distlist=dist3, nodefeat=nd_labels)


parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_on.pdb", "on")
parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_off.pdb", "off")

#parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_on_tmp.pdb", "on")
