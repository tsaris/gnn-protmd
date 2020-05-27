import numpy as np
from  scipy.spatial.distance import euclidean, cosine
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL']

def parse_pdb(path, chain='A', all_chains=False, first=False):
    '''
    '''
    # Parse residue, atom type and atomic coordinates
    seq_data = []
    helix_data = []
    beta_data = []
    complex_data = {}
    protein_data = []
    protein_data_all = []
    residue_depth_percentile = []
    res_ = None
    res_i = None
    res_c = None
    sidechain_data = []
    sidechain_flag = False
    sidechain_counter = 0
    model=0
    count=0

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
    protein_data_all = np.asarray(protein_data_all)
    residue_depth_percentile = np.asarray(residue_depth_percentile, dtype='str')
    protein_add_feature = np.zeros((len(protein_data_all),
                                    len(protein_data_all[0]),
                                    len(protein_data_all[0][0])+5))
    protein_add_feature = np.asarray(protein_add_feature, dtype='str')
    tmp2 = ["0.0" for x in range(len(protein_data_all[0][0])+5)]

    for graph_i in range(len(protein_data_all)):
        for j in range(len(protein_data_all[graph_i])):
            tmp = np.append(protein_data_all[graph_i][j], residue_depth_percentile[graph_i][j])
            tmp2[0] = int(1)
            tmp2[1:3] = tmp[:2]
            tmp2[4] = tmp[5]
            tmp2[7:] = tmp[2:5]
            protein_add_feature[graph_i][j] = tmp2

    # Return an arrays of strings
    protein_add_feature = np.asarray(protein_add_feature, dtype='str')
    return protein_add_feature


protein_all_ON = parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_on.pdb")
protein_all_OFF = parse_pdb("/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/pdb/small_off.pdb")
#/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/graph

for i in range(0, protein_all_ON.shape[0]):
    file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/graph/%d_ras_on_a.txt"%(i)
    tmp = protein_all_ON[i].astype(np.float32)
    np.savetxt(file_name, tmp[:,[2,4,7,8,9]], delimiter=' ')

for i in range(0, protein_all_OFF.shape[0]):
    file_name = "/gpfs/alpine/world-shared/stf011/atsaris/datagnn/datagnn_ras_2020/graph/%d_ras_off_a.txt"%(i)
    tmp = protein_all_OFF[i].astype(np.float32)
    np.savetxt(file_name, tmp[:,[2,4,7,8,9]], delimiter=' ')
