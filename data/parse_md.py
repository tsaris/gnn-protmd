"""
TUD Datasets

See:
* https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset
* https://chrsmrrs.github.io/datasets/docs/datasets/
"""

import os
import numpy as np

from  scipy.spatial.distance import euclidean, cosine
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist

import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data


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


def load_graph(filename):

    protein = np.loadtxt(filename)
    
    # Make all the combinations
    mesh = np.array(np.meshgrid(np.arange(protein.shape[0]), np.arange(protein.shape[0])))
    edge_np = mesh.T.reshape(-1, 2)

    # Remove the self-edge 
    list_ = []
    for i in range(0, len(edge_np)):
        if (edge_np[i][0]==edge_np[i][1]): list_.append(i)
    edge_np = np.delete(edge_np, list_, axis=0)

    # Make the edge tensor
    edge_index = torch.tensor(edge_np, dtype=torch.long)
    # Make the node tensor
    x = torch.tensor(protein[:,2:10], dtype=torch.float)
    
    # Make the labels
    if (filename.endswith('off_a.txt')): y = torch.tensor([0], dtype=torch.int)
    if (filename.endswith('on_a.txt')): y = torch.tensor([1], dtype=torch.int)

    # Fake the edge features
    tmp = np.zeros((edge_np.shape[0], 1))
    edge_attr = torch.tensor(tmp, dtype=torch.float)

    return Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)

class MDGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                            if f.endswith('.txt')])
        self.filenames = filenames if n_samples is None else filenames[:n_samples]

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_datasets(n_train, n_valid, input_dir=None):

    data = MDGraphDataset(input_dir=input_dir, n_samples=n_train+n_valid)

    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    loader_args = dict(collate_fn=torch_geometric.data.Batch.from_data_list)
    return train_data, valid_data, loader_args
