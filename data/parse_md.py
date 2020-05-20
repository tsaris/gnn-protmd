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

def load_graph(filename):

    protein = np.loadtxt(filename)
    
    # Make all the combinations
    mesh = np.array(np.meshgrid(np.arange(protein.shape[0]), np.arange(protein.shape[0])))
    edge_np = mesh.T.reshape(-1, 2)

    # Remove the self-edge (find an optimized way of this)
    list_ = []
    for i in range(0, len(edge_np)):
        if (edge_np[i][0]==edge_np[i][1]): list_.append(i)
    edge_np = np.delete(edge_np, list_, axis=0)

    # Edge features
    dist = np.array(pdist(protein[:,-3:].astype('float')))
    dist2 = np.concatenate((dist, dist))
    dist3 = dist2.reshape(dist2.shape[0], 1)

    # Remove the edges and edge features with distance > 10 A (find an optimized way of this)
    list_ = []
    for i in range(0, len(dist3)):
        if (dist3[i]>10): list_.append(i)
    edge_np = np.delete(edge_np, list_, axis=0)
    dist3 = np.delete(dist3, list_, axis=0)

    # Make the edge tensor
    edge_index = torch.tensor(edge_np, dtype=torch.long)
    edge_attr = torch.tensor(dist3, dtype=torch.float)
    # Make the node tensor
    x = torch.tensor(protein[:,2:3], dtype=torch.float)
    
    # Make the labels
    if (filename.endswith('off_a.txt')): y = torch.tensor([0], dtype=torch.int)
    if (filename.endswith('on_a.txt')): y = torch.tensor([1], dtype=torch.int)

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
