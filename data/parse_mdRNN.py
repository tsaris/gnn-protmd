"""
Molecular dynamics Datasets
"""

import os
import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch, DataLoader

import random

def load_graph(filename):
    """Load one graph from an npz file"""

    with np.load(filename) as npzfile:
        edge_np_0 = npzfile['edgelist_0']
        dist3_0 = npzfile['distlist_0']
        nd_labels_0 = npzfile['nodefeat_0']

        edge_np_1 = npzfile['edgelist_1']
        dist3_1 = npzfile['distlist_1']
        nd_labels_1 = npzfile['nodefeat_1']

        edge_np_2 = npzfile['edgelist_2']
        dist3_2 = npzfile['distlist_2']
        nd_labels_2 = npzfile['nodefeat_2']


    edge_index_0 = torch.tensor(edge_np_0, dtype=torch.long)
    edge_attr_0 = torch.tensor(dist3_0, dtype=torch.float)
    x_0 = torch.tensor(nd_labels_0, dtype=torch.float)

    edge_index_1 = torch.tensor(edge_np_1, dtype=torch.long)
    edge_attr_1 = torch.tensor(dist3_1, dtype=torch.float)
    x_1 = torch.tensor(nd_labels_1, dtype=torch.float)

    edge_index_2 = torch.tensor(edge_np_2, dtype=torch.long)
    edge_attr_2 = torch.tensor(dist3_2, dtype=torch.float)
    x_2 = torch.tensor(nd_labels_2, dtype=torch.float)
    
    # Make the labels
    if filename.endswith('off.npz'): y = torch.tensor([0], dtype=torch.int)
    if filename.endswith('on.npz'): y = torch.tensor([1], dtype=torch.int)

    tmp0 = Data(x=x_0, edge_index=edge_index_0.t().contiguous(), y=y, edge_attr=edge_attr_0)
    tmp1 = Data(x=x_1, edge_index=edge_index_1.t().contiguous(), y=y, edge_attr=edge_attr_1)
    tmp2 = Data(x=x_2, edge_index=edge_index_2.t().contiguous(), y=y, edge_attr=edge_attr_2)

    return [tmp0, tmp1, tmp2]

class MDGraphDataset(Dataset):
    """PyTorch dataset specification for MD protein graphs"""

    def __init__(self, input_dir=None, n_samples=None, filelist=None):
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist), header=None)
            filenames = np.array(self.metadata.values)
            filenames = list(filenames.reshape(filenames.shape[0]))

        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.endswith('.npz')])

        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')

        #random.shuffle(filenames)
        #random.shuffle(filenames)
        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        
    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_datasets(n_train, n_valid, input_dir=None, filelist=None):

    data = MDGraphDataset(input_dir=input_dir, n_samples=n_train+n_valid, filelist=filelist)

    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    loader_args = dict(collate_fn=Batch.from_data_list)
    return train_data, valid_data, loader_args
