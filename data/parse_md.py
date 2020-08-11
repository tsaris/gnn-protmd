"""
Molecular dynamics Datasets
"""

import os
import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch

import random

def load_graph(filename):
    """Load one graph from an npz file"""

    with np.load(filename) as npzfile:
        edge_np = npzfile['edgelist']
        dist3 = npzfile['distlist']
        nd_labels = npzfile['nodefeat']

    edge_index = torch.tensor(edge_np, dtype=torch.long)
    edge_attr = torch.tensor(dist3, dtype=torch.float)
    x = torch.tensor(nd_labels, dtype=torch.float)
    
    # Make the labels
    if filename.endswith('off.npz'): y = torch.tensor([0], dtype=torch.int)
    if filename.endswith('on.npz'): y = torch.tensor([1], dtype=torch.int)

    return Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)

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

        random.shuffle(filenames)
        random.shuffle(filenames)
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
