"""
This module defines a message-passing network based on NNConv layers
from PyTorch-Geometric, as used in the Neural Message Passing for Quantum
Chemistry paper.
"""

# Externals
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import NNConv, global_add_pool

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import Set2Set

# Locals
from utils.nn import make_mlp#, Set2Set

class MPNN3(torch.nn.Module):
    """Message-passing graph neural network

    This model can make graph-level predictions (classifications, regressions)
    via a global aggregation of node features and a linear layer output.
    """

    def __init__(self, n_node_features, n_edge_features, hidden_size, n_outputs=1):
        super().__init__()
        self.lin0 = torch.nn.Linear(n_node_features, hidden_size)

        self.conv1 = NNConv(hidden_size, hidden_size,
                            make_mlp(n_edge_features,
                                     [hidden_size*hidden_size],
                                     output_activation=None))
        self.gru = GRU(hidden_size, hidden_size)
        self.set2set = Set2Set(hidden_size, processing_steps=3)
 
        self.lin1 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv1(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

def build_model(**kwargs):
    return MPNN3(**kwargs)
