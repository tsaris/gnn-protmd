"""
This module defines a message-passing network based on NNConv layers
from PyTorch-Geometric, as used in the Neural Message Passing for Quantum
Chemistry paper.
"""

# Externals
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import NNConv, global_add_pool, GCNConv

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
import torch_geometric.transforms as T

# Locals
from utils.nn import make_mlp

class MPNN4(torch.nn.Module):
    """Message-passing graph neural network

    This model can make graph-level predictions (classifications, regressions)
    via a global aggregation of node features and a linear layer output.
    """

    def __init__(self, n_node_features, n_edge_features, hidden_size, n_outputs=1):
        super().__init__()

        self.conv1 = GCNConv(n_node_features, hidden_size)
        self.lin1 = torch.nn.Linear(hidden_size, 16)
        self.lin2 = torch.nn.Linear(16, 1)

    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, None
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        u = global_add_pool(h, data.batch)
        out = F.relu(self.lin1(u))
        out = self.lin2(out)
        return out.view(-1)

def build_model(**kwargs):
    return MPNN4(**kwargs)
