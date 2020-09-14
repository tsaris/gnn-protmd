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

from gconv_lstm_ChebConv import GConvLSTM
#from gconv_lstm_GCNConv import GConvLSTM

# Locals
from utils.nn import make_mlp

class MPNN6(torch.nn.Module):
    """Message-passing graph neural network

    This model can make graph-level predictions (classifications, regressions)
    via a global aggregation of node features and a linear layer output.
    """

    def __init__(self, n_node_features, n_edge_features, hidden_size, n_outputs=1):
        super().__init__()

        self.recurrent_1 = GConvLSTM(n_node_features, 32, 5)
        self.recurrent_2 = GConvLSTM(32, 16, 5)
        self.lin = torch.nn.Linear(16, 1)
        
    def forward(self, graphs):

        h1, c1, h2, c2 = None, None, None, None
        for graph in graphs:
            h1, c1 = self.recurrent_1(graph.x, graph.edge_index, graph.edge_attr, H=h1, C=c1)
            h2, c2 = self.recurrent_2(h1, graph.edge_index, graph.edge_attr, H=h2, C=c2)


        x = F.relu(h2)
        x = F.dropout(x, training=self.training)
        u = global_add_pool(h2, graphs[0].batch)
        out = self.lin(u)
        return out.view(-1)


def build_model(**kwargs):
    return MPNN6(**kwargs)
