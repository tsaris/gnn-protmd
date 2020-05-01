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

# Locals
from utils.nn import make_mlp

class MPNN(torch.nn.Module):
    """Message-passing graph neural network

    This model can make graph-level predictions (classifications, regressions)
    via a global aggregation of node features and a linear layer output.
    """

    def __init__(self, n_node_features, n_edge_features, hidden_size, n_outputs=1):
        super().__init__()
        
        # Define some NNConv layers.
        # For now using single-layer nn
        self.conv1 = NNConv(n_node_features, hidden_size,
                            make_mlp(n_edge_features,
                                     [n_node_features*hidden_size],
                                     output_activation=None))
        self.conv2 = NNConv(hidden_size, hidden_size,
                            make_mlp(n_edge_features,
                                     [hidden_size*hidden_size],
                                     output_activation=None))

        # Graph output module
        self.mlp = make_mlp(hidden_size, [n_outputs],
                            output_activation=None)

    def forward(self, data):
        """Apply forward pass of model"""

        # Apply the message passing layers
        h = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        h = F.relu(self.conv2(h, data.edge_index, data.edge_attr))

        # Aggregate node features
        u = global_add_pool(h, data.batch)

        # Graph-global output
        return self.mlp(u).squeeze(-1)

def build_model(**kwargs):
    return MPNN(**kwargs)
