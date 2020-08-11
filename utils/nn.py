"""Utility functions for building networks"""
from __future__ import division

import torch
import torch.nn as nn
from itertools import repeat

import torch
from torch.autograd import Function

#from torch_scatter.utils.ext import get_func
#from torch_scatter.utils.gen import gen

def make_mlp(input_size, sizes,
             hidden_activation='ReLU',
             output_activation='ReLU',
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers.

    Args:
        input_size (int): input network dimensions
        sizes (list[int]): list of layer sizes, including output layer
        hidden_activation: activation function for hidden layers
        output_activation: activation function for final layer
        layer_norm (bool): whether to apply LayerNorm between every layer
    """

    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes

    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)



