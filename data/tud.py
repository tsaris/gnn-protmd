"""
TUD Datasets

See:
* https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset
* https://chrsmrrs.github.io/datasets/docs/datasets/
"""

import os

import torch
import torch_geometric
from torch_geometric.datasets import TUDataset

def get_datasets(n_train, n_valid, tud_name, download_path='datasets'):
    """Download and construct a TUD Dataset and randomly split into train/val"""
    download_path = os.path.join(
        os.path.expandvars(download_path),
        tud_name)

    # The entire dataset; download if necessary
    dataset = torch_geometric.datasets.TUDataset(
        name=tud_name, root=download_path,
        use_node_attr=True, use_edge_attr=True)

    for i in dataset:
        print(i)

    # Split into training and validation
    train_dataset, valid_dataset, _ = torch.utils.data.random_split(
        dataset, [n_train, n_valid, len(dataset)-n_train-n_valid])

    # Use PyTorch-Geometric batch collate function
    loader_args = dict(collate_fn=torch_geometric.data.Batch.from_data_list)
    return train_dataset, valid_dataset, loader_args
