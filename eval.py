"""
Main training script for NERSC PyTorch examples
"""

# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch.distributed as dist

# Locals
from data import get_data_loaders
from trainers import get_trainer
from utils.logging import config_logging
from utils.distributed import init_workers

import torch
from data.parse_md import MDGraphDataset
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed-backend', choices=['mpi', 'nccl', 'gloo'])
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=int)
    add_arg('--rank-gpu', action='store_true')
    add_arg('--ranks-per-node', type=int, default=8)
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_output_dir(config):
    return os.path.expandvars(config['output_dir'])

def get_input_list(config):
    return os.path.expandvars(config['data']['filelist'])

def get_dataset(config):
    print(get_input_list(config))
    return MDGraphDataset(filelist=get_input_list(config))

def get_test_data_loader(config, batch_size=1):
    # Take the test set from the back
    full_dataset = get_dataset(config)
    return DataLoader(full_dataset, batch_size=batch_size,
                      collate_fn=Batch.from_data_list)


def main():
    """Main function"""

    # Initialization
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Prepare output directory
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Load the trainer
    trainer = get_trainer(name=config['trainer'], output_dir=output_dir)

    # Build the model
    model_config = config.get('model', {})
    optimizer_config = config.get('optimizer', {})
    trainer.build_model(optimizer=optimizer_config, **model_config)

    # Checkpoint resume
    trainer.load_checkpoint()

    print(trainer.model)
    print('Parameters:', sum(p.numel() for p in trainer.model.parameters()))
    
    # Load the datasets
    test_loader = get_test_data_loader(config)

    ###### I have hardcoded the value of the checkpoint... need to change to 099...!!!!!!!!!!!
    trainer.predict(test_loader)

    
if __name__ == '__main__':
    main()
