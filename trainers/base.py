"""
Common PyTorch trainer code.
"""

# System
import os
from os import environ
import time
import logging

# Externals
import numpy as np
import torch

# Locals
from models import get_model

from torch.nn.parallel import DistributedDataParallel

class BaseTrainer(object):
    """Base class for PyTorch trainers.

    This implements the common training logic, logging of summaries, and
    checkpoints. For this repository, it assumes the common single-model
    use-case. More complicated use cases (e.g. GANs) would require adjustment.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed=False, rank=0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        
        if gpu is not None:
            print("Using GPU")
            if environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None:
                local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) # hvd.local_rank()
                self.device = 'cuda:%i' % local_rank
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device('cuda', gpu)
                torch.cuda.set_device(gpu)
        else:
            print("Using CPU")
            self.device = torch.device('cpu')

        self.distributed = distributed
        self.rank = rank
        self.summaries = {}

    def _build_optimizer(self, model, name, **kwargs):
        OptimizerType = getattr(torch.optim, name)
        return OptimizerType(model.parameters(), **kwargs)

    def build_model(self, loss_function, optimizer, **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(**model_args).to(self.device)

        # Distributed data parallelism
        if self.distributed:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)


        # Construct the optimizer
        if (model_args['name'] == 'mpnn5'):
            self.optimizer = torch.optim.Adam([
                dict(params=self.model.reg_params, weight_decay=5e-4),
                dict(params=self.model.non_reg_params, weight_decay=0)
            ], lr=0.01)
        else:
            self.optimizer = self._build_optimizer(self.model, **optimizer)

        # Construct the loss function
        self.loss_func = getattr(torch.nn.functional, loss_function)

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel()
             for p in self.model.parameters()))
        )

    def save_summary(self, summaries):
        """Save summary information"""
        for (key, val) in summaries.items():
            summary_vals = self.summaries.get(key, [])
            self.summaries[key] = summary_vals + [val]

    def write_summaries(self):
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir,
                                    'summaries_%i.npz' % self.rank)
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        # TODO: needs update
        assert self.output_dir is not None

        model_state_dict = (self.model.module.state_dict()
                            if self.distributed
                            else self.model.state_dict())
        checkpoint = dict(checkpoint_id=checkpoint_id,
                          model=model_state_dict,
                          optimizer=self.optimizer.state_dict())

        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        os.makedirs(checkpoint_dir, exist_ok=True)
        #torch.save(dict(model=self.model.state_dict()),
        torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_id=-1):
        assert self.output_dir is not None

        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = os.path.join(
            checkpoint_dir, 'model_checkpoint_030.pth.tar')
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)
            # Train on this epoch
            start_time = time.time()
            summary.update(self.train_epoch(train_data_loader))
            summary['train_time'] = time.time() - start_time
            summary['train_samples'] = len(train_data_loader.sampler)
            summary['train_rate'] = summary['train_samples'] / summary['train_time']
            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader))
                summary['valid_time'] = time.time() - start_time
                summary['valid_samples'] = len(valid_data_loader.sampler)
                summary['valid_rate'] = summary['valid_samples'] / summary['valid_time']
            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None and self.rank==0:
                self.write_checkpoint(checkpoint_id=i)

        return self.summaries
