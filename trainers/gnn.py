"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch
from torch import nn

# Locals
from .base import BaseTrainer

class GNNTrainer(BaseTrainer):
    """Trainer code for basic GNN problems."""

    def __init__(self, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)

    def train_epoch(self, data_loader):
        """Train for one epoch"""

        self.model.train()
        sum_loss = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            #batch = batch.to(self.device)
            batch = [_batch.to(self.device) for _batch in batch]
            self.model.zero_grad()
            batch_output = self.model(batch)

            #print("type:", type(batch))
            #print("Batch:", batch)

            #print("nodes Vs node_features:", batch.x.shape)
            #print("2 Vs edges:", batch.edge_index.shape)
            #print("edges Vs edge features", batch.edge_attr.shape)

            batch_target = batch[0].y.float()
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            loss = batch_loss.item()
            sum_loss += loss
            self.logger.debug('batch %i loss %.3f', i, loss)

        train_loss = sum_loss / (i + 1)
        self.logger.debug('Processed %i batches', (i + 1))
        self.logger.info('Training loss: %.4f', train_loss)

        # Return summaries
        return dict(train_loss=train_loss)

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""

        self.model.eval()
        sum_loss = 0
        sum_correct = 0
        sum_pred = 0

        # Loop over batches
        for i, batch in enumerate(data_loader):
            #batch = batch.to(self.device)
            batch = [_batch.to(self.device) for _batch in batch]
            batch_output = self.model(batch)
            # FIXME: converting labels to float for binary loss with TUDataset
            batch_target = batch[0].y.float()
            loss = self.loss_func(batch_output, batch_target)
            sum_loss += loss.item()

            # FIXME: currently hardcoded for binary classification accuracy
            # Count number of correct predictions
            batch_pred = batch_output > 0
            batch_label = batch_target > 0.5
            n_correct = (batch_pred == batch_label).sum().item()
            sum_correct += n_correct
            # TODO fix this, not working on GPU
            #sum_pred += batch_pred.sum().item()
            #self.logger.debug('batch %i loss %.3f correct %i mean-pred %g',
            #                  i, loss, n_correct, batch_pred.numpy().mean())
            self.logger.debug('batch %i loss %.3f correct %i', i, loss, n_correct)

        # Summarize
        valid_loss = sum_loss / (i + 1)
        valid_acc = sum_correct / len(data_loader.sampler)
        mean_pred = sum_pred / len(data_loader.sampler)
        self.logger.debug('Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('Validation loss: %.4f acc: %.4f mean-pred: %g',
                         valid_loss, valid_acc, mean_pred)

        # Return summaries
        return dict(valid_loss=valid_loss, valid_acc=valid_acc, mean_pred=mean_pred)

    @torch.no_grad()
    def predict(self, data_loader):
        sum_correct = 0
        #for batch in data_loader:
        for i, batch in enumerate(data_loader):
            batch = [_batch.to(self.device) for _batch in batch]
            batch_output = self.model(batch)
            batch_target = batch[0].y.float()
            batch_pred = batch_output > 0
            batch_label = batch_target > 0.5
            n_correct = (batch_pred == batch_label).sum().item()
            sum_correct += n_correct

        # Summarize
        valid_acc = sum_correct / len(data_loader.sampler)
        print(valid_acc)


def get_trainer(**kwargs):
    return GNNTrainer(**kwargs)

def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
