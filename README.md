# gnn-protmd
Deep learning and Graph Neural Networks for proteins and
molecular dynamics simulations

## Contents
* The main script to run is the train.py
* For doing inference on data use the eval.py script
* The sub_dist.lcf shows has to run in data parallel the training

## Contents for the branches
* The LSTM branch uses the same GNN models as the master branch but has temporal component using an LSTM
* The MPNN-Toy branch uses the same models as the master branch but the data loaders are different


The basica GNN operations were used based on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) and [pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
