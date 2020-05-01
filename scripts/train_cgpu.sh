#!/bin/bash
#SBATCH -C gpu -c 10 -G 1 -n 1 -t 30

module load pytorch/v1.4.0-gpu

srun -n 1 -u python train.py configs/mpnn_bzrmd.yaml
