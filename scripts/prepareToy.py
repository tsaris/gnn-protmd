import numpy as np
from  scipy.spatial.distance import euclidean
from scipy.stats import percentileofscore as perc
from scipy.spatial.distance import pdist
from itertools import combinations
import tensorflow as tf

#residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
#            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
#            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
#            'UNK', 'VAL', 'HSD']

residues = ['1ALA', '2GLU', '3ASP', '4VAL', '5GLY', '6SER', 
            '7ASN', '8LYSH', '9GLY', '10ALA']

dist_cut = 5

def parse_pdb(path, label, sample_fq=1):

    listSim = []
    cnt = 0

    # Parse the pdb file
    with open(path, 'r') as f:
        line = f.readline()

        tmp_line = line.split()
        step = tmp_line[6]

        while line:
            line = f.readline()
            line = line.split()

            # Make sure itsn't the EOF
            if len(line) == 0: break
            
            if line[0] == 'Protein':
                listSim = []
                step = line[6]

            if line[0] in residues:
                tmp = []
                res = residues.index(line[0])
                tmp.append(str(res))
                pos = line[3:6]
                pos = tmp + pos
                listSim.append(pos)

            if line[0] == 'Protein' and int(line[6]) != step:
                print(step) # It needs to go 3 times, it only goes 2?
                

parse_pdb("/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/toy-protmd/tmp.gro", "on", sample_fq=10)
