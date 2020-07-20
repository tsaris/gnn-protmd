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

def parse_pdb(path):

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
            
            if line[0] in residues:
                tmp = []
                res = residues.index(line[0])
                tmp.append(str(res))
                pos = line[3:6]
                pos = tmp + pos
                listSim.append(pos)

            # It will go n-1 times
            if line[0] == 'Protein' and int(line[6]) != step:
                npSim = np.asarray(listSim, dtype=np.float32)

                # Make all the combinations
                edge_np = combinations(np.arange(npSim.shape[0]), 2)
                edge_np = np.array(list(edge_np))
                edge_np_f = np.flip(edge_np)
                edge_np = np.concatenate((edge_np, edge_np_f[::-1]), axis=0)

                # Edge features
                distX = np.array(pdist(npSim[:,[1]].astype('float')))
                distX2 = np.concatenate((distX, distX), axis=0)
                distX3 = distX2.reshape(distX2.shape[0], 1)
                distX3 = distX3/dist_cut

                distY = np.array(pdist(npSim[:,[2]].astype('float')))
                distY2 = np.concatenate((distY, distY), axis=0)
                distY3 = distY2.reshape(distY2.shape[0], 1)
                distY3 = distY3/dist_cut

                distZ = np.array(pdist(npSim[:,[3]].astype('float')))
                distZ2 = np.concatenate((distZ, distZ), axis=0)
                distZ3 = distZ2.reshape(distZ2.shape[0], 1)
                distZ3 = distZ3/dist_cut

                # Euclidean Distance
                dist = np.array(pdist(npSim[:,-3:].astype('float')))
                dist2 = np.concatenate((dist, dist), axis=0)
                dist3 = dist2.reshape(dist2.shape[0], 1)

                # Remove the edges and edge features with distance > X A
                list_ = []
                for i in range(0, len(dist3)):
                    if (dist3[i]>dist_cut): list_.append(i)
                edge_np = np.delete(edge_np, list_, axis=0)
                distX3 = np.delete(dist3, list_, axis=0)
                distY3 = np.delete(dist3, list_, axis=0)
                distZ3 = np.delete(dist3, list_, axis=0)
                distXYZ = np.hstack((distX3, distY3, distZ3))

                # I don't really need to do this, the Euclidean Distance doesn't saved
                dist3 = np.delete(dist3, list_, axis=0)
                dist3 = dist3/dist_cut # Do later on so I can use absoture number for the cut

                # Make the node type
                nd_labels = tf.keras.utils.to_categorical(npSim[:,0], num_classes=24)
                # Add node feature of relative position
                #nd_labels = np.hstack((nd_labels, npSim[:,[0]]))

                # Save the file
                file_name = "/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/toy-protmd/graphs/%d_mdToy.npz"%(cnt)
                np.savez(file_name, edgelist=edge_np, distlist=distXYZ, nodefeat=nd_labels, distlistE=dist3)

            if line[0] == 'Protein':
                listSim = []
                step = line[6]
                cnt+=1

                

parse_pdb("/gpfs/alpine/stf011/world-shared/atsaris/toy-protmd_new/toy-protmd/tmp.gro")
