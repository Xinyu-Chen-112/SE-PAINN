import sys
import pickle
import argparse
import numpy as np
from pymatgen.io.vasp import Poscar


parser = argparse.ArgumentParser(description='convert input data')
parser.add_argument('--che_max_num_nbrs', default=32, type=int, help='Max number of che nbrs to save')
parser.add_argument('--vdw_max_num_nbrs', default=32, type=int, help='Max number of vdw nbrs to save')
args = parser.parse_args(sys.argv[1:])

data = np.genfromtxt("id_prop.csv", encoding="utf-8", delimiter=',', dtype=str)

nbrs_data = {}
xid = 0
for filename in data[:, 0]:
    if xid % 1000 == 0:
        print("\t\t", xid)

    poscar = Poscar.from_file(f"poscars/{filename}")
    cut = float(poscar.comment)
    structure = poscar.structure
    all_coords = structure.cart_coords
    ceng = all_coords[:, 2] > cut

    all_nbrs = structure.get_all_neighbors(15)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    information = [len(structure),
                   list(structure.atomic_numbers),
                   0,
                   [],
                   []]
    """
    [
    num_atoms,
    atoms,
    state,
    [[index, distance],
     [index, distance],
     ...],
    [[index, distance],
     [index, distance],
     ...]
    ]
    """
    for i in range(len(structure)):
        nbrs = all_nbrs[i]
        n_che = 0
        n_vdw = 0
        che_info = [[], []]
        vdw_info = [[], []]
        for j in range(len(nbrs)):
            if ceng[i] == ceng[nbrs[j][2]] and n_che < args.che_max_num_nbrs:
                che_info[0].append([i, nbrs[j][2]])
                che_info[1].append(nbrs[j][1])
                n_che += 1
            elif ceng[i] != ceng[nbrs[j][2]] and n_vdw < args.vdw_max_num_nbrs:
                vdw_info[0].append([i, nbrs[j][2]])
                vdw_info[1].append(nbrs[j][1])
                n_vdw += 1
        information[3].append(che_info)
        information[4].append(vdw_info)

    nbrs_data[filename] = information
    xid += 1

with open('nbrs_data.pkl', 'wb') as file:
    pickle.dump(nbrs_data, file)
