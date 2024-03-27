import sys
import pickle
import argparse
import numpy as np
from pymatgen.io.vasp import Poscar


parser = argparse.ArgumentParser(description='convert input data')
parser.add_argument('--max_num_nbrs', default=32, type=int, help='Max number of nbrs to save')
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
                   list(ceng),
                   []]
    """
    [
    num_atoms,
    atoms,
    state,
    ceng,
    [[index, distance],
     [index, distance],
     ...]
    ]
    """
    for i in range(len(structure)):
        nbrs = all_nbrs[i]
        nn = 0
        info = [[], []]
        for j in range(len(nbrs)):
            if nn < args.max_num_nbrs:
                info[0].append([i, nbrs[j][2]])
                info[1].append(nbrs[j][1])
                nn += 1
            else:
                break
        information[4].append(info)

    nbrs_data[filename] = information
    xid += 1

with open('nbrs_data.pkl', 'wb') as file:
    pickle.dump(nbrs_data, file)
