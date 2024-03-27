import os
import csv
import pickle
import logging
import functools
import numpy as np
from typing import List
from sklearn import preprocessing
from pymatgen.core import Structure

import torch
from torch.utils.data import Dataset


class StructureReader:
    def __init__(self, pkl, data_dir, add_same=False,
                 cutoff: float = 5.0, max_num_nbrs: int = 12):
        if pkl:
            nbrs_pkl_path = os.path.join(data_dir, "nbrs_data.pkl")
            with open(nbrs_pkl_path, 'rb') as file:
                self.nbrs_data_dict = pickle.load(file)
        self.pkl = pkl
        self.data_dir = data_dir
        self.add_same = add_same
        self.cutoff = cutoff
        self.max_num_nbrs = max_num_nbrs

        assert os.path.exists("atom_init.txt"), 'atom_init.txt does not exist!'
        with open("atom_init.txt", mode='r', newline="", encoding="utf-8") as f:
            node_data = csv.reader(f)
            next(f)
            node_data = [[float(i) for i in row] for row in node_data]
            node_data = np.array(node_data)
            atomic_num = node_data[:, 0]
            scaled_node_data = preprocessing.scale(node_data, axis=0)
            self.node_dict = {int(atomic_num[i]): scaled_node_data[i].tolist() for i in range(len(atomic_num))}

    def __call__(self, filename):
        cutnumnbrs = [[], [], []]

        if self.pkl:
            nbrs_data = self.nbrs_data_dict[filename]
            """
            {"id.vasp":
                        [
                        num_atoms,
                        atoms,
                        [[pairs, diff, distance],
                         [pairs, diff, distance],
                         ...]
                        ]
            }
            """
            structure_data = {'num_atoms': torch.tensor([nbrs_data[0]], dtype=torch.int32),
                              'atoms': torch.tensor(nbrs_data[1], dtype=torch.int32),
                              'atoms_embed': torch.tensor([self.node_dict[a] for a in nbrs_data[1]], dtype=torch.float32),
                              }

            for i in range(nbrs_data[0]):
                nbrs = nbrs_data[2][i]
                nn = 0
                pre_dis = "sweet dreams"
                for j in range(len(nbrs[-1])):
                    if self.add_same:
                        if round(nbrs[2][j], 9) == pre_dis \
                                or (nbrs[2][j] <= self.cutoff and nn < self.max_num_nbrs):
                            cutnumnbrs[0].append(nbrs[0][j])
                            cutnumnbrs[1].append(nbrs[1][j])
                            cutnumnbrs[2].append(nbrs[2][j])
                            nn += 1
                            pre_dis = round(nbrs[2][j], 9)
                        else:
                            break
                    else:
                        if nbrs[2][j] <= self.cutoff and nn < self.max_num_nbrs:
                            cutnumnbrs[0].append(nbrs[0][j])
                            cutnumnbrs[1].append(nbrs[1][j])
                            cutnumnbrs[2].append(nbrs[2][j])
                            nn += 1
                        else:
                            break

                if nn < self.max_num_nbrs:
                    logging.debug(f'{filename} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    cutnumnbrs[0] += [[i, i]] * (self.max_num_nbrs - nn)
                    cutnumnbrs[1] += [[0, 0, 0]] * (self.max_num_nbrs - nn)
                    cutnumnbrs[2] += [99999] * (self.max_num_nbrs - nn)

        else:
            file_path = os.path.join(self.data_dir, filename)
            structure = Structure.from_file(file_path)

            structure_data = {'num_atoms': torch.tensor([len(structure)], dtype=torch.int32),
                              'atoms': torch.tensor(structure.atomic_numbers, dtype=torch.int32),
                              'atoms_embed': torch.tensor([self.node_dict[a] for a in structure.atomic_numbers], dtype=torch.float32)
                              }

            all_nbrs = structure.get_all_neighbors(self.cutoff)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

            for i in range(len(structure)):
                nbrs = all_nbrs[i]
                nn = 0
                pre_dis = "sweet dreams"
                for j in range(len(nbrs)):
                    if self.add_same:
                        if round(nbrs[j][1], 9) == pre_dis or nn < self.max_num_nbrs:
                            cutnumnbrs[0].append([i, nbrs[j][2]])
                            cutnumnbrs[1].append((np.array(nbrs[j].coords) - np.array(structure[i].coords)).tolist())
                            cutnumnbrs[2].append(nbrs[j][1])
                            nn += 1
                            pre_dis = round(nbrs[j][1], 9)
                        else:
                            break
                    else:
                        if nn < self.max_num_nbrs:
                            cutnumnbrs[0].append([i, nbrs[j][2]])
                            cutnumnbrs[1].append((np.array(nbrs[j].coords) - np.array(structure[i].coords)).tolist())
                            cutnumnbrs[2].append(nbrs[j][1])
                            nn += 1
                        else:
                            break

                if nn < self.max_num_nbrs:
                    logging.debug(f'{filename} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    for k in range(self.max_num_nbrs - nn):
                        cutnumnbrs[0].append([i, i])
                        cutnumnbrs[1].append([0, 0, 0])
                        cutnumnbrs[2].append(99999)

        structure_data['num_pairs'] = torch.tensor([len(cutnumnbrs[0])], dtype=torch.int32)
        structure_data['pairs'] = torch.tensor(cutnumnbrs[0], dtype=torch.int32)
        structure_data['diff'] = torch.tensor(cutnumnbrs[1], dtype=torch.float32)
        structure_data['distance'] = torch.tensor(cutnumnbrs[2], dtype=torch.float32)

        return structure_data


class Data(Dataset):
    def __init__(self, pkl, data_dir, add_same=False,
                 cutoff: float = 5.0, max_num_nbrs: int = 12):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        id_prop_file = os.path.join(data_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file, encoding='utf-8', mode='r') as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]  # [['id.vasp', target]...]

        self.strucreader = StructureReader(pkl, data_dir, add_same, cutoff, max_num_nbrs)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        filename, target = self.id_prop_data[idx]
        fea = self.strucreader(filename)
        fea['target'] = torch.tensor([float(target)], dtype=torch.float32)
        return fea  # {'num_atoms', 'atoms', 'atoms_embed', 'num_pairs', 'pairs', 'diff', 'distance', 'target'}


def collate_batch(dataset: List[dict]):
    # collate list of dict to dict of tensor
    dict_of_lists = {k: [info[k] for info in dataset] for k in dataset[0].keys()}
    collated = {k: torch.cat(v) for k, v in dict_of_lists.items()}

    edge_offset = torch.cumsum(torch.cat((torch.tensor([0]), collated['num_atoms'][:-1])), dim=0)
    edge_offset = torch.repeat_interleave(edge_offset, collated['num_pairs'])
    collated['pairs'] = collated['pairs'] + edge_offset.unsqueeze(-1)

    return collated
