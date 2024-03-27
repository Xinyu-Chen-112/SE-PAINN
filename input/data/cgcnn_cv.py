import os
import csv
import pickle
import logging
import functools
import numpy as np
from typing import List
from sklearn import preprocessing
from pymatgen.io.vasp import Poscar

import torch
from torch.utils.data import Dataset


class StructureReader:
    def __init__(self, pkl, data_dir,
                 che_cutoff: float = 5.0, che_max_num_nbrs: int = 12,
                 vdw_cutoff: float = 5.0, vdw_max_num_nbrs: int = 12):
        if pkl:
            nbrs_pkl_path = os.path.join(data_dir, "nbrs_data.pkl")
            with open(nbrs_pkl_path, 'rb') as file:
                self.nbrs_data_dict = pickle.load(file)
        self.pkl = pkl
        self.data_dir = data_dir
        self.che_cutoff = che_cutoff
        self.che_max_num_nbrs = che_max_num_nbrs
        self.vdw_cutoff = vdw_cutoff
        self.vdw_max_num_nbrs = vdw_max_num_nbrs

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
        che_nbrs_fea = []
        che_nbrs_idx = []
        vdw_nbrs_fea = []
        vdw_nbrs_idx = []

        if self.pkl:
            nbrs_data = self.nbrs_data_dict[filename]
            """
            {"id.vasp":
                        [
                        num_atoms,  
                        atoms,  
                        [[distance, index],
                         [distance, index],
                         ...],  
                        [[distance, index],
                         [distance, index],
                         ...]  
                        ]
            }
            """
            structure_data = {'num_atoms': torch.tensor([nbrs_data[0]], dtype=torch.int32),
                              'atoms': torch.tensor(nbrs_data[1], dtype=torch.int32),
                              'atoms_embed': torch.tensor([self.node_dict[a] for a in nbrs_data[1]], dtype=torch.float32),
                              }

            for i in range(nbrs_data[0]):
                che = nbrs_data[2][i]
                che_nbrs_fea.append([])
                che_nbrs_idx.append([])
                n_che = 0
                for j in range(len(che[0])):
                    if che[0][j] <= self.che_cutoff and n_che < self.che_max_num_nbrs:
                        che_nbrs_fea[i].append(che[0][j])
                        che_nbrs_idx[i].append(che[1][j])
                        n_che += 1
                    else:
                        break
                if n_che < self.che_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **che neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    che_nbrs_fea[i] += [99999] * (self.che_max_num_nbrs - n_che)
                    che_nbrs_idx[i] += [i] * (self.che_max_num_nbrs - n_che)

                vdw = nbrs_data[3][i]
                vdw_nbrs_fea.append([])
                vdw_nbrs_idx.append([])
                n_vdw = 0
                for j in range(len(vdw[0])):
                    if vdw[0][j] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs:
                        vdw_nbrs_fea[i].append(vdw[0][j])
                        vdw_nbrs_idx[i].append(vdw[1][j])
                        n_vdw += 1
                    else:
                        break
                if n_vdw < self.vdw_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **vdw neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    vdw_nbrs_fea[i] += [99999] * (self.vdw_max_num_nbrs - n_vdw)
                    vdw_nbrs_idx[i] += [i] * (self.vdw_max_num_nbrs - n_vdw)

        else:
            file_path = os.path.join(self.data_dir, filename)
            poscar = Poscar.from_file(file_path)
            cut = float(poscar.comment)
            structure = poscar.structure
            coords = structure.cart_coords
            ceng = coords[:, 2] > cut

            structure_data = {'num_atoms': torch.tensor([len(structure)], dtype=torch.int32),
                              'atoms': torch.tensor(structure.atomic_numbers, dtype=torch.int32),
                              'atoms_embed': torch.tensor([self.node_dict[a] for a in structure.atomic_numbers], dtype=torch.float32),
                              }

            max_cutoff = max(self.che_cutoff, self.vdw_cutoff)  # Reduce the number of searches from (a+b) to max(a, b) times
            all_nbrs = structure.get_all_neighbors(max_cutoff)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

            for i in range(len(structure)):
                nbrs = all_nbrs[i]
                n_che = 0
                n_vdw = 0
                che_nbrs_fea.append([])
                che_nbrs_idx.append([])
                vdw_nbrs_fea.append([])
                vdw_nbrs_idx.append([])
                for j in range(len(nbrs)):
                    if ceng[i] == ceng[nbrs[j][2]] and nbrs[j][1] <= self.che_cutoff and n_che < self.che_max_num_nbrs:
                        che_nbrs_fea[i].append(nbrs[j][1])
                        che_nbrs_idx[i].append(nbrs[j][2])
                        n_che += 1
                    elif ceng[i] != ceng[nbrs[j][2]] and nbrs[j][1] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs:
                        vdw_nbrs_fea[i].append(nbrs[j][1])
                        vdw_nbrs_idx[i].append(nbrs[j][2])
                        n_vdw += 1

                if n_che < self.che_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **che neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    for k in range(self.che_max_num_nbrs - n_che):
                        che_nbrs_fea[i].append(99999)
                        che_nbrs_idx[i].append(i)
                if n_vdw < self.vdw_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **vdw neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    for k in range(self.vdw_max_num_nbrs - n_vdw):
                        vdw_nbrs_fea[i].append(99999)
                        vdw_nbrs_idx[i].append(i)

        structure_data['che_nbrs_fea'] = torch.tensor(che_nbrs_fea, dtype=torch.float32)
        structure_data['che_nbrs_idx'] = torch.tensor(che_nbrs_idx, dtype=torch.int32)
        structure_data['vdw_nbrs_fea'] = torch.tensor(vdw_nbrs_fea, dtype=torch.float32)
        structure_data['vdw_nbrs_idx'] = torch.tensor(vdw_nbrs_idx, dtype=torch.int32)
        return structure_data


class Data(Dataset):
    def __init__(self, pkl, data_dir,
                 che_cutoff: float = 5.0, che_max_num_nbrs: int = 12,
                 vdw_cutoff: float = 5.0, vdw_max_num_nbrs: int = 12):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        id_prop_file = os.path.join(data_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file, encoding='utf-8', mode='r') as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]  # [['id.vasp', target]...]

        self.strucreader = StructureReader(pkl, data_dir, che_cutoff, che_max_num_nbrs, vdw_cutoff, vdw_max_num_nbrs)

    def __len__(self):  # Return the number of samples
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        filename, target = self.id_prop_data[idx]
        fea = self.strucreader(filename)
        fea['target'] = torch.tensor([float(target)], dtype=torch.float32)
        return fea  # {'num_atoms', 'atoms', 'atoms_embed', 'che_nbrs_fea', 'che_nbrs_idx', 'vdw_nbrs_fea', 'vdw_nbrs_idx', 'target'}


def collate_batch(dataset: List[dict]):
    # collate list of dict to dict of tensor
    dict_of_lists = {k: [info[k] for info in dataset] for k in dataset[0].keys()}
    collated = {k: torch.cat(v) for k, v in dict_of_lists.items()}

    edge_offset = torch.cumsum(torch.cat((torch.tensor([0]), collated['num_atoms'][:-1])), dim=0)
    edge_offset = torch.repeat_interleave(edge_offset, collated['num_atoms'])
    collated['che_nbrs_idx'] = collated['che_nbrs_idx'] + edge_offset.unsqueeze(-1)
    collated['vdw_nbrs_idx'] = collated['vdw_nbrs_idx'] + edge_offset.unsqueeze(-1)

    return collated
