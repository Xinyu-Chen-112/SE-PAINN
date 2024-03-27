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
    def __init__(self, pkl, data_dir, add_same=False,
                 che_cutoff: float = 5.0, che_max_num_nbrs: int = 12,
                 vdw_cutoff: float = 5.0, vdw_max_num_nbrs: int = 12):
        if pkl:
            nbrs_pkl_path = os.path.join(data_dir, "nbrs_data.pkl")
            with open(nbrs_pkl_path, 'rb') as file:
                self.nbrs_data_dict = pickle.load(file)
        self.pkl = pkl
        self.data_dir = data_dir
        self.add_same = add_same
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
        cutnumnbrs = [[[], []],
                      [[], []]]

        if self.pkl:
            nbrs_data = self.nbrs_data_dict[filename]
            """
            {"id.vasp":
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
            }
            """
            structure_data = {'num_atoms': torch.tensor([nbrs_data[0]], dtype=torch.int32),
                              'atoms': torch.tensor(nbrs_data[1], dtype=torch.int32),
                              'atoms_embed': torch.tensor([self.node_dict[a] for a in nbrs_data[1]], dtype=torch.float32),
                              'state': torch.tensor([nbrs_data[2]], dtype=torch.float32),
                              'che_max_num_nbrs': torch.tensor([self.che_max_num_nbrs], dtype=torch.float32),
                              'vdw_max_num_nbrs': torch.tensor([self.vdw_max_num_nbrs], dtype=torch.float32)
                              }

            for i in range(nbrs_data[0]):
                che = nbrs_data[3][i]
                n_che = 0
                pre_che_dis = "sweet dreams"
                for j in range(len(che[-1])):
                    if self.add_same:
                        if round(che[1][j], 9) == pre_che_dis \
                                or (che[1][j] <= self.che_cutoff and n_che < self.che_max_num_nbrs):
                            cutnumnbrs[0][0].append(che[0][j])
                            cutnumnbrs[0][1].append(che[1][j])
                            n_che += 1
                            pre_che_dis = round(che[1][j], 9)
                        else:
                            break
                    else:
                        if che[1][j] <= self.che_cutoff and n_che < self.che_max_num_nbrs:
                            cutnumnbrs[0][0].append(che[0][j])
                            cutnumnbrs[0][1].append(che[1][j])
                            n_che += 1
                        else:
                            break

                if n_che < self.che_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **che neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    cutnumnbrs[0][0] += [[i, i]] * (self.che_max_num_nbrs - n_che)
                    cutnumnbrs[0][1] += [99999] * (self.che_max_num_nbrs - n_che)

                vdw = nbrs_data[4][i]
                n_vdw = 0
                pre_vdw_dis = "Kiss Kiss Kiss"
                for j in range(len(vdw[-1])):
                    if self.add_same:
                        if round(vdw[1][j], 9) == pre_vdw_dis \
                                or (vdw[1][j] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs):
                            cutnumnbrs[1][0].append(vdw[0][j])
                            cutnumnbrs[1][1].append(vdw[1][j])
                            n_vdw += 1
                            pre_vdw_dis = round(vdw[1][j], 9)
                        else:
                            break
                    else:
                        if vdw[1][j] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs:
                            cutnumnbrs[1][0].append(vdw[0][j])
                            cutnumnbrs[1][1].append(vdw[1][j])
                            n_vdw += 1
                        else:
                            break

                if n_vdw < self.vdw_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **vdw neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    cutnumnbrs[1][0] += [[i, i]] * (self.vdw_max_num_nbrs - n_vdw)
                    cutnumnbrs[1][1] += [99999] * (self.vdw_max_num_nbrs - n_vdw)

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
                              'state': torch.tensor([0.0], dtype=torch.float32),
                              'che_max_num_nbrs': torch.tensor([self.che_max_num_nbrs], dtype=torch.float32),
                              'vdw_max_num_nbrs': torch.tensor([self.vdw_max_num_nbrs], dtype=torch.float32)
                              }

            max_cutoff = max(self.che_cutoff, self.vdw_cutoff)  # Reduce the number of searches from (a+b) to max(a, b) times
            all_nbrs = structure.get_all_neighbors(max_cutoff)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

            for i in range(len(all_nbrs)):
                nbrs = all_nbrs[i]
                n_che = 0
                n_vdw = 0
                pre_che_dis = "sweet dreams"
                pre_vdw_dis = "Kiss Kiss Kiss"
                for j in range(len(nbrs)):
                    if ceng[i] == ceng[nbrs[j][2]]:
                        if self.add_same:
                            if round(nbrs[j][1], 9) == pre_che_dis \
                                    or (nbrs[j][1] <= self.che_cutoff and n_che < self.che_max_num_nbrs):
                                cutnumnbrs[0][0].append([i, nbrs[j][2]])
                                cutnumnbrs[0][1].append(nbrs[j][1])
                                n_che += 1
                                pre_che_dis = round(nbrs[j][1], 9)
                        else:
                            if nbrs[j][1] <= self.che_cutoff and n_che < self.che_max_num_nbrs:
                                cutnumnbrs[0][0].append([i, nbrs[j][2]])
                                cutnumnbrs[0][1].append(nbrs[j][1])
                                n_che += 1

                    else:
                        if self.add_same:
                            if round(nbrs[j][1], 9) == pre_vdw_dis \
                                    or (nbrs[j][1] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs):
                                cutnumnbrs[1][0].append([i, nbrs[j][2]])
                                cutnumnbrs[1][1].append(nbrs[j][1])
                                n_vdw += 1
                                pre_vdw_dis = round(nbrs[j][1], 9)
                        else:
                            if nbrs[j][1] <= self.vdw_cutoff and n_vdw < self.vdw_max_num_nbrs:
                                cutnumnbrs[1][0].append([i, nbrs[j][2]])
                                cutnumnbrs[1][1].append(nbrs[j][1])
                                n_vdw += 1

                if n_che < self.che_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **che neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    for k in range(self.che_max_num_nbrs - n_che):
                        cutnumnbrs[0][0].append([i, i])
                        cutnumnbrs[0][1].append(99999)
                if n_vdw < self.vdw_max_num_nbrs:
                    logging.debug(f'{filename} not find enough **vdw neighbors to build graph. '
                                  'If it happens frequently, consider increase cutoff or decrease max_num_nbrs.')
                    for k in range(self.vdw_max_num_nbrs - n_vdw):
                        cutnumnbrs[1][0].append([i, i])
                        cutnumnbrs[1][1].append(99999)

        structure_data['che_num_pairs'] = torch.tensor([len(cutnumnbrs[0][0])], dtype=torch.int32)
        structure_data['che_index'] = torch.tensor(cutnumnbrs[0][0], dtype=torch.int32)
        structure_data['che_nbrs_fea'] = torch.tensor(cutnumnbrs[0][1], dtype=torch.float32)
        structure_data['vdw_num_pairs'] = torch.tensor([len(cutnumnbrs[1][0])], dtype=torch.int32)
        structure_data['vdw_index'] = torch.tensor(cutnumnbrs[1][0], dtype=torch.int32)
        structure_data['vdw_nbrs_fea'] = torch.tensor(cutnumnbrs[1][1], dtype=torch.float32)
        return structure_data


class Data(Dataset):
    def __init__(self, pkl, data_dir,  add_same=False,
                 che_cutoff: float = 5.0, che_max_num_nbrs: int = 12,
                 vdw_cutoff: float = 5.0, vdw_max_num_nbrs: int = 12):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        id_prop_file = os.path.join(data_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file, encoding='utf-8', mode='r') as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]  # [['id.vasp', target]...]

        self.strucreader = StructureReader(pkl, data_dir,  add_same, che_cutoff, che_max_num_nbrs, vdw_cutoff, vdw_max_num_nbrs)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        filename, target = self.id_prop_data[idx]
        fea = self.strucreader(filename)
        fea['target'] = torch.tensor([float(target)], dtype=torch.float32)
        return fea  # {'num_atoms', 'atoms', 'atoms_embed', 'state', 'che_max_num_nbrs', 'che_num_pairs', 'che_index', 'che_nbrs_fea', 'vdw_max_num_nbrs', 'vdw_num_pairs', 'vdw_index', 'vdw_nbrs_fea', 'target'}


def collate_batch(dataset: List[dict]) -> dict[str, torch.Tensor]:
    # collate list of dict to dict of tensor
    dict_of_lists = {k: [info[k] for info in dataset] for k in dataset[0].keys()}
    collated = {k: torch.cat(v) for k, v in dict_of_lists.items()}

    edge_offset = torch.cumsum(torch.cat((torch.tensor([0]), collated['num_atoms'][:-1])), dim=0)
    che_edge_offset = torch.repeat_interleave(edge_offset, collated['che_num_pairs'])
    collated['che_index'] = collated['che_index'] + che_edge_offset.unsqueeze(-1)
    vdw_edge_offset = torch.repeat_interleave(edge_offset, collated['vdw_num_pairs'])
    collated['vdw_index'] = collated['vdw_index'] + vdw_edge_offset.unsqueeze(-1)

    return collated
