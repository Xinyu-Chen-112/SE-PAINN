___author___ = 'chen xinyu'
___version___ = '1.7.0'

import copy
import itertools
import linecache
import math
import time
import numpy as np
import logging

import torch


class ReadFile:
    def __init__(self, filename):
        linecache.getlines(filename)

        self.system = linecache.getline(filename, 1).strip()

        self.factor = float(linecache.getline(filename, 2).strip())

        a = linecache.getline(filename, 3).split()
        a = [float(i) for i in a]
        a = np.array(a)
        b = linecache.getline(filename, 4).split()
        b = list(map(float, b))
        b = np.array(b)
        c = linecache.getline(filename, 5).split()
        for i, v in enumerate(c): c[i] = float(v)
        c = np.array(c)
        self.basic = np.array([a, b, c])

        self.elements = linecache.getline(filename, 6).split()

        self.cd = linecache.getline(filename, 8).strip().upper()[0]

        self.nums = linecache.getline(filename, 7).split()
        self.nums = [int(i) for i in self.nums]

        self.cell = {"basic": self.basic,
                     "atoms": {}
                     }

        line = 9
        for element, num in zip(self.elements, self.nums):
            for n in range(num):
                coordinate = linecache.getline(filename, line).split()[:3]
                coordinate = np.array([float(i) for i in coordinate])
                if self.cd == 'D':
                    coordinate = (np.matrix(coordinate) @ np.matrix(self.basic)).A[0]
                if element not in self.cell['atoms'].keys():
                    self.cell['atoms'][element] = coordinate[np.newaxis, :]
                else:
                    self.cell['atoms'][element] = np.append(self.cell['atoms'][element], coordinate[np.newaxis, :],
                                                            axis=0)
                line += 1

        linecache.clearcache()

        zzz = np.array([])
        for element in self.cell['atoms'].keys():
            zzz = np.append(zzz, self.cell['atoms'][element][:, 2].flatten())
        under = min(zzz)
        for element in self.cell['atoms'].keys():
            self.cell['atoms'][element] -= np.array([0, 0, under])

    def __len__(self):
        return sum(self.nums)

    def move(self, zbc=None, cd="D", cell=None, save=False, save_name=None, **kwargs):
        # Lattice translation
        cd = cd.upper()
        assert isinstance(zbc, list), "zbc must be a list"
        assert len(zbc) == 2, "the length of zbc must be 2"
        assert cd == "D" or cd == "C", "cd must be 'D' or 'C'"
        if cd == "D":
            if not 0 <= zbc[0] < 1:
                print("NOTICE: The fractional coordinate displacement should be in [0, 1)")
            if not 0 <= zbc[1] < 1:
                print("NOTICE: The fractional coordinate displacement should be in [0, 1)")

        if cell:
            moved_cell = cell
        else:
            moved_cell = copy.deepcopy(self.cell)

        _zbc = copy.deepcopy(zbc)
        _zbc.append(0)

        for element in moved_cell["atoms"].keys():
            if cd == "D":
                moved_cell["atoms"][element] += (np.matrix(_zbc) @ np.matrix(moved_cell['basic'])).tolist()[0]
            elif cd == "C":
                moved_cell["atoms"][element] += _zbc

        if save:
            if save_name:
                sn = save_name
            elif cd == "D":
                sn = self.system + f" move abc{zbc}.vasp"
            elif cd == "C":
                sn = self.system + f" move xyz{zbc}.vasp"
            savefile(save_name=sn, cell=moved_cell, **kwargs)
        else:
            return moved_cell

    def expand(self, abc=[1, 1, 1], cell=None, save=False, save_name=None, **kwargs):
        # Cell expansion
        assert isinstance(abc, list), "abc must be a list of int[Direct]"

        if cell:
            expanded_cell = cell
        else:
            expanded_cell = copy.deepcopy(self.cell)
        _ = copy.deepcopy(expanded_cell["atoms"])

        assert len(abc) == 3, "the length of abc tuple must be 3:a, b, c"
        assert isinstance(abc[0], int), "abc must be a tuple of int Direct"
        assert isinstance(abc[1], int), "abc must be a tuple of int Direct"
        assert isinstance(abc[2], int), "abc must be a tuple of int Direct"

        for ea in range(1, abc[0]):
            for element in _.keys():
                expanded_cell["atoms"][element] = np.append(expanded_cell["atoms"][element],
                                                            _[element] + ea *
                                                            expanded_cell["basic"][0][np.newaxis, :], axis=0)
        _ = copy.deepcopy(expanded_cell["atoms"])
        for eb in range(1, abc[1]):
            for element in _.keys():
                expanded_cell["atoms"][element] = np.append(expanded_cell["atoms"][element],
                                                            _[element] + eb *
                                                            expanded_cell["basic"][1][np.newaxis, :], axis=0)
        _ = copy.deepcopy(expanded_cell["atoms"])
        for ec in range(1, abc[2]):
            for element in _.keys():
                expanded_cell["atoms"][element] = np.append(expanded_cell["atoms"][element],
                                                            _[element] + ec *
                                                            expanded_cell["basic"][2][np.newaxis, :], axis=0)

        expanded_cell["basic"] = (np.matrix([[abc[0], 0, 0],
                                             [0, abc[1], 0],
                                             [0, 0, abc[2]]]) @ expanded_cell["basic"]).A

        if save:
            savefile(save_name=save_name if save_name else self.system + f" expand {abc}.vasp",
                     cell=expanded_cell, **kwargs)
        else:
            return expanded_cell

    def rotate(self, theta=None, cell=None, save=False, save_name=None, **kwargs):
        # The lattice rotates around z-axis in a→b direction
        assert isinstance(theta, (list,
                                  tuple)), "theta must be [degree] or (theta) that represents the counterclockwise rotation angle under the degree or radian measure"
        assert len(theta) == 1, "the angle only have one parameter"
        assert isinstance(theta[0], (int, float)), "the angle must be a int or float"

        if isinstance(theta, list):
            _degree = theta[0]
            _theta = theta[0] * math.pi / 180
        else:
            _degree = theta[0] * 180 / math.pi
            _theta = theta[0]

        if cell:
            rotated_cell = cell
        else:
            rotated_cell = copy.deepcopy(self.cell)

        for element in rotated_cell['atoms'].keys():
            rotated_cell['atoms'][element] = np.matrix(rotated_cell['atoms'][element]) @ np.matrix(
                rotated_cell['basic']).I

        rotated_cell['basic'] = np.matrix(rotated_cell['basic']) @ rotation(theta).T

        for element in rotated_cell['atoms'].keys():
            rotated_cell['atoms'][element] = (rotated_cell['atoms'][element] @ rotated_cell['basic']).A
        rotated_cell['basic'] = rotated_cell['basic'].A

        if save:
            savefile(save_name=save_name if save_name else self.system + f" rotate {_degree}({_theta}).vasp",
                     cell=rotated_cell, **kwargs)
        else:
            return rotated_cell

    def get(self, cell=None, expand=None, move=None, cd="D", rotate=None, save=False, **kwargs):
        # Mix operation
        _cell = cell if cell else copy.deepcopy(self.cell)

        if move:
            _cell = self.move(zbc=move, cd=cd, cell=_cell, save=False)

        if expand:
            _cell = self.expand(abc=expand, cell=_cell, save=False)

        if rotate:
            _cell = self.rotate(theta=rotate, cell=_cell, save=False)

        if save:
            savefile(_cell, **kwargs)
        else:
            return _cell


def rotation(theta):
    assert isinstance(theta, (list,
                              tuple)), "theta must be [degree] or (theta) that represents the counterclockwise rotation angle under the degree or radian measure"
    assert len(theta) == 1, "the angle only have one parameter"
    assert isinstance(theta[0], (int, float)), "the angle must be a int or float"

    _theta = theta[0] * math.pi / 180 if isinstance(theta, list) else theta[0]
    return np.matrix([[math.cos(_theta), - math.sin(_theta), 0],
                      [math.sin(_theta), math.cos(_theta), 0],
                      [0, 0, 1]])


def savefile(cell, save_name="new_save.vasp", system='unknown', factor=1.0, to_cd='C', dynamic=None):
    # The coordinates in the cell must be Cartesian coordinates
    assert isinstance(system, str), "system must be str"
    assert isinstance(factor, (int, float)), "factor must be a number, int or float"
    assert to_cd in ['d', 'D', 'c', 'C'], "to_cd must be one of 'd','D','c','C'"

    if to_cd.upper() == "C":
        _scd = "Cartesian"
    else:
        _scd = "Direct"

    with open(save_name, "w") as f:
        f.write(system + "\n")

        f.write(str(factor) + "\n")

        for __ in cell['basic']:
            for _ in __:
                f.write("  " + str(format(_, ".10f")))
            f.write("\n")

        for element in cell['atoms'].keys():
            f.write(" " + element)
        f.write("\n")

        for coordinates in cell['atoms'].values():
            f.write(" " + str(len(coordinates)))
        f.write("\n")

        if dynamic:
            f.write("Selective Dynamics\n")

        f.write(_scd + "\n")

        for element in cell['atoms'].keys():
            _ = copy.deepcopy(cell['atoms'][element])
            for coordinate in _:
                if _scd == "Direct":
                    coordinate = (np.matrix(coordinate) @ np.matrix(cell['basic']).I).A[0]
                for __ in coordinate:
                    f.write("  " + str(format(__, ".9f")))
                if dynamic:
                    assert len(dynamic) == 3
                    for dy in dynamic:
                        assert dy == 'F' or dy == 'T'
                        f.write("  " + dy)
                f.write("\n")


def build(file_up, file_down, distance=3.0, vacuum=20.0,
          theta=20.0, tolerance=0.05, up_weight=1.0, change=False,
          threshold=99.9, log=logging.WARNING, save=False, **kwargs):
    # Construct a double-layer supercell
    assert 0 < theta <= 90, "0 < theta <= 90"
    assert 0 <= up_weight <= 1, "0 <= up_weight <= 1"

    logging.basicConfig(level=log, format="%(asctime)s [%(levelname)s]  %(message)s")

    up = ReadFile(file_up)
    cell_up = up.cell
    up_basic = cell_up['basic'][:2, :2]
    down = ReadFile(file_down)
    cell_down = down.cell
    down_basic = cell_down['basic'][:2, :2]

    find = True
    super_up_basics = []
    circle = 1
    start = time.time()
    while find:
        search = []
        for add_b in range(circle + 1):
            search.append([circle, add_b])
        for add_a in np.arange(circle - 1, -circle, -1):
            search.append([add_a, circle])
        for add_b in np.arange(circle, 0, -1):
            search.append([-circle, add_b])

        for searching in search:
            s_c = np.matrix(searching) @ np.matrix(up_basic)
            s_down = (s_c @ np.matrix(down_basic).I).A[0]
            s_down_a = round(s_down[0])
            s_down_b = round(s_down[1])
            s_down_dis = math.sqrt((s_down_a - s_down[0]) ** 2 + (s_down_b - s_down[1]) ** 2)
            if s_down_dis <= tolerance:
                orientation = math.acos(round((s_c @ np.matrix(up_basic[0]).T).A[0][0] /
                                              (np.linalg.norm(s_c) * np.linalg.norm(up_basic[0])),
                                              7)) * 180 / math.pi
                super_up_basics.append([orientation, searching, s_down_dis])

        super_up_basics.sort(key=lambda x: x[0])
        if len(super_up_basics) >= 2:
            candidate_cells = []
            for i, j in itertools.combinations(list(range(len(super_up_basics))), 2):
                super_theta = abs(super_up_basics[j][0] - super_up_basics[i][0])
                if math.sin(super_theta * math.pi / 180) >= math.sin(theta * math.pi / 180):
                    candidate_basic_1 = np.matrix(super_up_basics[i][1]) @ np.matrix(up_basic)
                    candidate_basic_2 = np.matrix(super_up_basics[j][1]) @ np.matrix(up_basic)
                    s = np.linalg.norm(candidate_basic_1) * np.linalg.norm(candidate_basic_2) \
                        * math.sin(super_theta * math.pi / 180)
                    candidate_cells.append([super_up_basics[i][1:], super_up_basics[j][1:],
                                            super_theta, abs(super_theta - 90), round(s, 9)])
            if len(candidate_cells) != 0:
                candidate_cells.sort(key=lambda x: (x[4], x[3], x[2]))
                super_up_basic_1 = candidate_cells[0][0]
                super_up_basic_2 = candidate_cells[0][1]
                find = False

        if find:
            circle += 1
            assert time.time() - start <= threshold, "The system is too large, please adjust the parameters"

    logging.info(f"\tuse\t{time.time() - start}\ts")

    logging.info("building super cell now...")
    up_d = np.array([super_up_basic_1[0], super_up_basic_2[0]])
    up_c = (np.matrix(up_d) @ np.matrix(up_basic)).A
    down_x = (np.matrix(up_c) @ np.matrix(down_basic).I).A
    down_d = np.array([[round(_) for _ in __] for __ in down_x])
    down_c = (np.matrix(down_d) @ np.matrix(down_basic)).A
    weight = [up_weight, 1 - up_weight]
    super_basic = torch.einsum('i,ijk->jk', torch.Tensor(weight), torch.Tensor(np.array([up_c, down_c])))

    super_cell = {'basic': [],
                  'atoms': {}}

    logging.info(f"down expand: {down_d.tolist()}")
    down_super_basic = np.matrix(down_d) @ np.matrix(down_basic)
    super_down = np.array([[0, 0], down_d[0], down_d[1], down_d[0] + down_d[1]])
    down_a_min = min(super_down[:, 0])
    down_a_max = max(super_down[:, 0])
    down_b_min = min(super_down[:, 1])
    down_b_max = max(super_down[:, 1])
    down_coordinate = np.array([0, 0])
    for down_a in np.arange(down_a_min, down_a_max + 1):
        for down_b in np.arange(down_b_min, down_b_max + 1):
            coordinate = down_coordinate + down_a * down_basic[0] + down_b * down_basic[1]
            coordinate = (np.matrix(coordinate) @ down_super_basic.I).A
            if 0 <= round(coordinate[0][0], 9) < 1 and 0 <= round(coordinate[0][1], 9) < 1:
                for element in cell_down['atoms'].keys():
                    _ = copy.deepcopy(cell_down['atoms'][element])
                    if change:
                        xy = _[:, :2]
                        z = _[:, 2]
                        xy += down_a * down_basic[0] + down_b * down_basic[1]
                        xy = np.matrix(xy) @ down_super_basic.I
                        xy = (xy @ np.matrix(super_basic)).A
                        _ = np.c_[xy, z]
                    else:
                        _ += down_a * cell_down['basic'][0] + down_b * cell_down['basic'][1]
                    if element not in super_cell['atoms'].keys():
                        super_cell['atoms'][element] = _
                    else:
                        super_cell['atoms'][element] = np.append(super_cell['atoms'][element], _, axis=0)

    zzz = np.array([])
    for element in cell_down['atoms'].keys():
        zzz = np.append(zzz, cell_down['atoms'][element][:, 2].flatten())
    over = max(zzz)

    logging.info(f"up expand:   {up_d.tolist()}")
    up_super_basic = np.matrix(up_d) @ np.matrix(up_basic)
    super_up = np.array([[0, 0], up_d[0], up_d[1], up_d[0] + up_d[1]])
    up_a_min = min(super_up[:, 0])
    up_a_max = max(super_up[:, 0])
    up_b_min = min(super_up[:, 1])
    up_b_max = max(super_up[:, 1])
    up_coordinate = np.array([0, 0])
    for up_a in np.arange(up_a_min, up_a_max + 1):
        for up_b in np.arange(up_b_min, up_b_max + 1):
            coordinate = up_coordinate + up_a * up_basic[0] + up_b * up_basic[1]
            coordinate = (np.matrix(coordinate) @ np.matrix(up_super_basic).I).A
            if 0 <= round(coordinate[0][0], 9) < 1 and 0 <= round(coordinate[0][1], 9) < 1:
                for element in cell_up['atoms'].keys():
                    _ = copy.deepcopy(cell_up['atoms'][element])
                    xy = _[:, :2]
                    z = _[:, 2]
                    xy += up_a * up_basic[0] + up_b * up_basic[1]
                    z += over + distance
                    if change:
                        xy = np.matrix(xy) @ up_super_basic.I
                        xy = (xy @ np.matrix(super_basic)).A
                    _ = np.c_[xy, z]
                    if element not in super_cell['atoms'].keys():
                        super_cell['atoms'][element] = _
                    else:
                        super_cell['atoms'][element] = np.append(super_cell['atoms'][element], _, axis=0)

    ab = np.c_[super_basic.tolist(), [0, 0]]
    super_z = np.array([])
    for element in super_cell['atoms'].keys():
        super_z = np.append(super_z, super_cell['atoms'][element][:, 2].flatten())
    super_over = max(super_z)
    height = super_over + vacuum
    c = np.array([[0, 0, height]])
    super_cell['basic'] = np.append(ab, c, axis=0)

    if save:
        up_vector = up_super_basic.A[0] + up_super_basic.A[1]
        vector_down = (np.matrix(up_vector) @ np.matrix(down_basic).I).A[0]
        vector_down_a = round(vector_down[0])
        vector_down_b = round(vector_down[1])
        vector_down_dis = math.sqrt((vector_down_a - vector_down[0]) ** 2 +
                                    (vector_down_b - vector_down[1]) ** 2)
        dis = max(0, super_up_basic_1[1], super_up_basic_2[1], vector_down_dis)
        expand = [up_d.tolist(), down_d.tolist()]

        sa = super_basic.tolist()[0]
        ax = math.acos(round(sa[0] / np.linalg.norm(sa), 7)) * 180 / math.pi
        ori = round(ax, 9) if sa[1] >= 0 else round(360 - ax, 9)

        # system = str(expand) + "/" + str(round(dis, 9)) + "/" + str(ori)
        system = over + 0.5 * distance
        savefile(super_cell, system=str(system), **kwargs)
        logging.info("※※※build finish※※※")
    else:
        return super_cell


if __name__ == '__main__':
    # Step 1: Read the initial vasp file of the upper material.
    m = ReadFile(r"2d-poscars/C.vasp")
    # Step 2: Perform displacement and rotation operations on the upper material.
    m.get(move=[0.00, 0.00], cd='D', rotate=[0.00], save=True, save_name='_.vasp')
    """
    move: list      Coordinates of aob plane displacement.
    cd: str         Choose in ['d', 'D', 'c', 'C'],
                    representing whether the coordinates of move are fractional coordinates or Cartesian coordinates.
    rotate: list/tuple    Rotation angle around z-axis in a→b direction. List represents Degree, Tuple represents Radian.

    save: bool      Whether to save.
    save_name: str  Saving path of the intermediate file.
    to_cd: str      Choose in ['d', 'D', 'c', 'C'], Save as fractional coordinates or Cartesian coordinates.
    """
    # Step 3: Construct a double-layer structure.
    build('_.vasp', r"2d-poscars/C.vasp", distance=3.0, vacuum=20, theta=20, tolerance=0.05, up_weight=0.5, change=True, save=True,
          save_name='POSCAR', to_cd='C')
    """
    Required parameters:
        file_up     The vasp file path of the upper material.
                    If there is displacement or rotation, this should be an intermediate file.
        file_down   The vasp file path of the underlying material.
    Optional parameters:
        distance    Interlayer distance.
        vacuum      Vacuum layer thickness.
        theta       Range of supercell angle, theta <= γ <= 180-theta.
        tolerance   Maximum mismatch of the lower lattice at the endpoint of the supercell (fractional coordinates).
        up_weight   The basis vectors of the supercell are obtained by the weighted average of the basis vectors of the upper and lower unit cells.
                    This parameter controls the weight of the basis vectors of the upper unit cell, and the value range is [0, 1].
                    =0 means there is no distortion in the lower layer, and =1 means there is no distortion in the upper layer.
        change      Determine whether the supercell undergoes uniform distortion.
        threshold   The maximum time allowed to be consumed in each round when searching for the supercell.
        log         Control information output.
        save=bool   Whether to save.
        to_cd       Choose in ['d', 'D', 'c', 'C'], Save as fractional coordinates or Cartesian coordinates.
        dynamic     Whether to set selective dynamics.
        save_name   Saving path of the supercell.
    """
    # Other operation
    h = ReadFile('POSCAR')
    print(len(h))
    # print(h.system)
