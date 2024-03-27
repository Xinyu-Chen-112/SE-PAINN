import json
import numpy as np
import torch
from torch.utils.data import Subset


def gaussiandistance(data,
                     centers,
                     width: float = 0.5):
    rbf_data = torch.exp(-((data.unsqueeze(-1) - centers) ** 2) / width**2)
    return rbf_data


def sinc(data,
         cutoff: float = 5.0,
         embedding_size: int = 20):
    sinc_data = torch.where((data < cutoff).unsqueeze(-1),
                            torch.sin(data.unsqueeze(-1)
                                      * (torch.arange(embedding_size, device=data.device) + 1)
                                      * torch.pi
                                      / cutoff)
                            / data.unsqueeze(-1),
                            torch.tensor(0.0, device=data.device, dtype=data.dtype))
    return sinc_data


def cosine(data,
           sinc_data,
           cutoff: float = 5.0):
    rbf_data = sinc_data * 0.5 * (torch.cos(torch.pi * data / cutoff) + 1).unsqueeze(-1)
    return rbf_data


def split_dataset(dataset,
                  train_ratio=None, val_ratio=0.1, test_ratio=0.1, test=False,
                  save_path='splitdata.json'):

    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    np.random.shuffle(indices)

    if test:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1, "val_ratio  + test_ratio must < 1"
            train_ratio = 1 - val_ratio - test_ratio
            # logging.warning(f'! Warning ! train_ratio is None, using 1 - val_ratio - test_ratio = {train_ratio} as training_ratio.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1, "train_ratio + val_ratio + test_ratio must <= 1"
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        split_index = {"train": indices[:train_size].tolist(),
                       "val": indices[train_size: train_size + val_size].tolist(),
                       "test": indices[train_size + val_size:].tolist()}

    else:
        if train_ratio is None:
            assert val_ratio < 1, "val_ratio must < 1"
            train_ratio = 1 - val_ratio
            # logging.warning(f'! Warning ! train_ratio is None, using 1 - val_ratio = {train_ratio} as train_ratio.')
        else:
            assert train_ratio + val_ratio <= 1, "train_ratio + val_ratio must <= 1"
        train_size = int(train_ratio * total_size)
        split_index = {"train": indices[:train_size].tolist(),
                       "val": indices[train_size:].tolist()}

    # Save split file
    with open(save_path, "w") as f:
        json.dump(split_index, f)

    # Split the dataset
    splited_dataset = {}
    for key, indices in split_index.items():
        splited_dataset[key] = Subset(dataset, indices)

    return splited_dataset


class EarlyStopping:
    def __init__(self, escape=100, threshold=-0.0000001, patience=10):
        self.escape = escape
        self.n = 0

        self.threshold = threshold
        self.patience = patience
        self.counter = 0

        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        self.n += 1

        if self.n > self.escape:
            if val_loss - best_loss >= self.threshold:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

        return self.early_stop


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

    @property
    def now(self):
        return self.count, self.sum, self.avg
