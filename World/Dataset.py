# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import glob
import inspect
import itertools
import math
import os
import random
from copy import copy

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL.Image import Image

import torchvision
from torchvision.transforms import functional as F
from tqdm import tqdm

import Utils
from World.Memory import Batch
from minihydra import instantiate, Args, added_modules, open_yaml


# Returns a path to an existing Memory directory or an instantiated Pytorch Dataset
def load_dataset(path, dataset_config, allow_memory=True, train=True, **kwargs):
    if isinstance(dataset_config, Dataset):
        return dataset_config

    # Allow config as string path
    if isinstance(dataset_config, str):
        dataset_config = Args({'_target_': dataset_config})

    # If dataset is a directory path, return the string directory path
    if allow_memory and is_valid_path(dataset_config._target_, dir_path=True) \
            and glob.glob(dataset_config._target_ + 'card.yaml'):
        return dataset_config._target_  # Note: stream=false if called in Env

    # Add torchvision datasets to module search for config instantiation  TODO Add World/Datasets
    pytorch_datasets = {m: getattr(torchvision.datasets, m)
                        for m in dir(torchvision.datasets) if inspect.isclass(getattr(torchvision.datasets, m))
                        and issubclass(getattr(torchvision.datasets, m), Dataset)}
    added_modules.update(pytorch_datasets)

    if dataset_config._target_[:len('torchvision.datasets.')] == 'torchvision.datasets.':
        dataset_config._target_ = dataset_config._target_[len('torchvision.datasets.'):]  # Allow torchvision. syntax

    # Return a Dataset based on a module path or non-default modules like torchvision
    assert is_valid_path(dataset_config._target_, module_path=True, module=True), \
        'Not a valid Dataset instantiation argument.'

    path += get_dataset_path(dataset_config, path)  # DatasetClassName/Count/

    # Return directory path if Dataset module has already been saved in Memory
    if allow_memory:
        if glob.glob(path + '*.yaml'):
            return path

    # Different datasets have different specs
    root_specs = [dict(root=path), {}]
    train_specs = [] if train is None else [dict(train=train),
                                            dict(version='2021_' + 'train' if train else 'valid'),
                                            dict(subset='training' if train else 'testing'),
                                            dict(split='train' if train else 'test'), {}]
    download_specs = [dict(download=True), {}]
    transform_specs = [dict(transform=None), {}]

    dataset = None
    is_torchvision = False

    # From custom module path
    if is_valid_path(dataset_config._target_, module_path=True):
        root_specs = download_specs = transform_specs = [{}]  # Won't assume any signature args except possibly train
    # From torchvision Dataset  TODO It shouldn't re-download for every version of the dataset
    else:
        is_torchvision = True
        if train is not None:
            path += ('Downloaded_Train/' if train else 'Downloaded_Eval/')
        os.makedirs(path, exist_ok=True)

    dataset_config = copy(dataset_config)
    if 'Transform' in dataset_config:
        dataset_config.pop('Transform')
    transform = dataset_config.pop('transform') if 'transform' in dataset_config else None
    subset = dataset_config.pop('subset') if 'subset' in dataset_config else None

    # Instantiate dataset
    for all_specs in itertools.product(root_specs, train_specs, download_specs, transform_specs):
        try:
            root_spec, train_spec, download_spec, transform_spec = all_specs
            specs = dict(**root_spec, **train_spec, **download_spec, **transform_spec)
            specs = {key: specs[key] for key in set(specs) - set(dataset_config)}
            specs.update(kwargs)
            if is_torchvision:
                with Lock(path + 'lock'):  # System-wide mutex-lock
                    dataset = Utils.instantiate(dataset_config, **specs)
            else:
                dataset = Utils.instantiate(dataset_config, **specs)
        except (TypeError, ValueError):
            continue
        break

    assert dataset, 'Could not find Dataset.'

    classes = subset if subset is not None \
        else range(dataset.classes if isinstance(dataset.classes, int)
                   else len(dataset.classes)) if hasattr(dataset, 'classes') \
        else range(dataset.num_classes) if hasattr(dataset, 'num_classes') \
        else dataset.class_to_idx.keys() if hasattr(dataset, 'class_to_idx') \
        else [print(f'Identifying unique classes... '
                    f'This can take some time for large datasets.'),
              sorted(list(set(str(exp[1]) for exp in dataset)))][1]

    setattr(dataset, 'num_classes', len(classes))

    # Can select a subset of classes
    if subset is not None:
        dataset = ClassSubset(dataset, classes, train)

    # Map unique classes to integers
    dataset = ClassToIdx(dataset, classes)

    # Add transforms to dataset
    dataset = Transform(dataset, instantiate(transform if getattr(transform, '_target_', None) else None))

    return dataset


# Computes mean, stddev, low, high
def compute_stats(batches):
    cnt = 0
    fst_moment, snd_moment = None, None
    low, high = np.inf, -np.inf

    for batch in tqdm(batches, 'Computing mean, stddev, low, high for standardization/normalization.'):
        obs = batch.obs if 'obs' in batch else batch[0]
        b, c, *hw = obs.shape
        if not hw:
            *hw, c = c, 1  # At least 1 channel dim and spatial dim - can comment out
        obs = obs.view(b, c, *hw)
        fst_moment = torch.zeros(c) if fst_moment is None else fst_moment
        snd_moment = torch.zeros(c) if snd_moment is None else snd_moment
        nb_pixels = b * math.prod(hw)
        dim = [0, *[2 + i for i in range(len(hw))]]
        sum_ = torch.sum(obs, dim=dim)
        sum_of_square = torch.sum(obs ** 2, dim=dim)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

        low, high = min(obs.min().item(), low), max(obs.max().item(), high)

    stddev = torch.sqrt(snd_moment - fst_moment ** 2)
    stddev[stddev == 0] = 1

    mean, stddev = fst_moment.tolist(), stddev.tolist()
    return Args(mean=mean, stddev=stddev, low=low, high=high)  # Save stat values for future reuse


# Check if is valid path for instantiation
def is_valid_path(path, dir_path=False, module_path=False, module=False):
    truth = False

    if dir_path:
        try:
            truth = os.path.exists(path)
        except FileNotFoundError:
            pass

    if module_path and not truth and path.count('.') > 0:
        try:
            *root, file, module = path.replace('.', '/').rsplit('/', 2)
            root = root[0] + '/' if root else ''
            truth = os.path.exists(root + file + '.py')
        except FileNotFoundError:
            pass

    if module and not truth:
        sub_module, *sub_modules = path.split('.')

        if sub_module in added_modules:
            sub_module = added_modules[sub_module]

            try:
                for key in sub_modules:
                    sub_module = getattr(sub_module, key)
                truth = True
            except AttributeError:
                pass

    return truth


# System-wide mutex lock
# https://stackoverflow.com/a/60214222/22002059
class Lock:
    def __init__(self, path):
        self.path = path

        if os.name == "nt":
            import msvcrt

            def lock(file):
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)

            def unlock(file):
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock(file):
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)

            def unlock(file):
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)

        self.lock, self.unlock = lock, unlock

    def __enter__(self):
        self.file = open(self.path, 'w')
        self.lock(self.file)

    def __exit__(self, _type, value, tb):
        self.unlock(self.file)
        self.file.close()  # Perhaps delete


def datums_as_batch(datums):
    if isinstance(datums, (Batch, dict)):
        if 'done' not in datums:
            datums['done'] = True
        return Batch(datums)
    else:
        # Potentially extract by variable name
        # For now assuming obs, label
        obs, label, *_ = datums

        # May assume image uint8
        # if len(obs.shape) == 4 and int(obs.shape[1]) in [1, 3]:
        #     obs *= 255  # Note: Assumes [0, 1] low, high
        #     dtype = {'dtype': torch.uint8}
        # else:
        #     dtype = {}

        # Note: need to parse label TODO
        obs = torch.as_tensor(obs)
        label = torch.as_tensor(label)

        if len(label.shape) == 1:
            label = label.view(-1, 1)

        return Batch({'obs': obs, 'label': label, 'done': True})


class Transform(Dataset):
    def __init__(self, dataset, transform=None):
        # Inherit attributes of given dataset
        self.__dict__.update(dataset.__dict__)

        # Get transform from config
        if isinstance(transform, (Args, dict)):
            added_modules.update({'torchvision': torchvision})
            transform = instantiate(transform)

        # Map inputs
        self.__dataset, self.__transform = dataset, transform

    def __getitem__(self, idx):
        x, y = self.__dataset.__getitem__(idx)
        x, y = F.to_tensor(x) if isinstance(x, Image) else x, y
        x = (self.__transform or (lambda _: _))(x)  # Transform
        return x, y

    def __len__(self):
        return self.__dataset.__len__()


def get_dataset_path(dataset_config, path):
    dataset_class_name = dataset_config.__name__ if isinstance(dataset_config, Dataset) \
        else getattr(dataset_config, '_target_', dataset_config).rsplit('.', 1)[-1] + '/' if dataset_config \
        else ''

    count = 0

    for file in sorted(glob.glob(path + dataset_class_name + '*/*.yaml')):
        card = open_yaml(file)

        if not hasattr(dataset_config, '_target_'):
            card.pop('_target_')

        if 'stats' in card and 'stats' not in dataset_config:
            card.pop('stats')

        if 'num_classes' in card and 'num_classes' not in dataset_config:
            card.pop('num_classes')

        # Just a shorthand
        if 'Transform' in card:
            card.pop('Transform')
        if 'Transform' in dataset_config:
            dataset_config.pop('Transform')

        if not hasattr(dataset_config, '_target_') and not card or dataset_config.to_dict() == card:
            count = int(file.rsplit('/', 2)[-2])
            break
        else:
            count += 1

    return f'{dataset_class_name}{count}/'


# # Map class labels to Tensor integers
class ClassToIdx(Dataset):
    def __init__(self, dataset, classes):
        # Inherit attributes of given dataset
        self.__dict__.update(dataset.__dict__)

        # Map string labels to integers
        self.__dataset, self.__map = dataset, {str(classes[i]): torch.tensor(i) for i in range(len(classes))}

    def __getitem__(self, idx):
        x, y = self.__dataset.__getitem__(idx)
        return x, self.__map[str(y)]  # Map

    def __len__(self):
        return self.__dataset.__len__()


# Select classes from dataset e.g. python Run.py task=classify/mnist 'env.subset=[0,2,3]'
class ClassSubset(torch.utils.data.Subset):
    def __init__(self, dataset, classes, train=None):
        # Inherit attributes of given dataset
        self.__dict__.update(dataset.__dict__)

        train = '' if train is None else 'train' if train else 'test'

        # Find subset indices which only contain the specified classes, multi-label or single-label
        indices = [i for i in tqdm(range(len(dataset)), desc=f'Selecting subset of classes from {train} dataset...')
                   if str(dataset[i][1]) in map(str, classes)]

        # Initialize
        super().__init__(dataset=dataset, indices=indices)


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(int(seed))
