# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import glob
import inspect
import itertools
import os

import torch
from torch.utils.data import Dataset

from PIL.Image import Image

import torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

from World.Memory import Batch
from Hyperparams.minihydra import instantiate, Args, added_modules, open_yaml


# Returns a path to an existing Memory directory or an instantiated Pytorch Dataset
def load_dataset(path, dataset_config, allow_memory=True, train=True, **kwargs):
    # Allow config as string path
    if isinstance(dataset_config, str):
        dataset_config = Args({'_target_': dataset_config})

    # If dataset is a directory path, return the string directory path
    if allow_memory and is_valid_path(dataset_config._target_, dir_path=True):
        return dataset_config._target_  # Note: stream=false if called in Env

    # Add torchvision datasets to module search for config instantiation
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

    # Return the Dataset module
    if is_valid_path(dataset_config._target_, module_path=True):
        return instantiate(dataset_config)

    if train is not None:
        path += ('Downloaded_Train/' if train else 'Downloaded_Eval/')
    os.makedirs(path, exist_ok=True)

    # Different datasets have different specs
    root_specs = [dict(root=path), {}]
    train_specs = [] if train is None else [dict(train=train),
                                            dict(version='2021_' + 'train' if train else 'valid'),
                                            dict(subset='training' if train else 'testing'),
                                            dict(split='train' if train else 'test'), {}]
    download_specs = [dict(download=True), {}]
    transform_specs = [dict(transform=None), {}]

    dataset = None

    # Instantiate dataset
    for all_specs in itertools.product(root_specs, train_specs, download_specs, transform_specs):
        try:
            root_spec, train_spec, download_spec, transform_spec = all_specs
            specs = dict(**root_spec, **train_spec, **download_spec, **transform_spec)
            specs = {key: specs[key] for key in set(specs) - set(dataset_config)}
            specs.update(kwargs)
            with Lock(path + 'lock'):  # System-wide mutex-lock
                dataset = instantiate(dataset_config, **specs)
        except (TypeError, ValueError):
            continue
        break

    assert dataset, 'Could not find Dataset.'

    return dataset


# Check if is valid path for instantiation
def is_valid_path(path, dir_path=False, module_path=False, module=False):
    truth = False

    if dir_path:
        try:
            truth = os.path.exists(path)
        except FileNotFoundError:
            pass

    if module_path and not truth:
        try:
            truth = os.path.exists(path.replace('.', '/').rsplit('/', 1)[0])  # Doesn't check all the way to module
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
    if isinstance(datums, Batch):
        return datums

    if isinstance(datums, dict):
        return Batch(datums)

    if not isinstance(datums, Batch):
        # Potentially extract by variable name
        # For now assuming obs, label
        obs, label, *_ = datums

        # May assume image uint8
        dtype = torch.uint8 if len(obs.shape) == 3 and len(obs) in [0, 3] else torch.float32

        return Batch({'obs': torch.as_tensor(obs, dtype=dtype),
                      'label': torch.as_tensor(label), 'done': True})


class Transform(Dataset):
    def __init__(self, dataset, transform=None):
        # Inherit attributes of given dataset
        self.__dict__.update(dataset.__dict__)

        # Map inputs
        self.__dataset, self.__transform = dataset, transform

    def __getitem__(self, idx):
        x, y = self.__dataset.__getitem__(idx)
        x, y = F.to_tensor(x) if isinstance(x, Image) else x, y
        x = (self.__transform or (lambda _: _))(x)  # Transform
        return x, y

    def __len__(self):
        return self.__dataset.__len__()


def add_batch_dim(datum):
    return datum[None, ...] if datum.shape else datum.view(1, 1)


def get_dataset_path(dataset_config, path):
    dataset_class_name = getattr(dataset_config, '_target_',
                                 dataset_config).rsplit('.', 1)[-1] + '/' if dataset_config else ''

    count = 0

    for file in sorted(glob.glob(path + dataset_class_name + '*/*.yaml')):
        if dataset_config == open_yaml(file):
            count = int(file.rsplit('/', 2)[-2])
        else:
            count += 1

    return f'{dataset_class_name}{count}/'
