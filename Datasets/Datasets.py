# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import itertools
import os

import torchvision

from Datasets.Memory import Batch
from Hyperparams.minihydra import instantiate, Args, added_modules


# Returns a path to an existing Memory directory or an instantiated Pytorch Dataset
def load_dataset(path, dataset, allow_memory=True, **kwargs):
    # Allow config as string path
    if isinstance(dataset, str):
        dataset = Args({'_target_': dataset})

    # If dataset is a directory path, return the string directory path
    if allow_memory and is_valid_path(dataset._target_, dir_path=True):
        return dataset._target_  # Note: stream=false if called in Env

    name = ''  # Should be path like DatasetTailName/Count/

    # Return directory path if Dataset module has already been saved in Memory
    if allow_memory:
        # See if dataset exists in any DatasetTailName/Count/ .yamls
        # If so, return it
        pass

    # Return the Dataset module
    if is_valid_path(dataset._target_, module_path=True):
        return instantiate(dataset)

    # Return a Dataset based on non-default modules like torchvision
    assert is_valid_path(dataset._target_, module=True), 'Not a valid Dataset instantiation argument.'

    # Add torchvision, torchvision.datasets to module search during dataset config instantiation
    added_modules.update({torchvision, torchvision.datasets})

    train = getattr(dataset, 'train', None)
    if train is not None:
        path += '/' + name + ('Downloaded_Train' if train else 'Downloaded_Eval')
    os.makedirs(path, exist_ok=True)

    # Different datasets have different specs
    root_specs = [dict(root=path), {}]
    train_specs = [] if train is None else [dict(train=train),
                                            dict(version='2021_' + 'train' if train else 'valid'),
                                            dict(subset='training' if train else 'testing'),
                                            dict(split='train' if train else 'test'), {}]
    download_specs = [dict(download=True), {}]
    transform_specs = [dict(transform=None), {}]

    # Instantiate dataset
    for all_specs in itertools.product(root_specs, train_specs, download_specs, transform_specs):
        try:
            root_spec, train_spec, download_spec, transform_spec = all_specs
            specs = dict(**root_spec, **train_spec, **download_spec, **transform_spec)
            specs.update(kwargs)
            with Lock(path):  # System-wide mutex-lock
                dataset = instantiate(dataset, **specs)
        except (TypeError, ValueError):
            continue
        break

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
        for m in added_modules:
            try:
                for key in path.split('.'):
                    m = getattr(m, key)
                truth = True
            except AttributeError:
                continue

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
        self.file = open(self.path)
        self.lock(self.file)

    def __exit__(self, _type, value, tb):
        self.unlock(self.file)
        self.file.close()


def to_experience(data):
    if not isinstance(data, (dict, Batch)):
        pass


def make_card(dataset):
    pass
