# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import itertools
import os

from Hyperparams.minihydra import instantiate, Args


def load_dataset(path, dataset, allow_memory=True, **kwargs):
    # Allow config as string path
    if isinstance(dataset, str):
        dataset = Args({'_target_': dataset})

    # If dataset is a directory path, return the string directory path
    if allow_memory and valid_path(dataset._target_, dir_path=True):
        return dataset._target_  # Note: stream=false

    name = ''

    # Return directory path if Dataset module has already been saved in Memory
    if allow_memory:
        pass

    # Return the Dataset module
    if valid_path(dataset._target_, module_path=True):
        return instantiate(dataset)

    # Return a Dataset based on non-default modules like torchvision
    assert valid_path(dataset._target_, module=True), 'Not a valid Dataset instantiation argument.'

    train = getattr(dataset, 'train', None)
    if train is not None:
        path += '/' + name + ('_Train' if train else '_Eval')
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


def valid_path(path, dir_path=False, module_path=False, module=False):
    pass


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
