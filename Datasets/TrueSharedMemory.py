# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import uuid
import resource
from time import sleep

import numpy as np

from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import resource_tracker


class SharedDict:
    """
    Truly-shared RAM and memory-mapped hard disk data usable across parallel CPU workers.
    """
    def __init__(self, specs):
        self.specs = specs

        self.dict_id = str(uuid.uuid4())[:8]

        self.created = {}

        # Shared memory can create a lot of file descriptors
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Increase soft limit to hard limit just in case
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

    def __setitem__(self, key, value):
        self.start_worker()

        assert isinstance(value, dict), 'Shared Memory must be a episode dict'

        num_episodes = key.stem.split('/')[-1].split('_')[1]

        for spec, data in value.items():
            name = self.dict_id + num_episodes + spec

            mmap = isinstance(data, np.memmap)  # Whether to memory map

            # Shared integers
            if spec == 'id':
                mem = self.create([data], name)
                mem[0] = data
                mem.shm.close()
            # Shared numpy
            elif data.nbytes > 0:
                # Hard disk memory-map link
                if mmap:
                    mem = self.create(list(str(data.filename)), name + 'mmap')
                    mem.shm.close()
                # Shared RAM memory
                else:
                    mem = self.create(data, name)  # Create
                    mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf)
                    mem_[:] = data[:]  # Set data
                    mem.close()
                    mem = self.create([0], name + 'mmap')  # Set to False, no memory mapping
                    mem.shm.close()

            # Data shape
            mem = self.create([1] if np.isscalar(data) else list(data.shape), name + 'shape')
            mem.shm.close()

    def create(self, data, name=''):
        # Two ways to create a truly shared RAM memory in Python
        method = (ShareableList if isinstance(data, list) else SharedMemory)

        # Try to create shared memory link
        try:
            mem = method(data, name=name) if isinstance(data, list) \
                else method(create=True, name=name,  size=data.nbytes)
        except FileExistsError:
            mem = method(name=name)  # But if exists, retrieve existing shared memory link

        self.created.update({name: method})

        return mem

    def __getitem__(self, key):
        # Account for potential delay
        for _ in range(2400):
            try:
                return self._getitem(key)
            except FileNotFoundError as e:
                sleep(1)
        raise(e)

    def _getitem(self, key):
        num_episodes = key.stem.split('/')[-1].split('_')[1]

        episode = {}

        for spec in self.keys():
            name = self.dict_id + num_episodes + spec

            # Integer
            if spec == 'id':
                mem = ShareableList(name=name)
                episode[spec] = int(mem[0])
                mem.shm.close()
            # Numpy array
            else:
                # Shape
                mem = ShareableList(name=name + 'shape')
                shape = tuple(mem)
                mem.shm.close()

                if 0 in shape:
                    episode[spec] = np.zeros(shape)  # Empty array
                else:
                    # Whether memory mapped
                    mem = ShareableList(name=name + 'mmap')
                    is_mmap = list(mem)
                    mem.shm.close()
                    # View of memory
                    episode[spec] = Mem(name, shape, mmap_name=None if 0 in is_mmap else ''.join(is_mmap))

        return episode

    def keys(self):
        return self.specs.keys() | {'id'}

    def __del__(self):
        self.cleanup()

    def start_worker(self):
        # Hacky fix for https://bugs.python.org/issue38119
        if not self.created:
            check_rtype = lambda func: lambda name, rtype: None if rtype == 'shared_memory' else func(name, rtype)
            resource_tracker.register = check_rtype(resource_tracker.register)
            resource_tracker.unregister = check_rtype(resource_tracker.unregister)

            if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
                del resource_tracker._CLEANUP_FUNCS["shared_memory"]

    def cleanup(self):
        for name, method in self.created.items():
            mem = method(name=name)

            if isinstance(mem, ShareableList):
                mem = mem.shm

            mem.close()
            mem.unlink()  # Unlink shared memory, assumes each worker is uniquely assigned the episodes to create()


class Mem:
    """
    A special view into shared memory or memory mapped data that handles index-based reads and writes.
    """
    def __init__(self, name, shape, mmap_name=None):
        self.name, self.shape, self.mmap_name = name, shape, mmap_name

    def __getitem__(self, idx):
        if self.mmap_name is None:
            # Truly-shared RAM access
            mem = SharedMemory(name=self.name)

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)[idx].copy()

            mem.close()

            return value
        else:
            # Read from memory-mapped hard disk file rather than shared RAM
            return np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)[idx]

    def __setitem__(self, idx, value):
        if self.mmap_name is None:
            # Shared RAM memory
            mem = SharedMemory(name=self.name)

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)  # Un-mutable shape!
            value[idx] = value

            mem.close()
        else:
            # Hard disk memory-map link
            mem = np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)
            mem[idx] = value
            mem.flush()  # Write to hard disk

    def __len__(self):
        mem = ShareableList(name=self.name + 'shape')  # Shape
        size = mem[0]  # Batch dim
        mem.shm.close()
        return size
