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


# TODO allow indexing idx _from, _to before of copy!
class SharedDict:
    """
    An Offline dict generalized to manage numpy arrays, integers, and hard disk memory map-links in truly-shared RAM
    memory, rather than serialization/deserialization. Efficiently read-writes data directly across parallel CPU workers
    """
    def __init__(self, specs):
        self.dict_id = str(uuid.uuid4())[:8]

        self.mems = {}
        self.specs = specs

        # Shared memory can create a lot of file descriptors
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Increase soft limit to hard limit just in case
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

    def __setitem__(self, key, value):
        self.start_worker()

        assert isinstance(value, dict), 'Shared Memory must be dict'

        num_episodes = key.stem.split('/')[-1].split('_')[1]

        for spec, data in value.items():
            name = self.dict_id + num_episodes + spec

            # Shared integers
            if spec == 'id':
                mem = self.setdefault(name, [data])
                mem[0] = data
                self.close(name, mem)
            # Shared numpy
            elif data.nbytes > 0:
                # Memory map link
                if isinstance(data, np.memmap):
                    self.mems[name] = data
                    self.setdefault(name + 'mmap', list(str(data.filename)))  # Constant per spec/episode
                # Shared RAM memory
                else:
                    mem = self.setdefault(name, data)
                    mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf)
                    mem_[:] = data[:]
                    self.close(name, mem)
                    mem = self.setdefault(name + 'mmap', [0])  # False, no memory mapping. Also expects constant
                    self.close(name + 'mmap', mem)

            # Data shape
            mem = self.setdefault(name + 'shape', [1] if spec == 'id' else list(data.shape))  # Constant per spec/episode
            self.close(name + 'shape', mem)

    def __getitem__(self, key):
        # Account for potential delay
        for _ in range(2400):
            try:
                return self.get(key)
            except FileNotFoundError as e:
                sleep(1)
        raise(e)

    def get(self, key):
        num_episodes = key.stem.split('/')[-1].split('_')[1]

        episode = {}

        for spec in self.keys():
            name = self.dict_id + num_episodes + spec

            # Integer
            if spec == 'id':
                mem = self.getdefault(name, ShareableList)
                episode[spec] = int(mem[0])
                self.close(name, mem)
            # Numpy array
            else:
                # Shape
                mem = self.getdefault(name + 'shape', ShareableList)
                shape = tuple(mem)
                self.close(name + 'shape', mem)

                if 0 in shape:
                    episode[spec] = np.zeros(shape)  # Empty array
                else:
                    # Whether memory mapped
                    mem = self.getdefault(name + 'mmap', ShareableList)
                    is_mmap = list(mem)
                    self.close(name + 'mmap', mem)

                    if 0 in is_mmap:
                        mem = self.getdefault(name, SharedMemory)
                        # self.mems[name] = mem  # Caching all
                        # Without caching all replicas... need to copy. Otherwise, data somehow gets corrupted with nans
                        episode[spec] = np.ndarray(shape, np.float32, buffer=mem.buf)  # Copying
                        # if spec == 'obs':
                        #     print(episode[spec][0, 0, 10, :], '\n\n',
                        #           episode[spec][0, 0, 10, :].copy()
                        #           )
                        #     assert not np.isnan(episode[spec][0].any())
                        #     print(np.isnan(episode[spec][0].any()))
                        # episode[spec] = episode[spec][:200].copy()  # Index first
                        episode[spec] = episode[spec].copy()
                        self.close(name, mem)

                        # Can't copy/access after closing
                        # assert np.allclose(episode[spec], episode[spec].copy())
                    else:
                        # Read from memory-mapped hard disk file rather than shared RAM
                        episode[spec] = self.getdefault(name, lambda **_: np.memmap(''.join(is_mmap), np.float32,
                                                                                    'r+', shape=shape))
        return episode

    def setdefault(self, name, data):
        method = (ShareableList if isinstance(data, list) else SharedMemory)
        self.mems.update({name: method})  # For cleanup
        try:
            # Try to create shared memory link
            mem = method(data, name=name) if isinstance(data, list) \
                else method(create=True, name=name,  size=data.nbytes)
        except FileExistsError:
            # But if exists, retrieve shared memory link
            mem = method(name=name)
        return mem
        # return self.mems.pop(name)  # Remember to start_worker() only once

    def getdefault(self, name, method):
        # Return if cached, else evaluate
        # return self.mems[name] if name in self.mems else method(name=name)  # Caching all
        return method(name=name)

    def keys(self):
        return self.specs.keys() | {'id'}

    def __del__(self):
        self.cleanup()

    def start_worker(self):
        # Hacky fix for https://bugs.python.org/issue38119
        if not self.mems:
            check_rtype = lambda func: lambda name, rtype: None if rtype == 'shared_memory' else func(name, rtype)
            resource_tracker.register = check_rtype(resource_tracker.register)
            resource_tracker.unregister = check_rtype(resource_tracker.unregister)

            if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
                del resource_tracker._CLEANUP_FUNCS["shared_memory"]

    def close(self, name, mem):
        # if name not in self.mems:
        if isinstance(mem, ShareableList):
            mem = mem.shm
        mem.close()

    def cleanup(self):
        for name, method in self.mems.items():
            mem = self.getdefault(name, method)
            if not isinstance(mem, np.memmap):
                if isinstance(mem, ShareableList):
                    mem = mem.shm
                try:
                    mem.unlink()
                except FileNotFoundError:
                    continue