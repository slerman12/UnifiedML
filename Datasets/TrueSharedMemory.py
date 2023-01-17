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


# # TODO allow indexing idx _from, _to before of copy!
# class SharedDict:
#     """
#     An Offline dict generalized to manage numpy arrays, integers, and hard disk memory map-links in truly-shared RAM
#     memory, rather than serialization/deserialization. Efficiently read-writes data directly across parallel CPU workers
#     """
#     def __init__(self, specs):
#         self.dict_id = str(uuid.uuid4())[:8]
#
#         self.mems = {}
#         self.specs = specs
#
#         # Shared memory can create a lot of file descriptors
#         _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
#
#         # Increase soft limit to hard limit just in case
#         resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
#
#     def __setitem__(self, key, value):
#         self.start_worker()
#
#         assert isinstance(value, dict), 'Shared Memory must be dict'
#
#         num_episodes = key.stem.split('/')[-1].split('_')[1]
#
#         for spec, data in value.items():
#             name = self.dict_id + num_episodes + spec
#
#             # Shared integers
#             if spec == 'id':
#                 mem = self.setdefault(name, [data])
#                 mem[0] = data
#                 self.close(name, mem)
#             # Shared numpy
#             elif data.nbytes > 0:
#                 # Memory map link
#                 if isinstance(data, np.memmap):
#                     self.mems[name] = data
#                     self.setdefault(name + 'mmap', list(str(data.filename)))  # Constant per spec/episode
#                 # Shared RAM memory
#                 else:
#                     mem = self.setdefault(name, data)
#                     mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf)
#                     mem_[:] = data[:]
#                     self.close(name, mem)
#                     mem = self.setdefault(name + 'mmap', [0])  # False, no memory mapping. Also expects constant
#                     self.close(name + 'mmap', mem)
#
#             # Data shape
#             mem = self.setdefault(name + 'shape', [1] if spec == 'id' else list(data.shape))  # Constant per spec/episode
#             self.close(name + 'shape', mem)
#
#     def __getitem__(self, key):
#         # Account for potential delay
#         for _ in range(2400):
#             try:
#                 return self.get(key)
#             except FileNotFoundError as e:
#                 sleep(1)
#         raise(e)
#
#     def get(self, key):
#         num_episodes = key.stem.split('/')[-1].split('_')[1]
#
#         episode = {}
#
#         for spec in self.keys():
#             name = self.dict_id + num_episodes + spec
#
#             # Integer
#             if spec == 'id':
#                 mem = self.getdefault(name, ShareableList)
#                 episode[spec] = int(mem[0])
#                 self.close(name, mem)
#             # Numpy array
#             else:
#                 # Shape
#                 mem = self.getdefault(name + 'shape', ShareableList)
#                 shape = tuple(mem)
#                 self.close(name + 'shape', mem)
#
#                 if 0 in shape:
#                     episode[spec] = np.zeros(shape)  # Empty array
#                 else:
#                     # Whether memory mapped
#                     mem = self.getdefault(name + 'mmap', ShareableList)
#                     is_mmap = list(mem)
#                     self.close(name + 'mmap', mem)
#
#                     if 0 in is_mmap:
#                         mem = self.getdefault(name, SharedMemory)
#                         # self.mems[name] = mem  # Caching all
#                         # Without caching all replicas... need to copy. Otherwise, data somehow gets corrupted with nans
#                         episode[spec] = np.ndarray(shape, np.float32, buffer=mem.buf)  # Copying
#                         # if spec == 'obs':
#                         #     print(episode[spec][0, 0, 10, :], '\n\n',
#                         #           episode[spec][0, 0, 10, :].copy()
#                         #           )
#                         #     assert not np.isnan(episode[spec][0].any())
#                         #     print(np.isnan(episode[spec][0].any()))
#                         # episode[spec] = episode[spec][:200].copy()  # Index first
#                         episode[spec] = episode[spec].copy()
#                         self.close(name, mem)
#
#                         # Can't copy/access after closing
#                         # assert np.allclose(episode[spec], episode[spec].copy())
#                     else:
#                         # Read from memory-mapped hard disk file rather than shared RAM
#                         episode[spec] = self.getdefault(name, lambda **_: np.memmap(''.join(is_mmap), np.float32,
#                                                                                     'r+', shape=shape))
#         return episode
#
#     def setdefault(self, name, data):
#         method = (ShareableList if isinstance(data, list) else SharedMemory)
#         self.mems.update({name: method})  # For cleanup
#         try:
#             # Try to create shared memory link
#             mem = method(data, name=name) if isinstance(data, list) \
#                 else method(create=True, name=name,  size=data.nbytes)
#         except FileExistsError:
#             # But if exists, retrieve shared memory link
#             mem = method(name=name)
#         return mem
#         # return self.mems.pop(name)  # Remember to start_worker() only once
#
#     def getdefault(self, name, method):
#         # Return if cached, else evaluate
#         # return self.mems[name] if name in self.mems else method(name=name)  # Caching all
#         return method(name=name)
#
#     def keys(self):
#         return self.specs.keys() | {'id'}
#
#     def __del__(self):
#         self.cleanup()
#
#     def start_worker(self):
#         # Hacky fix for https://bugs.python.org/issue38119
#         if not self.mems:
#             check_rtype = lambda func: lambda name, rtype: None if rtype == 'shared_memory' else func(name, rtype)
#             resource_tracker.register = check_rtype(resource_tracker.register)
#             resource_tracker.unregister = check_rtype(resource_tracker.unregister)
#
#             if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
#                 del resource_tracker._CLEANUP_FUNCS["shared_memory"]
#
#     def close(self, name, mem):
#         # if name not in self.mems:
#         if isinstance(mem, ShareableList):
#             mem = mem.shm
#         mem.close()
#
#     def cleanup(self):
#         for name, method in self.mems.items():
#             mem = self.getdefault(name, method)
#             if not isinstance(mem, np.memmap):
#                 if isinstance(mem, ShareableList):
#                     mem = mem.shm
#                 try:
#                     mem.unlink()
#                 except FileNotFoundError:
#                     continue
#
#
# class SharedDict2:
#     """
#     An Offline dict generalized to manage numpy arrays, integers, and hard disk memory map-links in truly-shared RAM
#     memory, rather than serialization/deserialization. Efficiently read-writes data directly across parallel CPU workers
#     """
#     def __init__(self, specs):
#         self.dict_id = str(uuid.uuid4())[:8]
#
#         self.mems = {}
#         self.specs = specs
#
#         # Shared memory can create a lot of file descriptors
#         _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
#
#         # Increase soft limit to hard limit just in case
#         resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
#
#     def __setitem__(self, key, value):
#         # self.start_worker()
#
#         # Recurse to support dicts
#         if isinstance(value, dict):
#             self[key] = SharedDict(value.keys())
#             self.update(value)
#         else:
#             self[key] = Mem(key, value)
#
#     def __getitem__(self, key):
#         num_episodes = key.stem.split('/')[-1].split('_')[1]
#
#         name = self.dict_id + num_episodes + spec
#
#         if key not in self:
#             self[key] = None
#
#         return self[key]
#
#     def keys(self):
#         return self.specs.keys() | {'id'}
#
#     # def start_worker(self):
#     #     # Hacky fix for https://bugs.python.org/issue38119
#     #     if not self.mems:
#     #         check_rtype = lambda func: lambda name, rtype: None if rtype == 'shared_memory' else func(name, rtype)
#     #         resource_tracker.register = check_rtype(resource_tracker.register)
#     #         resource_tracker.unregister = check_rtype(resource_tracker.unregister)
#     #
#     #         if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
#     #             del resource_tracker._CLEANUP_FUNCS["shared_memory"]
#
#
# class Mem:
#     """
#     Truly-shared RAM and memory-mapped hard disk data-item usable across parallel CPU workers.
#     """
#     def __init__(self, name, value=None):
#         self.mem = None
#         self.name = name  # The only identifying information the CPU worker needs
#         self.created = {}
#
#         if value is not None:
#             self.assign(value)  # Set the truly shared value
#
#     def assign(self, data):
#         mmap = isinstance(data, np.memmap)  # Whether to memory map
#
#         # Shared integers  # TODO Memory-map scalars too. Currently retains single scalars in RAM.
#         if np.isscalar(data):
#             self.mem = self.create([data])
#             self.mem[0] = data
#             self.close()
#         # Shared numpy
#         elif data.nbytes > 0:
#             # Hard disk memory-map link
#             if mmap:
#                 self.mem = self.create(list(str(data.filename)), 'mmap')
#                 self.close()
#             # Shared RAM memory
#             else:
#                 self.mem = self.create(self.name, data)  # Create
#                 mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=self.mem.buf)
#                 mem_[:] = data[:]  # Set data
#                 self.close()
#                 self.mem = self.create([0], 'mmap')  # Set to False, no memory mapping
#                 self.close()
#
#         # Data shape
#         self.mem = self.create([1] if np.isscalar(data) else list(data.shape), 'shape')
#         self.close()
#
#     def create(self, data, key=''):
#         # Two ways to create a truly shared RAM memory link in Python
#         method = (ShareableList if isinstance(data, list) else SharedMemory)
#
#         # Try to create shared memory link
#         try:
#             mem = method(data, name=self.name + key) if isinstance(data, list) \
#                 else method(create=True, name=self.name + key,  size=data.nbytes)
#         except FileExistsError:
#             mem = method(name=self.name + key)  # But if exists, retrieve existing shared memory link
#
#         self.created.update({self.name + key: method})
#
#         return mem
#
#     def __getitem__(self, idx):
#         # Account for potential delay
#         for _ in range(2400):
#             try:
#                 return self._getitem(idx)
#             except FileNotFoundError as e:
#                 sleep(1)
#         raise(e)
#
#     def _getitem(self, idx=None):
#         # Shape of data
#         self.mem = ShareableList(name=self.name + 'shape')
#         shape = tuple(self.mem)
#         self.close()
#
#         # Integer
#         if shape == (1,):  # Can be accessed with a None idx
#             self.mem = ShareableList(name=self.name)
#             value = int(self.mem[0])
#             self.close()
#         # Numpy array
#         else:
#             if 0 in shape:
#                 value = np.zeros(shape)  # Empty array
#             else:
#                 # Whether memory mapped
#                 self.mem = ShareableList(name=self.name + 'mmap')
#                 is_mmap = list(self.mem)
#                 self.close()
#
#                 if 0 in is_mmap:
#                     # Truly-shared RAM access
#                     self.mem = SharedMemory(name=self.name)
#                     value = np.ndarray(shape, np.float32, buffer=self.mem.buf)
#                     value = value[idx].copy()
#                     self.close()
#                 else:
#                     # Read from memory-mapped hard disk file rather than shared RAM
#                     value = np.memmap(''.join(is_mmap), np.float32, 'r+', shape=shape)[idx]
#
#         return value
#
#     def close(self, unlink=False):
#         mem = self.mem
#         if isinstance(self.mem, ShareableList):
#             mem = self.mem.shm
#         mem.close()
#
#         if unlink:
#             mem.unlink()
#
#     def __len__(self):
#         self.mem = ShareableList(name=self.name + 'shape')
#         size, *_ = self.mem
#         self.close()
#         return size
#
#     def __del__(self):
#         for name, method in self.created.items():
#             self.mem = method(name=name)
#
#             self.close(unlink=True)  # Unlink all created shared memory instances


# https://stackoverflow.com/questions/61879811/why-can-multiple-list-indexes-be-used-with-getitem-but-not-setitem


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
        # Two ways to create a truly shared RAM memory link in Python
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
            mem.unlink()


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

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)
            value = value[idx].copy()

            mem.close()

            return value
        else:
            # Read from memory-mapped hard disk file rather than shared RAM
            return np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)

    def __setitem__(self, idx, value):
        if self.mmap_name is None:
            # Shared RAM memory
            mem = SharedMemory(name=self.name)

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)  # Un-mutable shape
            value[idx] = value

            mem.close()
        else:
            # Hard disk memory-map link
            mem = np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)
            mem[idx] = value
            mem.flush()  # Write to hard disk

    def __len__(self):
        mem = ShareableList(name=self.name + 'shape')
        size = mem[0]
        mem.shm.close()
        return size
