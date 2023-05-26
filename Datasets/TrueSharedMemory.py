# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
    1. Unbind
    2.  ̶I̶f̶ ̶t̶h̶e̶ ̶l̶a̶s̶t̶ ̶o̶n̶e̶ ̶w̶a̶s̶ ̶d̶o̶n̶e̶,̶ ̶a̶p̶p̶e̶n̶d̶
    3.  ̶I̶f̶ ̶t̶h̶e̶ ̶l̶a̶s̶t̶ ̶o̶n̶e̶ ̶w̶a̶s̶ ̶n̶o̶t̶ ̶d̶o̶n̶e̶ ̶o̶r̶ ̶t̶h̶i̶s̶ ̶o̶n̶e̶ ̶i̶s̶ ̶n̶o̶t̶ ̶d̶o̶n̶e̶,̶ ̶a̶n̶d̶ ̶n̶o̶ ̶t̶e̶m̶p̶o̶r̶a̶l̶_̶d̶i̶m̶,̶ ̶c̶r̶e̶a̶t̶e̶ ̶a̶ ̶t̶e̶m̶p̶o̶r̶a̶l̶ ̶d̶i̶m̶
    4.  ̶I̶f̶ ̶t̶h̶e̶ ̶l̶a̶s̶t̶ ̶o̶n̶e̶ ̶w̶a̶s̶ ̶n̶o̶t̶ ̶d̶o̶n̶e̶,̶ ̶c̶o̶n̶c̶a̶t̶e̶n̶a̶t̶e̶ ̶o̶n̶ ̶t̶e̶m̶p̶o̶r̶a̶l̶ ̶d̶i̶m̶
    2. If Done, append new list - Note tuple prob more efficient
        Else: set last list to last list + [exp]

    Note: Exps consist of batches of batch_size, for now assuming episode lengths uniform across each element

    self.episodes = [[Exp_1, Exp_2, ..., Exp_Done], ..., [Exp_A, Exp_B, ..., Exp_Zed]]
    self.index = zip([0, ..., batch_size_1 - 1, ..., 0, ..., batch_size_z - 1],
                     [episode_ind_1, ..., episode_ind_1, ..., episode_ind_z, ..., episode_ind_z])

    exp_ind, episode_ind = self.index(ind)
    episode = self.episodes[episode_ind][exp_ind]

    3. Episodes can be periodically concatenated along temporal_dim if specified by exp or stacked on new temporal_dim
        - If temporal_dim present, sampling intra-episode if rstep < inf requires prioritizing exps by sequence length
        - Sequences lengths batched arrays can be stored with episode for exp sample, then uniform time point sample
    4. Perhaps store per-element lengths in index if dones provided batch_wise, then include ID'd sub-batches in episode
    5. If IDs dict present, then: episode = self.episodes[episode_ind][IDs[exp_ind]] .
    6. Check capacities regarding device placement
    7. Dataset as argument: map(self.add, Dataset) where Dataset = load(dataset) supports replays, torchvision, etc
        - etc means for example custom datasets, like those defined in World/Datasets/__init__.py
        - checks if already exists in replay and if not, atomizes download -> mmap
        - World/ReplayBuffer stores Online and Offline Datasets.
            - Dataset=<offline>, stored as offline, online=True: is copied as online.
            - Dataset=<offline>, stored as online, online=True: is loaded as online.
            - Dataset=<offline>, stored as offline, offline=True, is loaded as offline.
            - Dataset=<offline>, stored as online, offline=True, is newly created as offline.
            - Dataset=<online>, stored as offline <- no such thing
            - Dataset=<online>, stored as online, online=True: is loaded as online.
            - Dataset=<online>, stored as offline <- no such thing
            - Dataset=<online>, stored as online, offline=True, is loaded as offline / no copy.
                - Or shared via Memory multi-task
            If a Dataset=<offline>, not stored, offline=True, make sure to mmap all of it.
        - ReplayBuffer stores .yaml cards describing the uniqueness / stats of the Dataset
            - Stats may be updated when needed; uniqueness includes command line / recipe flags via "+dataset.".
        - Functions are supported datums and are called at __getitem__ and are treated like mmaps. Dataset= can support.
            - Example usage: web links
    8. Dataset: Send indices when RAM shared memory and no Transform, else cached mmap or loaded mmap
    9. Collate fn collects indexed shared memories, sends all memories to device, and collates
    10. MMAPs save in ReplayBuffer and their paths can be loaded via Dataset=
    11. Online or offline, Dataset= works
    12. Images should be uint8? Labels int64? (Conserve dtype)

    Env: load (train=train), then Transform + DataLoader or Replay (stream=False).
    Replay: load (train=True) + Dataset with Memory + Transform + DataLoader with no pin_memory if device_capacity > 0.
    Memory can be passed to Samplers who can arrange index into bins, passed to collate who accelerates.
        - Actually, better if parallel workers arrange bins (Prioritizers / priority_fn).
        - Samplers can be dynamically built based on priority=.
        - Bins tell workers to periodically divide f(mem) into n bins or f=no-op. Each do so just for their portion,
            and store "bins" (lists) in a list assigned to them by Memory. Samplers treat these equally. Params: f, n.
        - Bins=Path.To.Bin can be list or individual. priority= tells the sampler which bin to sample based on. Sampler
            selects over worker clusters uniformly, then samples a bin uniformly, then samples again uniformly.
        - Memory can also have a Queue for rewrites and dict for toggling trajectories.

    1. .to(memory_format=torch.channels_last) if B, C, H, W format (Encoder with "image" modality")
    2. Don't forget to jit element-wise operations
    3. GPU-vectorized augmentations
    """
import os
import time

import numpy as np

import torch
import torch.multiprocessing as mp

from tensordict.memmap import MemmapTensor


def f1(a):
    while True:
        _start = time.time()
        print(a[0][0]['hi'][0, 0, 0].item(), time.time() - _start, 'get', len(a))
        time.sleep(3)


def f2(a):
    while True:
        _start = time.time()
        print(a[0][0]['hi'][0, 0, 0].item(), time.time() - _start, 'get', len(a))
        time.sleep(3)


class Memory:
    def __init__(self, device=None, device_capacity=None, ram_capacity=None, cache_capacity=None, hd_capacity=None):
        self.path = os.getcwd() + '/ReplayBuffer'

        # if not os.path.exists(self.path):
        #     os.mkdir(self.path)

        manager = mp.Manager()

        self.index = manager.list()
        self.episode_batches = manager.list()

    def add(self, batch):
        if batch['done'] or not self.episode_batches:
            self.episode_batches.append(())

        for key in batch:
            batch[key] = torch.as_tensor(batch[key]).share_memory_()  # .to(non_blocking=True)

        # for key in exp:
        #     # Just creates np; maybe saves some torch-accessed variables in _init_shape like _device, _shape, _dtype
        #     exp[key] = MemmapTensor.from_tensor(torch.as_tensor(exp[key]),
        #                                         filename=f'{self.path}/'
        #                                                  f'{len(self.index)}_{key}.dat')  # .to(non_blocking=True)

        self.episode_batches[-1] = self.episode_batches[-1] + (batch,)

        batch_size = 1

        for mem in batch.values():
            if mem.shape and len(mem) > 1:
                batch_size = len(mem)
                break

        episode_batches_ind = len(self.episode_batches) - 1

        self.index.extend(enumerate([episode_batches_ind] * batch_size))

    def __getitem__(self, ind):
        ind, episode_batches_ind = self.index[ind]
        batches = self.episode_batches[episode_batches_ind]

        return Episode(batches, ind)  # Return list of exps in episode

    def __len__(self):
        return len(self.index)


# Just make a single uniform Mem class for storing batch-datums within exps
# Cache independent. Just store as tensor. Then call .share(), .mmap(), .gpu(), .tensor().
# Episodes consist of Batches consist of dicts consist of Sub-Batches (experiences) consist of dicts
# Batches consist of Mems consist of datums
# Sub-Batches consist of Mems consist of datums
# Mems consist of Sub-Mems consist of datums
# Episode can store .__add__ method
class MMAP:
    def __init__(self, datum, path=None):
        if isinstance(datum, np.memmap):
            self.datum = datum
        else:
            datum = torch.as_tensor(datum).numpy()

    def __getitem__(self, item):
        return MMAP(item)

    def __setitem__(self, key, value):
        pass

    def as_tensor(self):
        pass


class Episode:
    def __init__(self, batches, ind):
        self.batches = batches
        self.ind = ind

    def __getitem__(self, step):
        return Experience(self.batches, step, self.ind)


class Experience:
    def __init__(self, batches, step, ind):
        self.batches = batches
        self.step = step
        self.ind = ind

    def __getitem__(self, key):
        return self.batches[self.step][key][self.ind]

    def __setitem__(self, key, value):
        self.batches[self.step][key][self.ind] = value




if __name__ == '__main__':
    m = Memory()
    d1 = {'hi': np.ones([2560, 3, 32, 32]), 'done': [False]}
    d2 = {'hi': np.ones([2560, 3, 32, 32]), 'done': [True]}
    start = time.time()
    m.add(d1)
    m.add(d2)
    print(time.time() - start, 'add')
    p1 = mp.Process(name='p1', target=f1, args=(m,))
    p2 = mp.Process(name='p', target=f2, args=(m,))
    p1.start()
    p2.start()
    start = time.time()
    # e = m[0]
    # e[0]['hi'] = 5
    print(time.time() - start, 'set')
    p1.join()
    p2.join()


# Collate GPU:

# import torch
# import torchvision
#
# def collate_gpu(batch):
#     x, t = torch.utils.data.dataloader.default_collate(batch)
#     return x.to(device="cuda:0"), t.to(device="cuda:0")
#
# train_dataset = torchvision.datasets.MNIST(
#     './data',
#     train=True,
#     download=True,
#     transform=torchvision.transforms.ToTensor(),
# )
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=1,
#     prefetch_factor=2,
#     persistent_workers=True,
#     collate_fn=collate_gpu,
# )
#
# if __name__ == "__main__":
#     x, t = next(iter(train_loader))
#     print(type(x), x.device, type(t), t.device)

# Or all the way https://github.com/ste362/WrappedDataloader

# Maybe some can be adaptively sent to GPU - in Dataset - before collate (and live there) - then collate can send rest


import collections
import contextlib
import re

from typing import Callable, Dict, Optional, Tuple, Type, Union

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.as_tensor(batch)


def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn


def default_collate(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map)

    # Ignore this:
    # Maybe can unbind exps, no index, then episode index can index to these (per datum key)
    # with zipped sequences of indices
    # Can sequentiate all datums in a list (tape) and add corresponding indexes to higher-level constructs
    # Datums can be mmap objects, shared tensors, ints, or links -- maybe ints can be part of index and not indexed
    # Str names can also be part of index
    # Done must be global or per batch item

    # Since shared memory, maybe no cost to concat and restore in manager - same as indices
    # So just replace manager index with concatenated episode; maybe tensordict for quick concat/mmap or unbind all

    # For batch-level not-dones may need extra considerations about efficiency of concatenating after unbinding

    # Episodes consist of batch-exps (AttrDicts/Batches) consist of dicts consist of Mems consist of exps
