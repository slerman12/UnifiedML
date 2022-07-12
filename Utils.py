# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re
import warnings
from inspect import signature
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Blocks.Architectures import *  # For direct accessibility via command line


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Initializes run state
def init(args):
    # Set seeds
    set_seeds(args.seed)

    # Set device
    mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup
    args.device = args.device or ('cuda' if torch.cuda.is_available()
                                  else 'mps' if mps and mps.is_available() else 'cpu')
    # args.device = args.device or ('cuda' if torch.cuda.is_available()
    #                               else 'mps' if torch.backends.mps.is_available() else 'cpu')


# Format path names
# e.g. Checkpoints/Agents.DQNAgent -> Checkpoints/DQNAgent
OmegaConf.register_new_resolver("format", lambda name: name.split('.')[-1])


# Saves model + args + attributes
def save(path, model, args, *attributes):
    root, name = path.rsplit('/', 1)
    Path(root).mkdir(exist_ok=True, parents=True)
    torch.save(args, Path(root) / f'{name.replace(".pt", "")}.args')  # Save args

    torch.save({'state_dict': model.state_dict(), **{attr: getattr(model, attr)
                                                     for attr in attributes}}, path)  # Save params + attributes
    print(f'Model successfully saved to {path}')


# Saves args, accounting for special case when 'load' in recipe
# def save_args(args, root_path, name, attr=None):
#     if attr is None:
#         save_args(args.recipes, root_path, name, '')
#         torch.save(args, root_path / f'{name}.args')
#     # Accounts for loading from one's own self or recursions between cross-experiment loads
#     else:
#         # If sub-part loaded via recipes (e.g. Eyes=load ...), saves original recipe (Eyes=...)
#         for key, recipe in args.items():
#             _target_ = getattr(recipe, '_target_', recipe)
#             attrs = f'{attr}.{key}'.strip('.')  # The recipe's attributes e.g. encoder.eyes
#
#             # Recursion to sub-recipes
#             if isinstance(_target_, DictConfig):
#                 save_args(_target_, root_path, name, attrs)
#             # Any recipe that calls 'load'
#             elif _target_ and ('load(' in _target_ or _target_[:4] == 'load' or _target_[:9] == 'Utils.load'):
#                 # Interpolate a load path
#                 load_path = getattr(recipe, 'path',
#                                     _target_.split('load(')[-1].split('path=')[-1].split(')')[0].split(',')[0])
#
#                 load_root, load_name = load_path.replace('.pt', '').rsplit('/', 1)
#
#                 # Can add:
#                 # if load_root == root_path and not (sub_parts_path / f'{load_name}_{attrs}.args').exists()
#                       # ... and delete below - but still important if loading from others
#
#                 sub_parts_path = Path(load_root) / 'sub_parts'
#                 sub_parts_path.mkdir(exist_ok=True)
#
#                 # If not already cached
#                 if not (sub_parts_path / f'{load_name}_{attrs}.args').exists():
#                     # Save a copy of the dependency args
#                     shutil.copy(f'{load_root}/{load_name}.args', sub_parts_path / f'{load_name}_{attrs}.args')
#             else:
#                 # No need to independently save others - [might be important to others but maybe necessary for update]
#                 Path(root_path / f'sub_parts/{name}_{attrs}.args').unlink(missing_ok=True)


# Loads model or part of model
def load(path, device='cuda', args=None, preserve=(), distributed=False, attr='', **kwargs):
    root, name = path.replace('.pt', '').rsplit('/', 1)

    while True:
        try:
            to_load = torch.load(path, map_location=device)  # Load
            args = args or DictConfig({})
            args.update(torch.load(f'{root}/{name}.args'))  # Load
            break
        except Exception as e:  # Pytorch's load and save are not atomic transactions, can conflict in distributed setup
            if not distributed:
                raise RuntimeError(e)
            warnings.warn(f'Load conflict, resolving...')  # For distributed training

    model = instantiate(args).to(device)

    # Load model's params
    model.load_state_dict(to_load['state_dict'], strict=False)

    # Load saved attributes as well
    for key in to_load:
        if hasattr(model, key) and key not in ['state_dict', *preserve]:
            setattr(model, key, to_load[key])

    # Can also load part of a model. Useful for recipes,
    # e.g. python Run.py Eyes=load +eyes.path=<Path To Agent Checkpoint> +eyes.attr=encoder.Eyes +eyes.device=<device>
    for key in attr.split('.'):
        if key:
            model = getattr(model, key)
    print(f'Successfully loaded {attr if attr else "agent"} from {path}')
    return model


# Simple-sophisticated instantiation of a class or module by various semantics
def instantiate(args, i=0, **kwargs):
    if hasattr(args, '_target_') and args._target_:
        try:
            return hydra.utils.instantiate(args, **kwargs)  # Regular hydra
        except ImportError as e:
            if '(' in args._target_ and ')' in args._target_:  # Direct code execution
                args = args._target_
            else:
                args._target_ = 'Utils.' + args._target_  # Portal into Utils
                try:
                    return hydra.utils.instantiate(args, **kwargs)
                except ImportError:
                    raise e  # Original error if all that doesn't work

    if isinstance(args, str):
        for key in kwargs:
            args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
        args = eval(args)  # Direct code execution

    return None if hasattr(args, '_target_') \
        else args(**{key: kwargs[key]
                     for key in kwargs.keys() & signature(args).parameters}) if isinstance(args, type) \
        else args[i] if isinstance(args, list) \
        else args  # Additional useful ones


# Initializes model weights a la orthogonal
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Initializes model optimizer. Default: AdamW + cosine annealing
def optimizer_init(params, optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None):
    params = list(params)
    has_params = len(params) > 0

    # Optimizer
    optim = has_params and (instantiate(optim, params=params, lr=getattr(optim, 'lr', lr))
                            or lr and torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay))  # Default

    # Learning rate scheduler
    scheduler = has_params and (instantiate(scheduler, optimizer=optim) or (lr and lr_decay_epochs or None)
                                and torch.optim.lr_scheduler.CosineAnnealingLR(optim, lr_decay_epochs))  # Default

    return optim, scheduler


# Copies parameters from one model to another, optionally EMA weighing
def param_copy(model, target, ema_decay=0):
    with torch.no_grad():
        for target_param, model_param in zip(target.state_dict().values(), model.state_dict().values()):
            target_param.copy_(ema_decay * target_param + (1 - ema_decay) * model_param)


# Compute the output shape of a CNN layer
def cnn_layer_feature_shape(in_height, in_width, kernel_size=1, stride=1, padding=0, dilation=1):
    if padding == 'same':
        return in_height, in_width
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding)
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
    out_height = math.floor(((in_height + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    out_width = math.floor(((in_width + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return out_height, out_width


# Compute the output shape of a whole CNN
def cnn_feature_shape(channels, height, width, *blocks, verbose=False):
    for block in blocks:
        if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)):
            channels = block.out_channels if hasattr(block, 'out_channels') else channels
            height, width = cnn_layer_feature_shape(height, width,
                                                    kernel_size=block.kernel_size,
                                                    stride=block.stride,
                                                    padding=block.padding)
        elif isinstance(block, nn.Linear):
            channels = block.out_features  # Assumes channels-last if linear
        elif isinstance(block, nn.Flatten) and (block.start_dim == -3 or block.start_dim == 1):
            channels, height, width = channels * height * width, 1, 1  # Placeholder height/width dims
        elif isinstance(block, nn.AdaptiveAvgPool2d):
            height, width = block.output_size
        elif hasattr(block, 'repr_shape'):
            channels, height, width = block.repr_shape(channels, height, width)
        elif hasattr(block, 'modules'):
            for layer in block.children():
                channels, height, width = cnn_feature_shape(channels, height, width, layer, verbose=verbose)
        if verbose:
            print(block, (channels, height, width))

    feature_shape = (channels, height, width)

    return feature_shape


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x, **kwargs):
        return torch.stack([m(*x, **kwargs) for m in self.ensemble],
                           self.dim)


# Replaces tensor's batch items with Normal-sampled random latent
class Rand(nn.Module):
    def __init__(self, size=1, uniform=False):
        super().__init__()
        self.size = size
        self.uniform = uniform

    def forward(self, x):
        x = torch.randn((x.shape[0], self.size), device=x.device)
        return x.uniform_() if self.uniform else x


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes, null_value=0):
    # assert x.shape[-1] == 1  # Can check this
    x = x.squeeze(-1).unsqueeze(-1)  # Or do this
    x = x.long()
    shape = x.shape[:-1]
    nulls = torch.full([*shape, num_classes], null_value, dtype=x.dtype, device=x.device)
    return nulls.scatter(len(shape), x, 1).float()


# Differentiable one_hot via "re-parameterization"
def rone_hot(x, null_value=0):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]) * (1 - null_value) + null_value)


# Differentiable clamp via "re-parameterization"
def rclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max))


# (Multi-dim) indexing
def gather_indices(item, ind, dim=-1):
    ind = ind.long().expand(*item.shape[:dim], ind.shape[-1])  # Assumes ind.shape[-1] is desired num indices
    if dim < len(item.shape) - 1 and dim != -1:
        trail_shape = item.shape[dim + 1:]
        ind = ind.reshape(ind.shape + (1,)*len(trail_shape))
        ind = ind.expand(*ind.shape[:dim + 1], *trail_shape)
    return torch.gather(item, dim, ind)


# Basic L2 normalization
class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps)


# Min-max normalizes to [0, 1]
# "Re-normalization", as in SPR (https://arxiv.org/abs/2007.05929), or at least in their code
class ShiftMaxNorm(nn.Module):
    def __init__(self, start_dim=-1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        y = x.flatten(self.start_dim)
        y = y - y.min(-1, keepdim=True)[0]
        y = y / y.max(-1, keepdim=True)[0]
        return y.view(*x.shape)


# Swaps image dims between channel-last and channel-first format
class ChannelSwap(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -3)


ChSwap = ChannelSwap()  # Convenient helper


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()
        self.models = models

    def __enter__(self):
        self.start_modes = []
        for model in self.models:
            if model is None:
                self.start_modes.append(None)
            else:
                self.start_modes.append(model.training)
                model.eval()

    def __exit__(self, *args):
        for model, mode in zip(self.models, self.start_modes):
            if model is not None:
                model.train(mode)
        return False


# Converts data to torch Tensors and moves them to the specified device as floats
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Stack batches of frame trajectories along channel axis
# def frame_stack(frames, stack_size=1, start=None, end=None):
#     if start is None:
#         start = stack_size
#
#     if end is None:
#         end = start + 1
#
#     if end < 0:
#         end = frames.shape[1] + end
#
#     if end == start + 1:
#         obs = frames[:, max([0, end - stack_size]):end]  # Temporal axis
#         for _ in range(stack_size - end):
#             obs = np.concatenate([frames[:, 0], obs], 1)  # Temporal axis
#         obs = frames.reshape(obs.shape[0], -1, obs.shape[3:])  # Channel axis
#     else:
#         obs = np.concatenate([frames[:, max([0, start - i]):max([1, end - i])]  # Temporal axis
#                               for i in range(stack_size - 1, -1, -1)], axis=2)  # Channel axis
#
#     return obs


# Pytorch incorrect (in this case) warning suppression
warnings.filterwarnings("ignore", message='.* skipping the first value of the learning rate schedule')


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers and schedulers
def optimize(loss, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, epoch=0, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            if model.optim:
                model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward(retain_graph=retain_graph)

    # Optimize
    if step_optim:
        for model in models:
            # Step scheduler
            if model.scheduler and epoch > model.scheduler.last_epoch:
                model.scheduler.step()
                model.scheduler.last_epoch = epoch

            # Update ema target
            if ema and hasattr(model, 'ema'):
                model.update_ema_params()

            if model.optim:
                model.optim.step()

                if loss is None and clear_grads:
                    model.optim.zero_grad(set_to_none=True)


# Increment/decrement a value in proportion to a step count and a string-formatted schedule
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * start + mix * stop
