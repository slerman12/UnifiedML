# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re
import warnings
from pathlib import Path

from hydra.utils import instantiate

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# # Saves model
# def save(path, model):
#     path = path.replace('Agents.', '')
#     Path('/'.join(path.split('/')[:-1])).mkdir(exist_ok=True, parents=True)
#     torch.save(model, path)
#
#
# # Loads model
# def load(path, device, attr=None, persevere=False):
#     path = path.replace('Agents.', '')
#
#     try:
#         model = torch.load(path)
#     except Exception as e:  # Pytorch's load and save are not atomic transactions
#         if persevere:
#             warnings.warn(f'Load conflict, resolving...')  # For distributed training
#             time.sleep(0.1)
#             return load(path, device, attr, True)
#         else:
#             assert Path(path).exists(), f'Load path {path} does not exist.'
#             raise Exception(e, '\nTry calling load with persevere=True to recursively try loading until resolution; '
#                                'this is a hacky way to make distributed training work')
#
#     if attr is not None:
#         for attr in attr.split('.'):
#             model = getattr(model, attr)
#
#     return model.to(device)


# Saves agent + attributes
def save(path, agent, *attributes):
    path = path.replace('Agents.', '')
    Path('/'.join(path.split('/')[:-1])).mkdir(exist_ok=True, parents=True)
    to_save = {'state_dict': agent.state_dict()}
    to_save.update({attr: getattr(agent, attr) for attr in attributes})
    torch.save(to_save, path)


# Loads agent or part of agent
def load(path, agent=None, device='cuda', attr=None):
    Agent = 'Agents.' + path.split('Agents.')[1].split('Agent')[0] + 'Agent'  # e.g. Agents.DQNAgent
    path = path.replace('Agents.', '')

    while True:
        try:
            if Path(path).exists():
                # Load agent
                to_load = torch.load(path, map_location=getattr(agent, 'device', device))
                if agent is None:
                    # Instantiate a new agent
                    agent = instantiate(Agent).to(device)
                else:
                    # Load agent's params
                    agent.load_state_dict(to_load['state_dict'], strict=False)
                del to_load['state_dict']
                # Update its saved attributes
                for key in to_load:
                    if hasattr(agent, key):
                        setattr(agent, key, to_load[key])
            else:
                assert agent is not None, f'Load path {path} does not exist.'
                warnings.warn(f'Load path {path} does not exist. Proceeding without loading.')
            break
        except Exception as e:  # For distributed training: Pytorch's load and save are not atomic transactions
            assert agent is not None, f'{e}\nCould not load agent.'
            # Catch conflict, try again
            warnings.warn(f'Load conflict, resolving...')

    # Can also load part of an agent, e.g. its encoder.
    # This method can be used as a recipe to pass in saved checkpoint components
    # e.g. python Run.py Eyes=Utils.Load +recipes.encoder.eyes.path=<checkpoint> +recipes.encoder.eyes.attr=encoder.Eyes
    if attr is not None:
        for attr in attr.split('.'):
            agent = getattr(agent, attr)

    return agent


# Assigns a default value to x if x is None
def default(x, value):
    if x is None:
        x = value
    return x


# Initializes model weights according to orthogonality
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


# Copies parameters from one model to another, with optional EMA weighing
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
        elif isinstance(block, nn.Flatten) and block.start_dim == -3:
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

    feature_shape = (channels, height, width)  # TODO should probably do (channels, width, height) universally

    return feature_shape


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x):
        return torch.stack([m(*x) for m in self.ensemble],
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
def one_hot(x, num_classes):
    # assert x.shape[-1] == 1
    x = x.squeeze(-1).unsqueeze(-1)  # Or this
    x = x.long()
    shape = x.shape[:-1]
    zeros = torch.zeros(*shape, num_classes, dtype=x.dtype, device=x.device)
    return zeros.scatter(len(shape), x, 1).float()


# Differentiable one_hot
def rone_hot(x, null_value=0):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]) * (1 - null_value) + null_value)


# Differentiable clamp
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


ChSwap = ChannelSwap()


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


# Converts data to Torch Tensors and moves them to the specified device as floats
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers
def optimize(loss=None, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward(retain_graph=retain_graph)

    # Optimize
    if step_optim:
        for model in models:
            model.optim.step()

            # Update ema target
            if ema and hasattr(model, 'ema'):
                model.update_ema_params()

            if loss is None and clear_grads:
                model.optim.zero_grad(set_to_none=True)


# Increment/decrement a value in proportion to a step count based on a string-formatted schedule
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * start + mix * stop

