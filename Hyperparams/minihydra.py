# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
minihydra / leviathan
A lightweight sys-arg manager, implemented from scratch by Sam Lerman.
See full hydra here: https://github.com/facebookresearch/hydra
"""

import ast
import importlib.util
import inspect
import re
import sys
from copy import deepcopy
from functools import partial
from math import inf
import yaml


app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])

# minihydra.yaml_search_paths.append(path)
yaml_search_paths = [app]  # List of paths


# Something like this
def instantiate(args, **kwargs):
    # For compatibility with old Hydra syntax
    if '_recursive_' in args:
        args.pop('_recursive_')

    args = deepcopy(args)
    file, module = args.pop('_target_').rsplit('.', 1)
    file = file.replace('.', '/')
    spec = importlib.util.spec_from_file_location(module, file + '.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module] = foo
    spec.loader.exec_module(foo)
    args.update(kwargs)
    return getattr(foo, module)(**args)


def open_yaml(source):
    for path in yaml_search_paths:
        try:
            with open(path + '/' + source, 'r') as file:
                args = yaml.safe_load(file)
            return recursive_Args(args)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(source, 'not found.')


class Args(dict):
    def __init__(self, _dict=None, **kwargs):
        super().__init__()
        self.__dict__ = self  # Allows access via attributes
        self.update({**(_dict or {}), **kwargs})


# Allow access via attributes recursively
def recursive_Args(args):
    if isinstance(args, dict):
        args = Args(args)

    items = enumerate(args) if isinstance(args, (list, tuple)) \
        else args.items() if isinstance(args, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        args[key] = recursive_Args(value)  # Recurse through inner values

    return args


def recursive_update(args, args2):
    for key, value in args2.items():
        args[key] = recursive_update(args.get(key, {}), value) if isinstance(value, type(args)) else value
    return args


def read(source):
    args = open_yaml(source)

    # Need to allow imports
    if 'imports' in args:
        imports = args.pop('imports')

        for module in imports:
            module = args if module == 'self' else open_yaml(module + '.yaml')
            recursive_update(args, module)

    # Command-line import
    if 'task' in args:
        recursive_update(args, open_yaml(args.task + '.yaml'))

    return args


def parse(args=None):
    # Parse command-line
    for sys_arg in sys.argv[1:]:
        arg = args
        keys, value = sys_arg.split('=', 1)
        keys = keys.split('.')
        for key in keys[:-1]:
            if key not in arg:
                setattr(arg, key, Args())
            arg = getattr(arg, key)
        setattr(arg, keys[-1], value)
        if re.compile(r'^\[.*\]$').match(value) or re.compile(r'^\{.*\}$').match(value) or \
                re.compile(r'^-?[0-9]*.?[0-9]+$').match(value):
            arg[keys[-1]] = ast.literal_eval(value)
        elif isinstance(value, str) and value.lower() in ['true', 'false', 'null', 'inf']:
            arg[keys[-1]] = True if value.lower() == 'true' else False if value.lower() == 'false' \
                else None if value.lower() == 'null' else inf
    return args


def get(args, keys):
    keys = keys.split('.')
    for key in keys:
        args = getattr(args, key)
    return interpolate(args)  # Interpolate to make sure gotten value is resolved  TODO Still resolves old value


# minihydra.grammar.append(rule)
grammar = []  # List of funcs


def interpolate(args):
    def _interpolate(match_obj):
        if match_obj.group() is not None:
            return str(get(args, match_obj.group()[2:][:-1]))

    items = enumerate(args) if isinstance(args, (list, tuple)) \
        else args.items() if isinstance(args, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        if isinstance(args[key], str):
            if re.compile(r'.+\$\{[^[\$\{]+\}.+').match(args[key]):
                args[key] = re.sub(r'\$\{[^[\$\{]+\}', _interpolate, args[key])  # Strings
            elif re.compile(r'\$\{[^[\$\{]+\}').match(args[key]):
                args[key] = get(args, args[key][2:][:-1])  # Objects

        for rule in grammar:
            args[key] = rule(args[key])

        interpolate(value)  # Recurse through inner values

    return args


def multirun(args):
    # Divide args into multiple copies
    return args


def get_args(source=None):
    def decorator_func(func):
        yaml_search_paths.append(app + '/' + source.split('/', 1)[0])

        args = None if source is None else read(source)
        args = parse(args)
        args = interpolate(args)
        # args = multirun(args)

        return partial(func, args)
    return decorator_func
