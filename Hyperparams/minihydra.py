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

added_modules = {}


# Something like this
def instantiate(args, **kwargs):
    if args is None:
        return

    # For compatibility with old Hydra syntax
    if '_recursive_' in args:
        args.pop('_recursive_')

    args = deepcopy(args)
    args.update(kwargs)

    file, *module = args.pop('_target_').rsplit('.', 1)

    sub_module, *sub_modules = file.split('.')

    # Can instantiate based on added modules
    if sub_module in added_modules:
        sub_module = added_modules[sub_module]

        try:
            for key in sub_modules + module:
                sub_module = getattr(sub_module, key)

            return sub_module(**args)
        except AttributeError:
            pass

    file = file.replace('.', '/')
    for path in yaml_search_paths:
        try:
            # Reuse cached imports
            if module + '_inst' in sys.modules:
                return getattr(sys.modules[module + '_inst'], module)(**args)

            # Reuse cached imports
            for key, value in sys.modules.items():
                if hasattr(value, '__file__') and value.__file__ and path + '/' + file + '.py' in value.__file__:
                    return getattr(value, module)(**args)

            # Import
            spec = importlib.util.spec_from_file_location(module, path + '/' + file + '.py')
            foo = importlib.util.module_from_spec(spec)
            sys.modules[module + '_inst'] = foo
            spec.loader.exec_module(foo)
            return getattr(foo, module)(**args)
        except (FileNotFoundError, AttributeError):
            continue
    raise FileNotFoundError(f'Could not find {module} in /{file}.py. Search paths include: {yaml_search_paths}')


def open_yaml(source):
    for path in yaml_search_paths + ['']:
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
        args[key] = recursive_update(args.get(key, {}), value) if isinstance(value, type(args2)) else value
    return args


def read(source, parse_task=True):
    args = open_yaml(source)

    # Need to allow imports  TODO Might have to add to yaml_search_paths
    if 'imports' in args:
        imports = args.pop('imports')

        self = deepcopy(args)

        for module in imports:
            module = self if module == 'self' else read(module + '.yaml', parse_task=False)
            recursive_update(args, module)

    # Parse task
    if parse_task:  # Not needed in imports recursions
        for sys_arg in sys.argv[1:]:
            key, value = sys_arg.split('=', 1)
            if key == 'task':
                args['task'] = value

        # Command-line task import
        if 'task' in args and args.task not in [None, 'null']:
            try:
                task = read('task/' + args.task + '.yaml', parse_task=False)
            except FileNotFoundError:
                task = read(args.task + '.yaml', parse_task=False)
            recursive_update(args, task)

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
    return interpolate(args)  # Interpolate to make sure gotten value is resolved


# minihydra.grammar.append(rule)
grammar = []  # List of funcs


def interpolate(arg, args=None):
    if args is None:
        args = arg

    def _interpolate(match_obj):
        if match_obj.group() is not None:
            return str(get(args, match_obj.group()[2:][:-1]))

    items = enumerate(arg) if isinstance(arg, (list, tuple)) \
        else arg.items() if isinstance(arg, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        if isinstance(arg[key], str):
            if re.compile(r'.+\$\{[^[\$\{]+\}.+').match(arg[key]):
                arg[key] = re.sub(r'\$\{[^[\$\{]+\}', _interpolate, arg[key])  # Strings
            elif re.compile(r'\$\{[^[\$\{]+\}').match(arg[key]):
                arg[key] = get(args, arg[key][2:][:-1])  # Objects

        for rule in grammar:
            arg[key] = rule(arg[key])

        interpolate(value, args)  # Recurse through inner values

    return args


def multirun(args):
    # Divide args into multiple copies
    return args


def decorate(func, source):
    if source is not None:
        yaml_search_paths.append(app + '/' + source.split('/', 1)[0])

    args = Args() if source is None else read(source)
    args = parse(args)
    args = interpolate(args)  # Command-line requires quotes for interpolation
    # args = multirun(args)

    return func(args)


def get_args(source=None):
    def decorator_func(func):
        return partial(decorate, func, source)
    return decorator_func
