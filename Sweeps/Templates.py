# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
def template(name):
    return convert_to_attr_dict({
        name: {
            'sweep': [
                # Sweep commands go here
            ],
            'plots': [
                # Sets of plots
                [],
            ],

            # Plotting-related commands go here
            'sftp': True,
            'bluehive': True,
            'lab': True,
            'steps': 5e5,
            'title': name,
            'x_axis': 'Step',
            'bluehive_only': [],
            'tasks': [],
            'agents': [],
            'suites': []},
    })


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)


# Deep-converts an iterable into an AttrDict
def convert_to_attr_dict(iterable):
    if isinstance(iterable, dict):
        iterable = AttrDict(iterable)

    items = enumerate(iterable) if isinstance(iterable, (list, tuple)) \
        else iterable.items() if isinstance(iterable, AttrDict) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        iterable[key] = convert_to_attr_dict(value)  # Recurse through inner values

    return iterable


"""
Common sweep iterables
"""

dmc = 'dmc/cheetah_run,dmc/quadruped_walk,dmc/reacher_easy,dmc/cup_catch,dmc/finger_spin,dmc/walker_walk'

atari = 'atari/pong,atari/breakout,atari/boxing,atari/krull,atari/seaquest,atari/qbert'

atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]
full_atari = f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'
