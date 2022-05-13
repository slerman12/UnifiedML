# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import json
import re
import sys
from typing import MutableSequence
import glob
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def plot(path, plot_experiments=None, plot_agents=None, plot_suites=None, plot_tasks=None, steps=np.inf,
         write_tabular=False, plot_bar=True,
         include_train=False):  # TODO
    include_train = False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Make sure non empty and lists, and gather names
    empty = True
    specs = [plot_experiments, plot_agents, plot_suites, plot_tasks]
    plot_name = ''
    for i, spec in enumerate(specs):
        if spec is not None:
            empty = False
            if not isinstance(spec, MutableSequence):
                specs[i] = [spec]
            # Plot name
            plot_name += "_".join(specs[i]) + '_'
    if empty:
        return

    # Style
    # RdYlBu, Set1, Set2, Set3, gist_stern, icefire
    sns.set_theme(style="darkgrid", palette='Set2', font_scale=0.7,
                  rc={
                      'legend.loc': 'lower right', 'figure.dpi': 400,
                      # 'legend.fontsize': 4, 'font.size': 4,
                      # 'axes.titlesize': 4, 'axes.labelsize': 4,
                      # 'xtick.labelsize': 7,
                      # 'ytick.labelsize': 7,
                      # 'figure.titlesize': 4, 'legend.title_fontsize': 4
                  })

    # All CSVs from path, recursive
    csv_names = glob.glob('./Benchmarking/*/*/*/*.csv', recursive=True)

    csv_list = []
    # max_csv_list = []  # Unused
    found_suite_tasks = set()
    found_suites = set()
    min_steps = steps

    # Data recollection/parsing
    for csv_name in csv_names:
        # Parse files
        experiment, agent, suite, task_seed_eval = csv_name.split('/')[2:]
        task_seed = task_seed_eval.split('_')
        suite_task, seed, eval = '_'.join(task_seed[:-2]), task_seed[-2], task_seed[-1].replace('.csv', '')

        # Map suite names to properly-cased names
        suite = {k.lower(): k for k in ['Atari', 'DMC', 'Classify']}[suite.lower()]

        # Whether to include this CSV
        include = True

        if not include_train and eval.lower() != 'eval':
            include = False

        datums = [experiment, agent, suite.lower(), suite_task]
        for i, spec in enumerate(specs):
            if spec is not None and not re.match('^(%s)+$' % '|'.join(spec).replace('(', '\(').replace(')', '\)'),
                                                 datums[i], re.IGNORECASE):
                if i == 3 and re.match('^.*(%s)+$' % '|'.join(spec).replace('(', '\(').replace(')', '\)'),
                                       datums[i], re.IGNORECASE):
                    break
                include = False

        if not include:
            continue

        # Add CSV
        csv = pd.read_csv(csv_name)

        length = int(csv['step'].max())
        if length == 0:
            continue

        # Min number of steps
        min_steps = min(min_steps, length)

        found_suite_task = suite_task + ' (' + suite + ')'
        csv['Agent'] = agent + ' (' + experiment + ')'
        csv['Suite'] = suite
        csv['Task'] = found_suite_task

        # Rolling max per run (as in CURL, SUNRISE) This was critiqued heavily in https://arxiv.org/pdf/2108.13264.pdf
        # max_csv = csv.copy()
        # max_csv['reward'] = max_csv[['reward', 'step']].rolling(length, min_periods=1, on='step').max()['reward']

        csv_list.append(csv)
        # max_csv_list.append(max_csv)
        found_suite_tasks.update({found_suite_task})
        found_suites.update({suite})

    # Non-empty check
    if len(csv_list) == 0:
        return

    df = pd.concat(csv_list, ignore_index=True)
    # max_df = pd.concat(max_csv_list, ignore_index=True)  # Unused
    found_suite_tasks = np.sort(list(found_suite_tasks))

    tabular_mean = {}
    tabular_median = {}
    tabular_normalized_mean = {}
    tabular_normalized_median = {}

    # PLOTTING (tasks)

    # Dynamically compute num columns/rows
    num_rows = int(np.floor(np.sqrt(len(found_suite_tasks))))
    while len(found_suite_tasks) % num_rows != 0:
        num_rows -= 1
    num_cols = len(found_suite_tasks) // num_rows

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

    # Plot tasks
    for i, suite_task in enumerate(found_suite_tasks):
        task_data = df[df['Task'] == suite_task]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        if steps < np.inf:
            task_data = task_data[task_data['Step'] <= steps]

        # No need to show Agent in legend if all same
        if len(task_data.Agent.str.split('(').str[0].unique()) == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                task_data['Agent'] = task_data.Agent.str.split('(').str[1:].str.join('(').str.split(')').str[:-1].str.join(')')

        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs
        hue_order = np.sort(task_data.Agent.unique())

        # Format title
        title = ' '.join([task_name[0].upper() + task_name[1:] for task_name in suite_task.split('_')])

        suite = title.split('(')[1].split(')')[0]
        task = title.split(' (')[0]

        y_axis = 'Accuracy' if 'classify' in suite.lower() else 'Reward'

        if write_tabular or plot_bar:
            # Aggregate tabular data over all seeds/runs
            for agent in task_data.Agent.unique():
                for tabular in [tabular_mean, tabular_median, tabular_normalized_mean, tabular_normalized_median]:
                    if agent not in tabular:
                        tabular[agent] = {}
                    if suite not in tabular[agent]:
                        tabular[agent][suite] = {}
                scores = task_data.loc[(task_data['Step'] == min_steps) & (task_data['Agent'] == agent), y_axis]
                tabular_mean[agent][suite][task] = scores.mean()
                tabular_median[agent][suite][task] = scores.median()
                for t in low:
                    if t.lower() in suite_task.lower():
                        normalized = (scores - low[t]) / (high[t] - low[t])
                        tabular_normalized_mean[agent][suite][task] = normalized.mean()
                        tabular_normalized_median[agent][suite][task] = normalized.median()
                        break

        sns.lineplot(x='Step', y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax,
                     # palette='pastel'
                     )
        ax.set_title(f'{title}')

        if 'classify' in suite.lower():
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Eval Accuracy')

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Tasks.png'))

    plt.close()

    # PLOTTING (suites)

    num_cols = len(found_suites)

    # Create subplots
    fig, axs = plt.subplots(1, num_cols, figsize=(4 * num_cols, 3))

    # Sort suites
    found_suites = [found for s in ['Atari', 'DMC', 'Classify'] for found in found_suites if s in found]

    # Plot suites
    for col, suite in enumerate(found_suites):
        task_data = df[df['Suite'] == suite]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        if steps < np.inf:
            task_data = task_data[task_data['Step'] <= steps]

        y_axis = 'Accuracy' if 'classify' in suite.lower() else 'Reward'

        # High-low-normalize
        for suite_task in task_data.Task.unique():
            for t in low:
                if t.lower() in suite_task.lower():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=SettingWithCopyWarning)

                        task_data.loc[task_data['Task'] == suite_task, y_axis] -= low[t]
                        task_data.loc[task_data['Task'] == suite_task, y_axis] /= high[t] - low[t]
                        continue

        ax = axs[col] if num_cols > 1 else axs
        hue_order = np.sort(task_data.Agent.unique())

        sns.lineplot(x='Step', y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax,
                     # palette='pastel'
                     )
        ax.set_title(f'{suite}')

        if suite.lower() == 'atari':
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Human-Normalized Score')
        elif suite.lower() == 'dmc':
            ax.set_ybound(0, 1000)
        elif suite.lower() == 'classify':
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Eval Accuracy')

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Suites.png'))

    plt.close()

    # Tabular data
    if write_tabular:
        f = open(path / (plot_name + f'{int(min_steps)}-Steps_Tabular.json'), "w")
        tabular_data = {'Mean': tabular_mean,
                        'Median': tabular_median,
                        'Normalized Mean': tabular_normalized_mean,
                        'Normalized Median': tabular_normalized_median}
        for agg_name, agg in zip(['Mean', 'Median'], [np.mean, np.median]):
            for name, tabular in zip(['Mean', 'Median'], [tabular_normalized_mean, tabular_normalized_median]):
                tabular_data.update({
                    f'{agg_name} Normalized-{name}': {
                        agent: {
                            suite:
                                agg([val for val in tabular[agent][suite].values()])
                            for suite in tabular[agent]}
                        for agent in tabular}
                })
        json.dump(tabular_data, f, indent=2)
        f.close()

    # Bar plot
    if plot_bar:
        bar_data = {suite_name: {'Task': [], 'Median': [], 'Agent': []} for suite_name in found_suites}
        for agent in tabular_median:
            for suite in tabular_median[agent]:
                for task in tabular_median[agent][suite]:
                    median = tabular_median
                    for t in low:
                        if t.lower() == suite.lower() or t.lower() == task.lower():
                            median = tabular_normalized_median
                            break
                    bar_data[suite]['Task'].append(task)
                    bar_data[suite]['Median'].append(median[agent][suite][task])
                    bar_data[suite]['Agent'].append(agent)

        # Create subplots
        fig, axs = plt.subplots(1, num_cols, figsize=(4 * num_cols, 3))

        for col, suite in enumerate(bar_data):
            task_data = pd.DataFrame(bar_data[suite])

            ax = axs[col] if num_cols > 1 else axs

            hue_order = sorted(set(bar_data[suite]['Agent']))
            sns.barplot(x='Task', y='Median', ci='sd', hue='Agent', data=task_data, ax=ax, hue_order=hue_order,
                        # palette='pastel'
                        )

            ax.set_title(f'{suite} (@{min_steps} Steps)')

            if suite.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Median Human-Normalized')
            elif suite.lower() == 'dmc':
                ax.set_ybound(0, 1000)
                ax.set_ylabel('Median Reward')
            elif suite.lower() == 'classify':
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Median Eval Accuracy')

            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate('{:.0f}'.format(height) if suite.lower() == 'dmc' else f'{height:.0%}',
                            (x + width/2, y + height), ha='center', size=45)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(path / (plot_name + 'Bar.png'))

        plt.close()


# Lows and highs for normalization

atari_random = {
    'Alien': 88.4,
    'Amidar': 1.09,
    'Assault': 67.2,
    'Asterix': 154.0,
    'BankHeist': 2.6,
    'BattleZone': 660.0,
    'Boxing': 1.1,
    'Breakout': 0.1,
    'ChopperCommand': 285.0,
    'CrazyClimber': 1205.0,
    'DemonAttack': 45.4,
    'Freeway': 0.0,
    'Frostbite': 20.2,
    'Gopher': 208.8,
    'Hero': 16.5,
    'Jamesbond': 0.5,
    'Kangaroo': 14.0,
    'Krull': 611.2,
    'KungFuMaster': 92.0,
    'MsPacman': 107.2,
    'Pong': -20.44,
    'PrivateEye': -1.14,
    'Qbert': 67.75,
    'RoadRunner': 2.0,
    'Seaquest': 20.4,
    'UpNDown': 65.8
}
atari_human = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}

low = {**atari_random}
high = {**atari_human}


@hydra.main(config_path='Hyperparams', config_name='args')
def main(args):
    OmegaConf.set_struct(args, False)
    del args.plotting['_target_']
    if 'path' not in sys_args:
        if isinstance(args.plotting.plot_experiments, str):
            args.plotting.plot_experiments = [args.plotting.plot_experiments]
        args.plotting.path = f"./Benchmarking/{'_'.join(args.plotting.plot_experiments)}/Plots"
    if 'steps' not in sys_args:
        args.plotting.steps = np.inf
    plot(**args.plotting)


if __name__ == "__main__":
    sys_args = []
    for i in range(1, len(sys.argv)):
        sys_args.append(sys.argv[i].split('=')[0].strip('"').strip("'"))
        sys.argv[i] = 'plotting.' + sys.argv[i] if sys.argv[i][0] != "'" and sys.argv[i][0] != '"' \
            else sys.argv[i][0] + 'plotting.' + sys.argv[i][1:]
    main()
