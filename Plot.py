# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import re
import sys
from typing import MutableSequence
from operator import iand
from functools import reduce
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

import matplotlib.pyplot as plt
from matplotlib import ticker, dates, lines
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns


def plot(path, plot_experiments=None, plot_agents=None, plot_suites=None, plot_tasks=None, steps=None,
         write_tabular=False, plot_train=False, title='UnifiedML', x_axis='Step', verbose=False):

    path = Path(path + f'/{"Train" if plot_train else "Eval"}')
    path.mkdir(parents=True, exist_ok=True)

    if steps is None:
        steps = np.inf

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
            plot_name += "_".join(specs[i] if i == 0 or len(specs[i]) < 5 else (specs[i][:5] + ['etc'])) + '_'
    plot_name = plot_name.strip('.')
    if empty:
        return

    # Style

    # RdYlBu, Set1, Set2, Set3, gist_stern, icefire, tab10_r, Dark2
    possible_palettes = ['Accent', 'RdYlBu', 'Set1', 'Set2', 'Set3', 'gist_stern', 'icefire', 'tab10_r', 'Dark2']
    # Note: finite number of color palettes: could error out if try to plot a billion tasks in one figure
    palette_colors = sum([sns.color_palette(palette) for palette in possible_palettes], [])

    sns.set_theme(font_scale=0.7,
                  rc={
                      'legend.loc': 'lower right', 'figure.dpi': 400,
                      'legend.fontsize': 5.5, 'legend.title_fontsize': 5.5,
                      # 'axes.titlesize': 4, 'axes.labelsize': 4, 'font.size': 4,
                      # 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                      # 'figure.titlesize': 4
                  })

    # All CSVs from path, recursive
    csv_names = glob.glob('./Benchmarking/*/*/*/*.csv', recursive=True)

    csv_list = []
    # max_csv_list = []  # Unused
    min_steps = steps

    predicted_vs_actual_list = []
    found_predicted_vs_actual = set()

    # Data recollection/parsing
    for csv_name in csv_names:
        # Parse files
        experiment, agent, suite, task_seed_eval = csv_name.split('/')[2:]
        split_size = 3 if 'Generate' in task_seed_eval else 5 if 'Predicted_vs_Actual' in task_seed_eval else 2
        task_seed = task_seed_eval.rsplit('_', split_size)
        suite_task, seed, eval = task_seed[0], task_seed[1], '_'.join(task_seed[2:]).replace('.csv', '')

        # Map suite names to properly-cased names
        suite = {k.lower(): k for k in ['Atari', 'DMC', 'Classify']}.get(suite.lower(), suite)

        # Whether to include this CSV
        include = True

        mode = 'train' if plot_train else 'eval'

        if eval.lower() not in [mode, f'predicted_vs_actual_{mode}']:
            include = False

        datums = [experiment, agent, suite.lower(), suite_task]
        for i, spec in enumerate(specs):
            if spec is not None and not re.match('^(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                    ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):
                if i == 3 and re.match('^.*(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                        ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):  # Tasks can be specified with suite
                    break
                include = False
                break

        if not include:
            continue

        # Add CSV
        csv = pd.read_csv(csv_name)

        if 'step' in csv.columns and 'predicted_vs_actual' not in eval.lower():
            length = int(csv.loc[csv['step'] <= steps, 'step'].max())
            if length == 0:
                continue

            # Min number of steps over all tasks/experiments  TODO per suite, task
            # Assumes data available for a single shared step across tasks - may not be true when log steps inconsistent
            min_steps = min(min_steps, length)

            if verbose and length < steps != np.inf:
                print(f'[Experiment {experiment} Agent {agent} Suite {suite} Task {suite_task} Seed {seed}] '
                      f'has {length} steps.')

        found_suite_task = suite_task + ' (' + suite + ')'

        csv['Agent'] = agent + ' (' + experiment + ')'
        csv['Suite'] = suite
        csv['Task'] = found_suite_task
        csv['Seed'] = seed

        # Rolling max per run (as in CURL, SUNRISE) This was critiqued heavily in https://arxiv.org/pdf/2108.13264.pdf
        # max_csv = csv.copy()
        # max_csv['reward'] = max_csv[['reward', 'step']].rolling(length, min_periods=1, on='step').max()['reward']

        if 'predicted_vs_actual' in eval.lower():
            predicted_vs_actual_list.append(csv)
            found_predicted_vs_actual.update({found_suite_task})
        else:
            csv_list.append(csv)

    universal_hue_order, palette = [], {}

    pd.options.mode.chained_assignment = None

    # Non-empty check
    if len(csv_list) > 0:
        df = pd.concat(csv_list, ignore_index=True)

        # Capitalize column names
        df.columns = [' '.join([name[0].capitalize() + name[1:] for name in re.split(r'_|\s+', col_name)])
                      for col_name in df.columns]

        # Steps cap
        if steps < np.inf:
            df = df[df['Step'] <= steps]

        universal_hue_order, handles = np.sort(df.Agent.unique()), {}
        palette = {agent: color for agent, color in zip(universal_hue_order, palette_colors[:len(universal_hue_order)])}

        x_axis = x_axis.capitalize()

        # PLOTTING (tasks)

        def make(cell_data, ax, ax_title, cell_palettes, hue_names, **kwargs):
            suite = ax_title.split('(')[1].split(')')[0]

            # Pre-processing
            x = x_axis if x_axis in cell_data.columns else 'Step'
            if x == 'Time':
                cell_data['Time'] = pd.to_datetime(cell_data['Time'], unit='s')

            y = 'Accuracy' if 'classify' in suite.lower() \
                else 'Reward'

            sns.lineplot(x=x, y=y, data=cell_data, ci='sd', hue='Agent', hue_order=np.sort(hue_names), ax=ax,
                         palette=cell_palettes)

            # Post-processing
            if x == 'Time':
                ax.set_xlabel("Time (h)")
                ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # For now group x axis into bins only for time

            if 'classify' in suite.lower():
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            # Legend in subplots
            ax.legend(frameon=False).set_title(None)

            # Rotate x-axis names
            ax.tick_params(axis='x', rotation=20)

        general_plot(df, path, plot_name + 'Tasks.png', palette, make, 'Task', title)

        # PLOTTING (suites)

        def make(ax, cell_data, ax_title, cell_palettes, hue_names, **kwargs):
            # Pre-processing
            x = x_axis if x_axis in cell_data.columns else 'Step'
            if x == 'Time':
                cell_data['Time'] = pd.to_datetime(cell_data['Time'], unit='s')

            y = 'Accuracy' if 'classify' in ax_title.lower() \
                else 'Reward'

            # Normalize
            for suite_task in cell_data.Task.unique():
                for t in low:
                    if t.lower() in suite_task.lower():
                        cell_data.loc[cell_data['Task'] == suite_task, y] -= low[t]
                        cell_data.loc[cell_data['Task'] == suite_task, y] /= high[t] - low[t]
                        continue

            sns.lineplot(x=x, y=y, data=cell_data, ci='sd', hue='Agent', hue_order=np.sort(hue_names), ax=ax,
                         palette=cell_palettes)

            # Post-processing
            if x == 'Time':
                ax.set_xlabel("Time (h)")
                ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
                # For now, group x axis into bins only for time
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

            if ax_title.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Human-Normalized Score')
            elif ax_title.lower() == 'dmc':
                ax.set_ybound(0, 1000)
            elif ax_title.lower() == 'classify':
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            # Legend in subplots
            ax.legend(frameon=False).set_title(None)

            # Rotate x-axis names
            ax.tick_params(axis='x', rotation=20)

        general_plot(df, path, plot_name + 'Suites.png', palette, make, 'Suite', title)

        # WRITING (tabular)

        # Consistent steps across all tasks for bar plot and tabular data
        min_time = df.loc[df['Step'] == min_steps, x_axis].unique()  # Checks if specified x-axis "time" is shared
        if len(min_time) > 1:  # If not, just use Step
            x_axis = 'Step'
            min_time = min_steps
        else:
            min_time = min_time[0]  # if so, use that time

        # Only show results for a consistent step count
        df = df[df['Step'] == min_steps]

        # Score name y-axis
        metrics = [metric for metric in ['Accuracy', 'Reward'] if metric in df.columns]

        # Use Reward or Accuracy as "Score"
        df['Score'] = df[metrics[0]]

        if len(metrics) > 1:
            # Where N/A, use the other
            df.loc[df['Score'].isna(), 'Score'] = df[metrics[1]]

        # Mean and Median scores across seeds
        df = df[['Task', 'Suite', 'Agent', 'Score']].groupby(
            ['Task', 'Suite', 'Agent']).agg(Mean=('Score', np.mean), Median=('Score', np.median)).reset_index()

        # Normalized Mean and Median scores across seeds
        for i, (task, suite) in df[['Task', 'Suite']].iterrows():
            df.loc[i, 'NormalizedMean'] = df.loc[i, 'Mean']
            df.loc[i, 'NormalizedMedian'] = df.loc[i, 'Median']

            # Index norm lows/highs by task name or suite name
            task = task.split(' (')[0]
            name = task if task in low and task in high else suite

            if name in low and name in high:
                # Mean
                df.loc[i, f'NormalizedMean'] -= low[name]
                df.loc[i, f'NormalizedMean'] /= high[name] - low[name]

                # Median
                df.loc[i, f'NormalizedMedian'] -= low[name]
                df.loc[i, f'NormalizedMedian'] /= high[name] - low[name]

        # Writes normalized metrics for this step out to csv
        if write_tabular:
            # Tasks Tabular
            df.to_csv(path / (plot_name + f'{int(min_steps)}-Steps_Tasks_Tabular.csv'), index=False)

            # Suites Tabular - Mean or median of scores per suite
            metrics = {name: (seed_agg_name, agg) for seed_agg_name in ['NormalizedMean', 'NormalizedMedian']
                       for name, agg in [('Mean' + seed_agg_name, np.mean), ('Median' + seed_agg_name, np.median)]}
            df[['Suite', 'NormalizedMedian', 'NormalizedMean']].groupby(['Suite']).agg(**metrics).reset_index(). \
                to_csv(path / (plot_name + f'{int(min_steps)}-Steps_Suites_Tabular.csv'), index=False)

        # PLOTTING (bar plot)

        def make(ax, cell_data, cell_palettes, hue_names, **kwargs):
            # Pre-processing
            cell_data['Task'] = cell_data['Task'].str.split('(').str[0]

            sns.barplot(x='Task', y='NormalizedMedian', ci='sd', hue='Agent', data=cell_data, ax=ax,
                        hue_order=np.sort(hue_names), palette=cell_palettes)

            # Post-processing
            if x_axis.lower() == 'time':
                time_str = pd.to_datetime(min_time, unit='s').strftime('%H:%M:%S')
                ax.set_title(f'{suite} (@{time_str}h)')
            else:
                ax.set_title(f'{suite} (@{min_time:.0f} {x_axis}s)')

            if suite.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Median Human-Normalized')
            elif suite.lower() == 'dmc':
                ax.set_ybound(0, 1000)
                ax.set_ylabel('Median Reward')
            elif suite.lower() == 'classify':
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate('{:.0f}'.format(height) if suite.lower() not in ['atari', 'classify'] else f'{height:.0%}',
                            (x + width/2, y + height), ha='center', size=max(min(24 * width, 7), 5),  # No max(keep, 5)?
                            # color='#498057'
                            # color='#3b423d'
                            )

            # Rotate x-axis names and remove axis title
            ax.tick_params(axis='x', rotation=20)
            ax.set(xlabel=None)

        # Max Agents for a Task - for configuring Bar Plot width
        mean_num_agents_per_task = df.groupby(['Task', 'Agent']).size().reset_index().groupby(['Task']).size().mean()
        num_tasks = len(df.Task.unique())
        # num_bars = len(df.groupby(['Task', 'Agent']).size().reset_index().index)

        general_plot(df, path, plot_name + 'Bar.png', palette, make, 'Suite', title, 'Agent', True,
                     figsize=(max(4, num_tasks * mean_num_agents_per_task * 0.7), 3))
        # TODO Try:
        # general_plot(df, path, plot_name + 'Bar.png', palette, make, 'Suite', title, 'Agent', True,
        #              figsize=(num_tasks * mean_num_agents_per_task / 2, 3))

    # Class Sizes & Heatmap
    if len(predicted_vs_actual_list) > 0:
        df = pd.concat(predicted_vs_actual_list, ignore_index=True)

        # Capitalize column names
        df.columns = [' '.join([name[0].capitalize() + name[1:] for name in re.split(r'_|\s+', col_name)])
                      for col_name in df.columns]

        df = df.astype({'Predicted': int, 'Actual': int})

        i = 0
        for agent in np.sort(df.Agent.unique()):
            if agent not in palette:
                palette[agent] = palette_colors[len(universal_hue_order) + i]
                i += 1

        original_df = df.copy()

        step = df[['Task', 'Step']].groupby('Task').max().reset_index() if 'Step' in df.columns else None
        # step = df[['Task', 'Step']].groupby('Task').max().reset_index()  # TODO Use this after XRD

        df['Accuracy'] = 0
        df.loc[df['Predicted'] == df['Actual'], 'Accuracy'] = 1
        df['Count'] = 1
        df.drop(['Predicted'], axis=1)
        df = df.rename(columns={'Actual': 'Class Label'})

        num_seeds = df.groupby(['Class Label', 'Agent', 'Task'])['Seed'].value_counts()
        num_seeds = num_seeds.groupby(['Class Label', 'Agent', 'Task']).count().reset_index()

        df = df.groupby(['Class Label', 'Agent', 'Task']).agg({'Accuracy': 'sum', 'Count': 'size'}).reset_index()
        df['Accuracy'] /= df['Count']
        df['Count'] /= num_seeds['Seed']

        # Plotting (class Sizes)

        def make(ax, ax_title, cell_data, cell, hue_names, cell_palettes, **kwargs):
            sns.scatterplot(data=cell_data, x='Class Label', y='Accuracy', hue='Agent', size='Count',
                            alpha=0.7, hue_order=np.sort(hue_names), ax=ax, palette=cell_palettes)

            #  Post-processing
            # step_ = f' (@{int(step.loc[step["Task"] == cell[0], "Step"])} Steps)'
            step_ = ' (@500000 Steps)'
            ax.set_title(f'{ax_title}{step_}')
            ax.set_ybound(-0.05, 1.05)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

        general_plot(df, path, plot_name + 'ClassSizes.png', palette,
                     make, 'Task', title, 'Agent', True, False, False)

        # Plotting (heatmap)

        def make(cell_data, ax, cell_palettes, hue_names, **kwargs):
            # Pre-processing
            cell_data = pd.crosstab(cell_data.Predicted, cell_data.Actual)  # To matrix
            cell_data = cell_data.div(cell_data.sum(axis=0), axis=1)  # Normalize

            sns.heatmap(cell_data, ax=ax, linewidths=.5, vmin=0, vmax=1,  # Normalizes color bar in [0, 1]
                        cmap=sns.light_palette(cell_palettes[hue_names[0]], as_cmap=True))

            # Post-processing
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
            # ax.invert_yaxis()  # Can optionally represent vertically flipped

        general_plot(original_df, path, plot_name + 'Heatmap.png', palette,
                     make, ['Task', 'Agent'], title, 'Agent', False, True)


def general_plot(data, path, plot_name, palette, make_func, per='Task', title='UnifiedML', hue='Agent',
                 legend_aside=False, universal_legend=False, legend_title=True, figsize=None):
    if not isinstance(per, list):
        per = [per]

    cells = data[per].groupby(per).size().reset_index().drop(columns=0)

    total = len(cells.index)

    # Manually compute a full grid of full square-ish shape
    extra = 0
    num_rows = int(np.floor(np.sqrt(total)))
    while total % num_rows != 0:
        num_rows -= 1
    num_cols = total // num_rows

    # If too rectangular, automatically infer num cols/rows square with empty extra cells
    if num_cols / num_rows > 5:
        num_cols = int(np.ceil(np.sqrt(total)))
        num_rows = int(np.ceil(total / num_cols))
        extra = num_rows * num_cols - total

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 3 * num_rows) if figsize is None else figsize)

    # Title
    if title is not None:
        fig.suptitle(title, y=0.99 if total > 10 else None)  # y controls height of title
        # fig.suptitle(title)

    cell_palettes = {}

    for i, cell in cells.iterrows():
        cell_data = data[reduce(iand, [data[per_name] == cell_name for per_name, cell_name in zip(per, cell)])]

        # Unique colors for this cell
        hue_names = cell_data[hue].unique()

        # No need to show Agent name in legend if all same - TODO perhaps across the whole plot, not per cell
        if len((data if universal_legend else cell_data)[hue].str.split('(').str[0].unique()) == 1:  # Unique Agent name
            # Remove Agent name from data columns
            cell_data[hue] = cell_data[hue].str.split('(').str[1:].str.join('(').str.split(')'
                                                                                           ).str[:-1].str.join(')')
            # Remove Agent name from legend
            for j, hue_name in enumerate(hue_names):
                hue_names[j] = ')'.join('('.join(hue_name.split('(')[1:]).split(')')[:-1])
                cell_palettes.update({hue_names[j]: palette[hue_name]})
        else:
            cell_palettes.update({hue_name: palette[hue_name] for hue_name in hue_names})

        # Rows and cols
        row = i // num_cols
        col = i % num_cols

        # Empty extras
        if row == num_rows - 1 and col > num_cols - 1 - extra:
            break

        # Cell plot ("ax")
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs

        # Cell title
        ax_title = ' '.join([name[0].upper() + name[1:] for name in cell[0].split('_')])

        if len(per) > 1:
            ax_title += ' :  ' + ' :  '.join(cell[1:])

        ax.set_title(ax_title)

        make_func(**locals())

        # Hide legend title
        if not legend_title:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])

        # Legend next to subplots
        if legend_aside:
            ax.legend(**{} if legend_title else {'handles': handles[1:], 'labels': labels[1:]},
                      loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, frameon=False,)

    # Universal legend
    if universal_legend:
        # Color order in legend between hue <-> palette
        hue_order = np.sort([hue_name for hue_name in cell_palettes])

        ax = axs[0, -1] if num_rows > 1 and num_cols > 1 else axs[-1] if num_cols > 1 \
            else axs[0] if num_rows > 1 else axs
        handles = [lines.Line2D([0], [0], marker='o', color=cell_palettes[label], label=label, linewidth=0)
                   for label in hue_order]
        ax.legend(handles, hue_order, loc=2, bbox_to_anchor=(1.25, 1.05),
                  borderaxespad=0, frameon=False).set_title(hue)

    for i in range(extra):
        fig.delaxes(axs[num_rows - 1, num_cols - i - 1])

    plt.tight_layout()
    plt.savefig(path / plot_name)

    plt.close()


# Lows and highs for normalization

atari_random = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
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


if __name__ == "__main__":

    @hydra.main(config_path='Hyperparams', config_name='args')  # Note: This still outputs a hydra params file
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

    # Format path names
    # e.g. Checkpoints/Agents.DQNAgent -> Checkpoints/DQNAgent
    OmegaConf.register_new_resolver("format", lambda name: name.split('.')[-1])

    sys_args = []
    for i in range(1, len(sys.argv)):
        sys_args.append(sys.argv[i].split('=')[0].strip('"').strip("'"))
        sys.argv[i] = 'plotting.' + sys.argv[i] if sys.argv[i][0] != "'" and sys.argv[i][0] != '"' \
            else sys.argv[i][0] + 'plotting.' + sys.argv[i][1:]

    main()
