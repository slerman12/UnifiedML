# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
import re
from pathlib import Path
from termcolor import colored

import numpy as np

import torch


def shorthand(log_name):
    return ''.join([s[0].upper() for s in re.split('_|[ ]', log_name)] if len(log_name) > 3 else log_name.upper())


def format(log, log_name):
    l = shorthand(log_name)

    if 'time' in log_name.lower():
        log = str(datetime.timedelta(seconds=int(log)))
        return f'{l}: {log}'
    elif float(log).is_integer():
        log = int(log)
        return f'{l}: {log}'
    else:
        return f'{l}: {log:.04f}'


class Logger:
    def __init__(self, task, seed, path='.', aggregation='mean', wandb=False):

        self.path = path
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.task = task
        self.seed = seed

        self.logs = {}

        self.aggregation = aggregation
        self.default_aggregations = {'step': np.median, 'frame': np.median, 'episode': np.median, 'epoch': np.median,
                                     'time': np.mean, 'fps': np.mean}

        self.wandb = 'uninitialized' if wandb \
            else None

    def log(self, log=None, name="Logs", dump=False):
        if log is not None:

            if name not in self.logs:
                self.logs[name] = {}

            logs = self.logs[name]

            for k, l in log.items():
                if isinstance(l, torch.Tensor):
                    l = l.detach().numpy()
                logs[k] = logs[k] + [l] if k in logs else [l]

        if dump:
            self.dump_logs(name)

    def dump_logs(self, name=None):
        if name is None:
            # Iterate through all logs
            for n in self.logs:
                for log_name in self.logs[n]:
                    agg = self.default_aggregations.get(log_name, np.mean if self.aggregation == 'mean' else np.median)
                    self.logs[n][log_name] = agg(self.logs[n][log_name])
                self._dump_logs(self.logs[n], name=n)
                del self.logs[n]
        else:
            # Iterate through just the named log
            if name not in self.logs:
                return
            for log_name in self.logs[name]:
                agg = self.default_aggregations.get(log_name, np.mean if self.aggregation == 'mean' else np.median)
                self.logs[name][log_name] = agg(self.logs[name][log_name])
            self._dump_logs(self.logs[name], name=name)
            self.logs[name] = {}
            del self.logs[name]

    def _dump_logs(self, logs, name):
        self.dump_to_console(logs, name=name)
        self.dump_to_csv(logs, name=name)
        if self.wandb is not None:
            self.log_wandb(logs, name=name)

    def dump_to_console(self, logs, name):
        name = colored(name, 'yellow' if name.lower() == 'train' else 'green' if name.lower() == 'eval' else None,
                       attrs=['dark'] if name.lower() == 'seed' else None)
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def remove_old_entries(self, logs, file_name):
        rows = []
        with file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= logs['step']:
                    break
                rows.append(row)
        with file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=logs.keys(),
                                    extrasaction='ignore',
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_to_csv(self, logs, name):
        logs = dict(logs)

        assert 'step' in logs

        file_name = Path(self.path) / f'{self.task}_{self.seed}_{name}.csv'

        write_header = True
        if file_name.exists():
            write_header = False
            self.remove_old_entries(logs, file_name)

        file = file_name.open('a')
        writer = csv.DictWriter(file,
                                fieldnames=logs.keys(),
                                restval=0.0)
        if write_header:
            writer.writeheader()

        writer.writerow(logs)
        file.flush()

    def log_wandb(self, logs, name):
        if self.wandb == 'uninitialized':
            import wandb

            experiment, agent, suite = self.path.split('/')[2:5]

            wandb.init(project=experiment, name=f'{agent}_{suite}_{self.task}_{self.seed}', dir=self.path)

            for file in ['', '*/', '*/*/', '*/*/*/']:
                try:
                    wandb.save(f'./Hyperparams/{file}*.yaml')
                except Exception:
                    pass

            self.wandb = wandb

        measure = 'reward' if 'reward' in logs else 'accuracy'
        logs[f'{measure} ({name})'] = logs.pop(f'{measure}')

        self.wandb.log(logs, step=int(logs['step']))
