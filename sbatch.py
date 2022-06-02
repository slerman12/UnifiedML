# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import subprocess
import sys
from pathlib import Path

import hydra


sys_args = [arg.split('=')[0].strip('"').strip("'") for arg in sys.argv[1:]]
meta = ['username', 'conda', 'num_gpus', 'gpu', 'mem', 'time', 'lab', 'reservation_id', '-m']


def getattr_recursive(__o, name):
    for key in name.split('.'):
        __o = getattr(__o, key)
    return __o


@hydra.main(config_path='./Hyperparams', config_name='sbatch')
def main(args):
    path = args.logger.path.replace('Agents.', '')
    Path(path).mkdir(parents=True, exist_ok=True)

    if 'task' in sys_args:
        args.task = args.task.lower()

    if 'transform' in sys_args:
        args.transform = f'"{args.transform}"'.replace("'", '')

    if 'stddev_schedule' in sys_args:
        args.stddev_schedule = f'"{args.stddev_schedule}"'

    if 'experiment' in sys_args:
        args.experiment = f'"{args.experiment}"'

    conda = ''.join([f'*"{gpu}"*)\nsource /scratch/{args.username}/miniconda/bin/activate {env}\n;;\n'
                     for gpu, cuda_version, env, _ in [('K80', 11.0, 'agi', 10.2), ('V100', 11.0, 'agi', 10.2),
                                                       ('A100', 11.2, 'CUDA11', 11.3), ('RTX', 11.2, 'agi', 10.2)]])
    cuda = f'GPU_TYPE=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail  -1)\ncase $GPU_TYPE in\n{conda}esac'

    script = f"""#!/bin/bash
#SBATCH -c {args.num_workers + 1}
{f'#SBATCH -p gpu --gres=gpu:{args.num_gpus}' if args.num_gpus else ''}
{'#SBATCH -p csxu -A cxu22_lab' if args.lab else ''}
{f'#SBATCH -p reserved --reservation={args.username}-{args.reservation_id}' if args.reservation_id else ''}
#SBATCH -t {args.time} -o {path}{args.task_name}_{args.seed}.log -J {args.experiment}
#SBATCH --mem={args.mem}gb 
{f'#SBATCH -C {args.gpu}' if args.num_gpus else ''}
{cuda}
python3 Run.py {' '.join([f"'{key}={getattr_recursive(args, key.strip('+'))}'" for key in sys_args if key not in meta])}
"""

    # Write script
    with open("./sbatch_script", "w") as file:
        file.write(script)

    # Launch script (with error checking / re-launching)
    while True:
        try:
            success = str(subprocess.check_output(['sbatch {}'.format("sbatch_script")], shell=True))
            print(success[2:][:-3])
            if "error" not in success:
                break
        except Exception:
            pass
        print("Errored... trying again")
    print("Success!")


if __name__ == '__main__':
    main()
