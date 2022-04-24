# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import subprocess
import sys
from pathlib import Path

import hydra


sys_args = [arg.split('=')[0] for arg in sys.argv[1:]]
meta = ['conda', 'num_gpus', 'mem', 'lab', '-m', 'task']


def getattr_recursive(__o, name):
    for key in name.split('.'):
        __o = getattr(__o, key)
    print(__o, type(__o))
    return __o


@hydra.main(config_path='./Hyperparams', config_name='sbatch')
def main(args):
    path = args.logger.path.replace('Agents.', '')
    Path(path).mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH -c {args.num_workers + 1}
{f'#SBATCH -p gpu --gres=gpu:{args.num_gpus}' if args.num_gpus else ''}
{'#SBATCH -p csxu -A cxu22_lab' if args.lab else ''}
#SBATCH -t 5-00:00:00 -o {path}log -J {args.experiment}
#SBATCH --mem={args.mem}gb 
{'#SBATCH -C K80|V100' if args.num_gpus else ''}
{args.conda}
python3 Run.py {' '.join([f'{key}={getattr_recursive(args, key)}' for key in sys_args if key not in meta])}
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
