import glob
import os

easy = ['cartpole_balance', 'cartpole_balance_sparse', 'cartpole_swingup', 'cup_catch', 'finger_spin', 'hopper_stand', 'pendulum_swingup', 'walker_stand', 'walker_walk']
medium = ['acrobot_swingup', 'cartpole_swingup_sparse', 'cheetah_run', 'finger_turn_easy', 'finger_turn_hard', 'hopper_hop', 'quadruped_run', 'quadruped_walk', 'reach_duplo', 'reacher_easy', 'reacher_hard', 'walker_run']
hard = ['humanoid_stand', 'reacher_hard', 'humanoid_walk', 'humanoid_run', 'finger_turn_hard']

if __name__ == '__main__':
    files = glob.glob(os.getcwd() + "/*")

    # Prints tasks by difficulty
    # print([f.split('.')[-2].split('/')[-1] for f in files if 'hard' in open(f, 'r').read() and 'generate' not in f])

    out = ""
    for task in easy + medium + hard:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(fr"""defaults:
      - {'500K' if task in easy else '1M500K' if task in medium else '15M'}
      - _self_
    
suite: dmc
action_repeat: 2
frame_stack: 3
nstep: {1 if 'walker' in task else 3}
task_name: {task}
{'lr: 8e-5' if 'humanoid' in task else ''}
{'trunk_dim: 100' if 'humanoid' in task else ''}
{'batch_size: 512' if 'walker' in task else ''}
    
hydra:
    job:
        env_set:
          # Environment variables for MuJoCo
          MKL_SERVICE_FORCE_INTEL: '1'
          MUJOCO_GL: 'egl'""")
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
