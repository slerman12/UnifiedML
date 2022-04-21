# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate, call

import Utils

import torch
torch.backends.cudnn.benchmark = True


# Hydra conveniently and cleanly manages sys args
# Hyper-param arg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='args')
def main(arg):
    # Set seeds
    Utils.set_seeds(arg.seed)

    arg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train, test environments
    env = instantiate(arg.environment)
    generalize = instantiate(arg.environment, train=False, seed=arg.seed + 1234)

    for arg in ('obs_shape', 'action_shape', 'discrete', 'obs_spec', 'action_spec', 'evaluate_episodes', 'data_norm'):
        if hasattr(generalize, arg):
            setattr(arg, arg, getattr(generalize, arg))

    # Agent
    agent = Utils.load(arg.save_path, arg.device) if arg.load \
        else instantiate(arg.agent).to(arg.device)

    arg.train_steps += agent.step

    # Experience replay
    replay = instantiate(arg.replay)

    # Loggers
    logger = instantiate(arg.logger)

    vlogger = instantiate(arg.vlogger)

    # Start
    converged = training = False
    while True:
        # Evaluate
        if arg.evaluate_per_steps and agent.step % arg.evaluate_per_steps == 0:

            for ep in range(arg.evaluate_episodes):
                _, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                    vlog=arg.log_video)

                logger.log(logs, 'Eval')

            logger.dump_logs('Eval')

            if arg.log_video:
                vlogger.dump_vlogs(vlogs, f'{agent.step}')

        if arg.plot_per_steps and agent.step % arg.plot_per_steps == 0:
            call(arg.plotting)

        if converged:
            break

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            if arg.log_per_episodes and agent.episode % arg.log_per_episodes == 0:
                logger.log(logs, 'Train' if training else 'Seed', dump=True)

            if env.last_episode_len > arg.nstep:
                replay.add(store=True)  # Only store full episodes

        converged = agent.step >= arg.train_steps
        training = training or (agent.step > arg.seed_steps or env.offline) and len(replay) >= arg.num_workers

        # Train agent
        if training and arg.learn_per_steps and agent.step % arg.learn_per_steps == 0 or converged:

            for _ in range(arg.learn_steps_after if converged else 1):  # Additional updates after all rollouts
                logs = agent.train().learn(replay)  # Trains the agent
                if arg.log_per_episodes and agent.episode % arg.log_per_episodes == 0:
                    logger.log(logs, 'Train')

        if training and arg.save_per_steps and agent.step % arg.save_per_steps == 0 or (converged and arg.save):
            Utils.save(arg.save_path, agent, arg.agent, 'step', 'episode')

        if training and arg.load_per_steps and agent.step % arg.load_per_steps == 0:
            agent = Utils.load(arg.save_path, arg.device, agent, preserve=['step', 'episode'], distributed=True)


if __name__ == '__main__':
    main()
