# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate

import Utils

import torch
torch.backends.cudnn.benchmark = True


# Hydra conveniently and cleanly manages sys args
# Hyper-param cfg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='cfg')
def main(args):
    # Set seeds
    Utils.set_seeds(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 256


    class Transform:
        def __call__(self, sample):
            return FF.to_tensor(sample) * 2 - 1


    transform = Transform()

    train_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Train', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Eval', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

    # Agent
    z_dim = 1568
    lr = 0.0002

    G = GaussianActorEnsemble([z_dim], 50, 1024, mnist_dim, 1, optim_lr=lr).to(device)
    D = EnsembleQCritic([z_dim], 50, 1024, mnist_dim, optim_lr=lr).to(device)

    # Experience replay
    replay = instantiate(args.replay)

    # Loggers
    logger = instantiate(args.logger)

    vlogger = instantiate(args.vlogger)

    # Start
    converged = False
    while True:
        # Evaluate
        if args.evaluate_per_steps and agent.step % args.evaluate_per_steps == 0:

            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                    vlog=args.log_video)

                logger.log(logs, 'Eval')

            logger.dump_logs('Eval')

            if args.log_video:
                vlogger.dump_vlogs(vlogs, f'{agent.step}')

        if converged:
            break

        if args.log_per_episodes and agent.episode % args.log_per_episodes == 0:
            logger.dump_logs('Train')

        converged = agent.step >= args.train_steps

        # Train agent
        if args.learn_per_steps and agent.step % args.learn_per_steps == 0 or converged:

            for _ in range(args.learn_steps_after if converged else 1):  # Additional updates after all rollouts
                logs = agent.train().learn(replay)  # Trains the agent
                if args.log_per_episodes and agent.episode % args.log_per_episodes == 0:
                    logger.log(logs, 'Train')


if __name__ == '__main__':
    main()
