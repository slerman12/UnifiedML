from torch import nn


class F_ckGradientDescent(nn.Module):
    def __init__(self, optim, step_optim_per_steps=10):
        super().__init__()

        self.optim = optim

        self.step = 0
        self.step_optim_per_steps = step_optim_per_steps