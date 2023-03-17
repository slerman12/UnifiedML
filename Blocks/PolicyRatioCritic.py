from Blocks.Critics import EnsembleQCritic


# Note: Perhaps Critic or Creator need to track agent.step. Or PRO critic defined in actor or even creator
# Creator = Critic (or subsumes it) : perhaps just use goodness in Q learning and PGA
# Or have creator or actor track step and do as below e.g. agent.explore_schedule.step(). Actor can have its own creator
# Or no creator. Just policy.
# Or creator can set a Pi attribute in critic
# Or actor can track step via agent.explore_schedule.step(), and pass in actor and creator below; keep them separate
# To avoid cyclical loop have to call probs with as_ensemble=False (note: log_probs)
class PolicyRatioCritic(EnsembleQCritic):
    """
    PRO critic, employs ensemble Q learning via policy ratio, A.K.A Proportionality,
    returns a Normal distribution over the ensemble.
    """
    def __init__(self, actor, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2, l2_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):
        super().__init__(repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size * 2, l2_norm, discrete)

        PRO_heads = []

        for i in range(ensemble_size * 2, step=2):
            class PRO(nn.Module):  # TODO discrete actions
                def forward(self, obs, action, context=None):
                    M, B = super().Q_head(obs, action, context)
                    return M.abs() * actor(obs).log_prob(action) + B  # <- actor needs step
            PRO_heads.append(PRO())

        self.Q_head = PRO_heads

        self.init(optim_lr, target_tau)


# class DuelingEnsembleQCritic(EnsembleQCritic):
#     """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""
#
#     def __init__(self,
#                  input_channels,
#                  output_size,
#                  pixels=30,
#                  n_atoms=51,
#                  hidden_size=256,
#                  grad_scale=2 ** (-1 / 2),
#                  noisy=0,
#                  std_init=0.1):
#         super().__init__()
#         if noisy:
#             self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
#                             NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
#                             NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
#                             NoisyLinear(hidden_size, n_atoms, std_init=std_init)
#                             ]
#         else:
#             self.linears = [nn.Linear(pixels * input_channels, hidden_size),
#                             nn.Linear(hidden_size, output_size * n_atoms),
#                             nn.Linear(pixels * input_channels, hidden_size),
#                             nn.Linear(hidden_size, n_atoms)
#                             ]
#         self.advantage_layers = [nn.Flatten(-3, -1),
#                                  self.linears[0],
#                                  nn.ReLU(),
#                                  self.linears[1]]
#         self.value_layers = [nn.Flatten(-3, -1),
#                              self.linears[2],
#                              nn.ReLU(),
#                              self.linears[3]]
#         self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
#         self.advantage_out = self.advantage_layers[3]
#         self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
#         self.value = nn.Sequential(*self.value_layers)
#         self.network = self.advantage_hidden
#         self._grad_scale = grad_scale
#         self._output_size = output_size
#         self._n_atoms = n_atoms
#
#     def forward(self, input):
#         x = scale_grad(input, self._grad_scale)
#         advantage = self.advantage(x)
#         value = self.value(x).view(-1, 1, self._n_atoms)
#         return value + (advantage - advantage.mean(dim=1, keepdim=True))
#
#     def advantage(self, input):
#         x = self.advantage_hidden(input)
#         x = self.advantage_out(x)
#         x = x.view(-1, self._output_size, self._n_atoms)
#         return x + self.advantage_bias
#
#     def reset_noise(self):
#         for module in self.linears:
#             module.reset_noise()
#
#     def set_sampling(self, sampling):
#         for module in self.linears:
#             module.sampling = sampling