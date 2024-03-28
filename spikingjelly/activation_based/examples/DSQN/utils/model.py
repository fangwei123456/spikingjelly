import torch
import torch.nn as nn

import math

from spikingjelly.activation_based import layer, neuron, surrogate


# Deep Reinforcement Learning with Spiking Q-learning
class DSQN(nn.Module):
    def __init__(self, input_shape, n_actions, T=5, dec_type='max-mem', use_cuda=False):
        super(DSQN, self).__init__()

        self.model_name = 'spiking_dqn'

        self.dec_type = dec_type

        if 'mem' in dec_type:
            self.network = nn.Sequential(
                layer.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Conv2d(32, 64, kernel_size=4, stride=2),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Conv2d(64, 64, kernel_size=3, stride=1),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Flatten(),
                layer.Linear(64 * 7 * 7, 512),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Linear(512, n_actions),
                neuron.NonSpikingLIFNode(decode=dec_type)
            )

        else: # fr-mlp
            self.network = nn.Sequential(
                layer.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Conv2d(32, 64, kernel_size=4, stride=2),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Conv2d(64, 64, kernel_size=3, stride=1),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

                layer.Flatten(),
                layer.Linear(64 * 7 * 7, 512),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            )

            self.decoder = nn.Linear(512, n_actions)

        self.T = T

        functional.set_step_mode(self.network, step_mode='m')
        if use_cuda:
            functional.set_backend(self.network, backend='cupy')

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        if 'mem' in self.dec_type:
            return self.network(x_seq)
        
        # fr-mlp
        x_seq = self.network(x_seq)
        fr = x_seq.mean(0)

        return self.decoder(fr)