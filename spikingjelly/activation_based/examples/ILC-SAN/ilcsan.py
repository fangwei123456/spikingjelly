### Fully Spiking Actor Network with Intralayer Connections for Reinforcement Learning (TNNLS 2024) ###

from collections import OrderedDict

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer, neuron, encoding, surrogate


class SpikeMLP(nn.Module):
    def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes):
        super().__init__()
        hidden_num = len(hidden_sizes)
        
        # Define Layers
        hidden_layers = OrderedDict([
            ('Linear0', layer.Linear(in_pop_dim, hidden_sizes[0])),
            (neuron_type + '0', neuron.CLIFNode(surrogate_function=surrogate.Rect()))
        ])
        if hidden_num > 1:
            for hidden_layer in range(1, hidden_num):
                hidden_layers['Linear' + str(hidden_layer)] = layer.Linear(hidden_sizes[hidden_layer-1], hidden_sizes[hidden_layer])
                hidden_layers[neuron_type + str(hidden_layer)] = neuron.CLIFNode(surrogate_function=surrogate.Rect())

        hidden_layers['Linear' + str(hidden_num)] = layer.Linear(hidden_sizes[-1], act_dim * dec_pop_dim)
        hidden_layers[neuron_type + str(hidden_num)] = neuron.ILCCLIFNode(act_dim, dec_pop_dim, surrogate_function=surrogate.Rect())

        self.hidden_layers = nn.Sequential(hidden_layers)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, in_pop_spikes):
        return self.hidden_layers(in_pop_spikes)


class PopDecoder(nn.Module):
    """ Learnable Population Coding Decoder """
    def __init__(self, act_dim, pop_dim, spike_ts, decode='last-mem'):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.spike_ts = spike_ts
        self.decode = decode

        if decode == 'fr-mlp':
            self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        else:
            self.decoder = nn.Sequential(
                layer.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim),
                neuron.NonSpikingIFNode(decode=decode)
            )

            functional.set_step_mode(self, step_mode='m')

    def forward(self, out_pop_spikes):
        if self.decode == 'fr-mlp':
            out_pop_fr = out_pop_spikes.mean(dim=0).view(-1, self.act_dim, self.pop_dim)
            return self.decoder(out_pop_fr).view(-1, self.act_dim)

        out_pop_spikes = out_pop_spikes.view(self.spike_ts, -1, self.act_dim, self.pop_dim)
        return self.decoder(out_pop_spikes).view(-1, self.act_dim)


class PopSpikeActor(nn.Module):
    def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, encode, decode, act_limit):
        super().__init__()
        self.act_limit = act_limit

        if encode == 'pop-det':
            self.encoder = encoding.PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        elif encode == 'pop-ran':
            self.encoder = encoding.PopSpikeEncoderRandom(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        else: # 'pop-raw'
            self.encoder = encoding.PopEncoder(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        self.snn = SpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes)
        self.decoder = PopDecoder(act_dim, dec_pop_dim, spike_ts, decode)

    def forward(self, obs):
        in_pop_vals = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_vals)
        return self.act_limit * torch.tanh(self.decoder(out_pop_spikes))
