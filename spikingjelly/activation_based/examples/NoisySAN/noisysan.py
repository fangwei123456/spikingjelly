from collections import OrderedDict

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, layer, encoding


class NoisySpikeMLP(nn.Module):
    def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, beta, sigma_init):
        super().__init__()
        hidden_num = len(hidden_sizes)
        
        # Define Layers
        hidden_layers = OrderedDict([
            ('linear0', layer.Linear(in_pop_dim, hidden_sizes[0])),
            ('sn0', neuron.NoisyCLIFNode(hidden_sizes[0], T=spike_ts, sigma_init=sigma_init, beta=beta))
        ])
        if hidden_num > 1:
            for hidden_layer in range(1, hidden_num):
                hidden_layers['linear' + str(hidden_layer)] = layer.Linear(hidden_sizes[hidden_layer-1], hidden_sizes[hidden_layer])
                hidden_layers['sn' + str(hidden_layer)] = neuron.NoisyCLIFNode(hidden_sizes[hidden_layer], T=spike_ts, sigma_init=sigma_init, beta=beta)

        hidden_layers['linear' + str(hidden_num)] = layer.Linear(hidden_sizes[-1], act_dim * dec_pop_dim)
        hidden_layers['sn' + str(hidden_num)] = neuron.NoisyILCCLIFNode(act_dim, dec_pop_dim, T=spike_ts, sigma_init=sigma_init, beta=beta)

        self.hidden_layers = nn.Sequential(hidden_layers)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, in_pop_spikes):
        return self.hidden_layers(in_pop_spikes)

    def use_noise(self, is_training=True):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.is_training = is_training

    def reset_noise(self, num_steps):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.reset_noise(num_steps)

    def get_colored_noise(self):
        cn = []
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                cn.append(module.get_colored_noise())
        cn = torch.cat(cn, dim=1)
        return cn

    def get_colored_noise_length(self):
        length = 0
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                length += module.num_node * 2
        self.cn_length = length
        return length

    def load_colored_noise(self, cn):
        start_idx = 0
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                length = module.num_node * 2
                module.load_colored_noise(cn[:, :, start_idx:start_idx+length])
                start_idx += length

    def cancel_load(self):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.cancel_load()


class NoisyPopSpikeDecoder(nn.Module):
    def __init__(self, act_dim, pop_dim, spike_ts, beta, sigma_init):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.spike_ts = spike_ts
        self.group_fc = layer.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.decoder = neuron.NoisyNonSpikingIFNode(act_dim, T=spike_ts, sigma_init=sigma_init, beta=beta, decode='last-mem')

        functional.set_step_mode(self, step_mode='m')

    def forward(self, out_pop_spikes):
        out_pop_spikes = out_pop_spikes.view(self.spike_ts, -1, self.act_dim, self.pop_dim)
        return self.decoder(self.group_fc(out_pop_spikes).view(self.spike_ts, -1, self.act_dim))

    def use_noise(self, is_training=True):
        self.decoder.is_training = is_training

    def reset_noise(self, num_steps):
        self.decoder.reset_noise(num_steps)

    def get_colored_noise(self):
        return self.decoder.get_colored_noise()

    def get_colored_noise_length(self):
        return self.act_dim

    def load_colored_noise(self, cn):
        self.decoder.load_colored_noise(cn)

    def cancel_load(self):
        self.decoder.cancel_load()

    def get_noise_sigma(self):
        return self.decoder.sigma.mean()


class NoisyPopSpikeActor(nn.Module):
    def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes, mean_range, std, 
                 spike_ts, act_limit, beta, sigma_init):
        super().__init__()
        self.act_limit = act_limit
        
        self.encoder = encoding.PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        self.snn = NoisySpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, beta, sigma_init)
        self.decoder = NoisyPopSpikeDecoder(act_dim, dec_pop_dim, spike_ts, beta, sigma_init)

    def forward(self, obs):
        in_pop_spikes = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_spikes)
        return self.act_limit * torch.tanh(self.decoder(out_pop_spikes))

    def act(self, obs):
        self.use_noise(False)
        in_pop_spikes = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_spikes)
        action = self.act_limit * torch.tanh(self.decoder(out_pop_spikes))
        self.use_noise(True)
        return action

    def use_noise(self, is_training=True):
        self.snn.use_noise(is_training)
        self.decoder.use_noise(is_training)

    def reset_noise(self, num_steps):
        self.snn.reset_noise(num_steps)
        self.decoder.reset_noise(num_steps)

    def get_colored_noise(self):
        cn = [self.snn.get_colored_noise(), self.decoder.get_colored_noise()]
        return torch.cat(cn, dim=1).cpu().numpy()

    def get_colored_noise_length(self):
        return self.snn.get_colored_noise_length() + self.decoder.get_colored_noise_length()

    def load_colored_noise(self, cn):
        self.snn.load_colored_noise(cn[:, :, :self.snn.cn_length])
        self.decoder.load_colored_noise(cn[:, :, self.snn.cn_length:])

    def cancel_load(self):
        self.snn.cancel_load()
        self.decoder.cancel_load()

    def get_noise_sigma(self):
        return self.decoder.get_noise_sigma()