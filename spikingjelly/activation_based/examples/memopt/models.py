import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, functional, neuron


class ConvBNNeuron(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        preceding_avg_pool: bool = False,
        **kwargs,
    ):
        super().__init__()
        conv = [layer.AvgPool1d(2, 2)] if preceding_avg_pool else []
        conv += [
            layer.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        ]
        self.conv = nn.Sequential(*conv)

        self.bn_neuron = nn.Sequential(
            layer.BatchNorm1d(out_channels),
            neuron.LIFNode(**kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class AvgPoolFlattenLinearNeuron(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.fc = nn.Sequential(
            layer.AvgPool1d(2, 2),
            layer.Flatten(start_dim=-2),
            layer.Linear(channels * 8, channels * 8 // 4),
        )
        self.neuron = neuron.LIFNode(**kwargs)

    def forward(self, x_seq):
        return self.neuron(self.fc(x_seq))

    def __spatial_split__(self):
        return self.fc, self.neuron


class SequentialCIFARNet(nn.Module):
    def __init__(self,
        channels: int=128,
        num_classes=10,
        tau: float = 2.0,
        decay_input: bool = False,
        detach_reset: bool = True,
        surrogate_function=surrogate.ATan(),
        backend: str = "triton",
    ):
        super().__init__()
        neuron_kwargs = {
            "tau": tau,
            "decay_input": decay_input,
            "detach_reset": detach_reset,
            "surrogate_function": surrogate_function,
            "backend": backend,
            "step_mode": "m",
        }
        self.channels = channels
        self.num_classes = num_classes

        conv = []
        for i in range(2):
            for j in range(3):
                if len(conv) == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv_block = ConvBNNeuron(
                    in_channels,
                    channels,
                    preceding_avg_pool=(j == 0 and i != 0),
                    **neuron_kwargs,
                )
                conv.append(conv_block)

        self.conv = nn.Sequential(*conv)
        self.conv[0].x_compressor = "NullSpikeCompressor" # explicitly specify

        self.fc = AvgPoolFlattenLinearNeuron(channels, **neuron_kwargs)
        self.decode = nn.Linear(channels * 8 // 4, num_classes)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        # x.shape = [N, C, H, W]
        x = x.permute(3, 0, 1, 2)
        # x.shape = [T, N, Cin, L]
        y = self.conv(x)
        y = self.fc(y)  # [T, N, C']
        y = y.mean(dim=0)  # [N, C']
        y = self.decode(y)
        return y
