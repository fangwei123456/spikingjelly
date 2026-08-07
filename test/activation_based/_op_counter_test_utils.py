import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer, neuron


class TinySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = layer.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = layer.BatchNorm2d(4)
        self.sn = neuron.LIFNode()
        self.pool = layer.AdaptiveAvgPool2d(1)
        self.fc = layer.Linear(4, 2)
        functional.set_step_mode(self, "m")

    def forward(self, x):
        x = self.pool(self.sn(self.bn(self.conv(x))))
        return self.fc(torch.flatten(x, 2))
