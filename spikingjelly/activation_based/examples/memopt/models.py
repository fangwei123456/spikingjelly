import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        kernel_size,
        stride,
        padding,
        preceding_avg_pool=False,
        **kwargs,
    ):
        super().__init__()
        proj_bn = []
        if preceding_avg_pool:
            proj_bn.append(layer.AvgPool2d(2))
        proj_bn += [
            layer.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            layer.BatchNorm2d(out_plane),
        ]
        self.proj_bn = nn.Sequential(*proj_bn)
        self.neuron = neuron.LIFNode(**kwargs)

    def forward(self, x_seq):
        return self.neuron(self.proj_bn(x_seq))

    def __spatial_split__(self):
        return self.proj_bn, self.neuron


class CIFAR10DVSVGG(nn.Module):
    def __init__(
        self, dropout: float=0.25, tau: float=1.333,
        decay_input: bool=False, detach_reset: bool=True,
        surrogate_function=surrogate.ATan(), backend="triton",
    ):
        super().__init__()
        kwargs = {
            "tau": tau,
            "decay_input": decay_input,
            "detach_reset": detach_reset,
            "surrogate_function": surrogate_function,
            "backend": backend,
            "step_mode": "m",
        }
        self.features = nn.Sequential(
            VGGBlock(2, 64, 3, 1, 1, False, **kwargs),
            VGGBlock(64, 128, 3, 1, 1, False, **kwargs),
            VGGBlock(128, 256, 3, 1, 1, True, **kwargs),
            VGGBlock(256, 256, 3, 1, 1,  False, **kwargs),
            VGGBlock(256, 512, 3, 1, 1,  True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, False, **kwargs),
            VGGBlock(512, 512, 3, 1, 1,  True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1,  False, **kwargs),
            layer.AvgPool2d(2),
        )
        self.features[0].x_compressor = "NullSpikeCompressor"
        d = int(48 / 2 / 2 / 2 / 2)
        l = [nn.Dropout(dropout)] if dropout > 0 else []
        l.append(nn.Linear(512 * d * d, 10))
        self.classifier = nn.Sequential(*l)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        functional.set_step_mode(self, "m")

    def forward(self, input):
        functional.reset_net(self)
        # input.shape = [N, T, C, H, W]
        input = input.transpose(0, 1).contiguous()  # [T, N, C, H, W]
        x = self.features(input)
        x = torch.flatten(x, 2)  # [T, N, D]
        x = self.classifier(x)
        return x
