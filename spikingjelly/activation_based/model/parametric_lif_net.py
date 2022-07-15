import torch
import torch.nn as nn
from copy import deepcopy
from .. import layer
# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks https://arxiv.org/abs/2007.05785

__all__ = ['MNISTNet', 'FashionMNISTNet', 'NMNISTNet', 'CIFAR10Net', 'CIFAR10DVSNet', 'DVSGestureNet']


class MNISTNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 7 * 7, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer()
        )


    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class FashionMNISTNet(MNISTNet):
    pass


class NMNISTNet(MNISTNet):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__(channels, spiking_neuron, **kwargs)
        self.conv_fc[0] = layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.conv_fc[-6] = layer.Linear(channels * 8 * 8, 2048)

class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))

            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 2048),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x):
        return self.conv_fc(x)


class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 100),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

def test_models():
    import torch
    from .. import neuron, surrogate, functional
    x = torch.rand([2, 1, 28, 28])
    net = MNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 1, 28, 28])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 1, 28, 28])
    net = FashionMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 1, 28, 28])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 32, 32])
    net = NMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 2, 32, 32])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 3, 32, 32])
    net = CIFAR10Net(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 3, 32, 32])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = CIFAR10DVSNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = DVSGestureNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x







