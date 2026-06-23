from copy import deepcopy

import torch
import torch.nn as nn

from .. import layer

# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks https://arxiv.org/abs/2007.05785

__all__ = [
    "MNISTNet",
    "FashionMNISTNet",
    "NMNISTNet",
    "CIFAR10Net",
    "CIFAR10DVSNet",
    "DVSGestureNet",
]


class MNISTNet(nn.Module):
    r"""
    **API Language** - :ref:`中文 <MNISTNet-cn>` | :ref:`English <MNISTNet-en>`

    ----

    .. _MNISTNet-cn:

    * **中文**

    用于 MNIST 手写数字分类的 Parametric LIF 网络。

    ----

    .. _MNISTNet-en:

    * **English**

    Parametric LIF network for MNIST digit classification.
    """

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
            layer.VotingLayer(),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class FashionMNISTNet(MNISTNet):
    r"""
    **API Language** - :ref:`中文 <FashionMNISTNet-cn>` | :ref:`English <FashionMNISTNet-en>`

    ----

    .. _FashionMNISTNet-cn:

    * **中文**

    用于 Fashion-MNIST 分类的 Parametric LIF 网络。即 :class:`MNISTNet` 的别名。
    :param args: 与 :class:`MNISTNet` 相同的参数
    :type args: tuple
    :param kwargs: 与 :class:`MNISTNet` 相同的关键字参数
    :type kwargs: dict

    ----

    .. _FashionMNISTNet-en:

    * **English**

    Parametric LIF network for Fashion-MNIST classification. Alias of :class:`MNISTNet`.
    :param args: Same as :class:`MNISTNet`
    :type args: tuple
    :param kwargs: Same as :class:`MNISTNet`
    :type kwargs: dict
    """

    pass


class NMNISTNet(MNISTNet):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        """
        **API Language** - :ref:`中文 <NMNISTNet-cn>` | :ref:`English <NMNISTNet-en>`

        ----

        .. _NMNISTNet-cn:

        * **中文**

        用于 N-MNIST 事件流分类的 Parametric LIF 网络。基于 :class:`MNISTNet`，将首层卷积输入通道调整为 2。

        :param channels: 卷积层的通道数
        :type channels: int
        :param spiking_neuron: 脉冲神经元类
        :type spiking_neuron: callable

        ----

        .. _NMNISTNet-en:

        * **English**

        Parametric LIF network for N-MNIST event stream classification. Based on :class:`MNISTNet` with first conv layer adjusted to 2 input channels.

        :param channels: Number of channels in conv layers
        :type channels: int
        :param spiking_neuron: Spiking neuron class
        :type spiking_neuron: callable
        """
        super().__init__(channels, spiking_neuron, **kwargs)
        self.conv_fc[0] = layer.Conv2d(
            2, channels, kernel_size=3, padding=1, bias=False
        )
        self.conv_fc[-6] = layer.Linear(channels * 8 * 8, 2048)


class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, spiking_neuron: callable = None, **kwargs):
        """
        **API Language** - :ref:`中文 <CIFAR10Net-cn>` | :ref:`English <CIFAR10Net-en>`

        ----

        .. _CIFAR10Net-cn:

        * **中文**

        用于 CIFAR-10 分类的 Parametric LIF 网络。

        ----

        .. _CIFAR10Net-en:

        * **English**

        Parametric LIF network for CIFAR-10 classification.
        """
        super().__init__()

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv.append(
                    layer.Conv2d(
                        in_channels, channels, kernel_size=3, padding=1, bias=False
                    )
                )
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
            layer.VotingLayer(10),
        )

    def forward(self, x):
        return self.conv_fc(x)


class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        """
        **API Language** - :ref:`中文 <CIFAR10DVSNet-cn>` | :ref:`English <CIFAR10DVSNet-en>`

        ----

        .. _CIFAR10DVSNet-cn:

        * **中文**

        用于 CIFAR10-DVS 事件流分类的 Parametric LIF 网络。

        ----

        .. _CIFAR10DVSNet-en:

        * **English**

        Parametric LIF network for CIFAR10-DVS event stream classification.
        """
        super().__init__()

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
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
            layer.VotingLayer(10),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        """
        **API Language** - :ref:`中文 <DVSGestureNet-cn>` | :ref:`English <DVSGestureNet-en>`

        ----

        .. _DVSGestureNet-cn:

        * **中文**

        用于 DVS128 Gesture 手势识别的 Parametric LIF 网络。

        ----

        .. _DVSGestureNet-en:

        * **English**

        Parametric LIF network for DVS128 Gesture recognition.
        """
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
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
            layer.VotingLayer(10),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


def test_models():
    import torch

    from .. import functional, neuron, surrogate

    x = torch.rand([2, 1, 28, 28])
    net = MNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 1, 28, 28])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 1, 28, 28])
    net = FashionMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 1, 28, 28])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 32, 32])
    net = NMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 2, 32, 32])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 3, 32, 32])
    net = CIFAR10Net(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 3, 32, 32])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = CIFAR10DVSNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = DVSGestureNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x
