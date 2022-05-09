import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import functional, layer
# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks https://arxiv.org/abs/2007.05785

__all__ = ['MNISTNet', 'FashionMNISTNet', 'NMNISTNet', 'CIFAR10Net', 'CIFAR10DVSNet', 'DVSGestureNet',
           'MultiStepMNISTNet', 'MultiStepFashionMNISTNet', 'MultiStepNMNISTNet', 'MultiStepCIFAR10Net', 'MultiStepCIFAR10DVSNet', 'MultiStepDVSGestureNet'
           ]


class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10):
        super().__init__()
        self.voting_size = voting_size

    def forward(self, x: torch.Tensor):
        y = F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)
        return y


class MNISTNet(nn.Module):
    def __init__(self, channels=128, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = single_step_neuron(**kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = single_step_neuron(**kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dp3 = layer.Dropout(0.5)
        self.fc3 = nn.Linear(channels * 7 * 7, 2048)
        self.sn3 = single_step_neuron(**kwargs)

        self.dp4 = layer.Dropout(0.5)
        self.fc4 = nn.Linear(2048, 100)
        self.sn4 = single_step_neuron(**kwargs)
        self.voting = VotingLayer(10)

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)
        x = self.pool2(x)

        x = x.flatten(1)

        x = self.dp3(x)
        x = self.fc3(x)
        x = self.sn3(x)

        x = self.dp4(x)
        x = self.fc4(x)
        x = self.sn4(x)

        x = self.voting(x)

        return x


class FashionMNISTNet(MNISTNet):
    pass


class NMNISTNet(MNISTNet):
    def __init__(self, channels=128, single_step_neuron: callable = None, **kwargs):
        super().__init__(channels, single_step_neuron, **kwargs)
        del self.conv1, self.fc3
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.fc3 = nn.Linear(channels * 8 * 8, 2048)


class MultiStepMNISTNet(MNISTNet):
    def __init__(self, channels=128, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.dp3, self.dp4

        self.dp3 = layer.MultiStepDropout(0.5)
        self.dp4 = layer.MultiStepDropout(0.5)

    def forward(self, x: torch.Tensor, T:int):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(T, 1, 1, 1, 1)

        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.pool1, self.conv2, self.bn2])

        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, self.pool2)
        x = x.flatten(2)
        x = self.dp3(x)
        x = functional.seq_to_ann_forward(x, self.fc3)
        x = self.sn3(x)

        x = self.dp4(x)
        x = functional.seq_to_ann_forward(x, self.fc4)
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, self.voting)

        return x


class MultiStepFashionMNISTNet(MultiStepMNISTNet):
    pass


class MultiStepNMNISTNet(MultiStepMNISTNet):
    def __init__(self, channels=128, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.conv1, self.fc3
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.fc3 = nn.Linear(channels * 8 * 8, 2048)

    def forward(self, x: torch.Tensor):
        x = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.pool1, self.conv2, self.bn2])

        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, self.pool2)
        x = x.flatten(2)
        x = self.dp3(x)
        x = functional.seq_to_ann_forward(x, self.fc3)
        x = self.sn3(x)

        x = self.dp4(x)
        x = functional.seq_to_ann_forward(x, self.fc4)
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, self.voting)

        return x

class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = single_step_neuron(**kwargs)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = single_step_neuron(**kwargs)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sn3 = single_step_neuron(**kwargs)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)
        self.sn4 = single_step_neuron(**kwargs)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(channels)
        self.sn5 = single_step_neuron(**kwargs)

        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(channels)
        self.sn6 = single_step_neuron(**kwargs)

        self.pool6 = nn.MaxPool2d(2, 2)






        self.dp7 = layer.Dropout(0.5)
        self.fc7 = nn.Linear(channels * 8 * 8, 2048)
        self.sn7 = single_step_neuron(**kwargs)

        self.dp8 = layer.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 100)
        self.sn8 = single_step_neuron(**kwargs)
        self.voting = VotingLayer(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        x = self.pool6(x)

        x = x.flatten(1)

        x = self.dp7(x)
        x = self.fc7(x)
        x = self.sn7(x)

        x = self.dp8(x)
        x = self.fc8(x)
        x = self.sn8(x)

        x = self.voting(x)
        return x


class MultiStepCIFAR10Net(CIFAR10Net):
    def __init__(self, channels=256, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.dp7, self.dp8
        self.dp7 = layer.MultiStepDropout(0.5)
        self.dp8 = layer.MultiStepDropout(0.5)


    def forward(self, x: torch.Tensor, T: int):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(T, 1, 1, 1, 1)

        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.conv2, self.bn2])
        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, [self.conv3, self.bn3])
        x = self.sn3(x)

        x = functional.seq_to_ann_forward(x, [self.pool3, self.conv4, self.bn4])
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, [self.conv5, self.bn5])
        x = self.sn5(x)

        x = functional.seq_to_ann_forward(x, [self.conv6, self.bn6])
        x = self.sn6(x)
        x = functional.seq_to_ann_forward(x, self.pool6)

        x = x.flatten(2)

        x = self.dp7(x)
        x = functional.seq_to_ann_forward(x, self.fc7)
        x = self.sn7(x)

        x = self.dp8(x)
        x = functional.seq_to_ann_forward(x, self.fc8)
        x = self.sn8(x)

        x = functional.seq_to_ann_forward(x, self.voting)
        return x


class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, single_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = single_step_neuron(**kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = single_step_neuron(**kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sn3 = single_step_neuron(**kwargs)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(channels)
        self.sn4 = single_step_neuron(**kwargs)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.dp5 = layer.Dropout(0.5)
        self.fc5 = nn.Linear(channels * 8 * 8, 512)
        self.sn5 = single_step_neuron(**kwargs)

        self.dp6 = layer.Dropout(0.5)
        self.fc6 = nn.Linear(512, 100)
        self.sn6 = single_step_neuron(**kwargs)
        self.voting = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sn4(x)
        x = self.pool4(x)

        x = x.flatten(1)
        x = self.dp5(x)
        x = self.fc5(x)
        x = self.sn5(x)

        x = self.dp6(x)
        x = self.fc6(x)
        x = self.sn6(x)
        x = self.voting(x)

        return x


class MultiStepCIFAR10DVSNet(CIFAR10DVSNet):
    def __init__(self, channels=128, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.dp5, self.dp6
        self.dp5 = layer.MultiStepDropout(0.5)
        self.dp6 = layer.MultiStepDropout(0.5)

    def forward(self, x: torch.Tensor):
        x = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.pool1, self.conv2, self.bn2])
        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, [self.pool2, self.conv3, self.bn3])
        x = self.sn3(x)

        x = functional.seq_to_ann_forward(x, [self.pool3, self.conv4, self.bn4])
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, self.pool4)

        x = x.flatten(2)
        x = self.dp5(x)
        x = functional.seq_to_ann_forward(x, self.fc5)
        x = self.sn5(x)

        x = self.dp6(x)
        x = functional.seq_to_ann_forward(x, self.fc6)
        x = self.sn6(x)
        x = functional.seq_to_ann_forward(x, self.voting)

        return x


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, single_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = single_step_neuron(**kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = single_step_neuron(**kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sn3 = single_step_neuron(**kwargs)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(channels)
        self.sn4 = single_step_neuron(**kwargs)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(channels)
        self.sn5 = single_step_neuron(**kwargs)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.dp6 = layer.Dropout(0.5)
        self.fc6 = nn.Linear(channels * 4 * 4, 512)
        self.sn6 = single_step_neuron(**kwargs)

        self.dp7 = layer.Dropout(0.5)
        self.fc7 = nn.Linear(512, 110)
        self.sn7 = single_step_neuron(**kwargs)
        self.voting = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sn4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)
        x = self.pool5(x)

        x = x.flatten(1)
        x = self.dp6(x)
        x = self.fc6(x)
        x = self.sn6(x)

        x = self.dp7(x)
        x = self.fc7(x)
        x = self.sn7(x)
        x = self.voting(x)

        return x


class MultiStepDVSGestureNet(DVSGestureNet):
    def __init__(self, channels=128, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.dp6, self.dp7
        self.dp6 = layer.MultiStepDropout(0.5)
        self.dp7 = layer.MultiStepDropout(0.5)

    def forward(self, x: torch.Tensor):
        x = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.pool1, self.conv2, self.bn2])
        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, [self.pool2, self.conv3, self.bn3])
        x = self.sn3(x)

        x = functional.seq_to_ann_forward(x, [self.pool3, self.conv4, self.bn4])
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, [self.pool4, self.conv5, self.bn5])
        x = self.sn5(x)
        x = functional.seq_to_ann_forward(x, self.pool5)

        x = x.flatten(2)
        x = self.dp6(x)
        x = functional.seq_to_ann_forward(x, self.fc6)
        x = self.sn6(x)

        x = self.dp7(x)
        x = functional.seq_to_ann_forward(x, self.fc7)
        x = self.sn7(x)
        x = functional.seq_to_ann_forward(x, self.voting)

        return x



def test_models():
    from .. import neuron, surrogate
    x = torch.rand([2, 1, 28, 28])
    net = MNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 1, 28, 28])
    net = FashionMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 2, 32, 32])
    net = NMNISTNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 3, 32, 32])
    net = CIFAR10Net(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = CIFAR10DVSNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 2, 128, 128])
    net = DVSGestureNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x


    # muilti step
    x = torch.rand([2, 1, 28, 28])
    net = MultiStepMNISTNet(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x, 4).shape)
    del net
    del x

    x = torch.rand([2, 1, 28, 28])
    net = MultiStepFashionMNISTNet(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x, 4).shape)
    del net
    del x

    x = torch.rand([4, 2, 2, 32, 32])
    net = MultiStepNMNISTNet(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([2, 3, 32, 32])
    net = MultiStepCIFAR10Net(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x, 4).shape)
    del net
    del x

    x = torch.rand([4, 2, 2, 128, 128])
    net = MultiStepCIFAR10DVSNet(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x

    x = torch.rand([4, 2, 2, 128, 128])
    net = MultiStepDVSGestureNet(16, neuron.MultiStepIFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    del net
    del x







