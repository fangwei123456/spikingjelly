import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = [
    'SpikingVGG', 'MultiStepSpikingVGG',
    'multi_step_spiking_vgg11','multi_step_spiking_vgg11_bn','spiking_vgg11','spiking_vgg11_bn',
    'multi_step_spiking_vgg13','multi_step_spiking_vgg13_bn','spiking_vgg13','spiking_vgg13_bn',
    'multi_step_spiking_vgg16','multi_step_spiking_vgg16_bn','spiking_vgg16','spiking_vgg16_bn',
    'multi_step_spiking_vgg19','multi_step_spiking_vgg19_bn','spiking_vgg19','spiking_vgg19_bn',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

class SpikingVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, init_weights=True,
                 single_step_neuron: callable = None, **kwargs):
        super(SpikingVGG, self).__init__()
        self.features = self.make_layers(cfg=cfg, batch_norm=batch_norm,
                                         norm_layer=norm_layer, neuron=single_step_neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            single_step_neuron(**kwargs),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            single_step_neuron(**kwargs),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, batch_norm=False, norm_layer=None, neuron: callable = None, **kwargs):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), neuron(**kwargs)]
                else:
                    layers += [conv2d, neuron(**kwargs)]
                in_channels = v
        return nn.Sequential(*layers)


def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out


class MultiStepSpikingVGG(SpikingVGG):
    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, init_weights=True, T: int = None,
                 multi_step_neuron: callable = None, **kwargs):
        self.T = T
        super().__init__(cfg, batch_norm, norm_layer, num_classes, init_weights,
                 multi_step_neuron, **kwargs)

    def _forward_impl(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, self.features[0])
        else:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
            # x.shape = [N, C, H, W]
            x = self.features[0](x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)
        x_seq = sequential_forward(self.features[1:], x_seq)
        x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)
        x_seq = torch.flatten(x_seq, 2)
        x_seq = sequential_forward(self.classifier[:-1], x_seq)
        # x_seq = self.classifier[-1](x_seq.mean(0))
        x_seq = functional.seq_to_ann_forward(x_seq, self.classifier[-1])

        return x_seq


    def forward(self, x):
        """
        :param x: the input with `shape=[N, C, H, W]` or `[*, N, C, H, W]`
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        return self._forward_impl(x)



cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _spiking_vgg(arch, cfg, batch_norm, pretrained, progress, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if batch_norm:
        norm_layer = norm_layer
    else:
        norm_layer = None
    model = SpikingVGG(cfg=cfgs[cfg], batch_norm=batch_norm, norm_layer=norm_layer, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def _multi_step_spiking_vgg(arch, cfg, batch_norm, pretrained, progress, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if batch_norm:
        norm_layer = norm_layer
    else:
        norm_layer = None
    model = MultiStepSpikingVGG(cfg=cfgs[cfg], batch_norm=batch_norm, norm_layer=norm_layer, T=T, multi_step_neuron=multi_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_vgg11(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11
        :rtype: torch.nn.Module

        A spiking version of VGG-11 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11', 'A', False, pretrained, progress, None, single_step_neuron, **kwargs)


def multi_step_spiking_vgg11(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-11 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg11', 'A', False, pretrained, progress, None, T, multi_step_neuron, **kwargs)


def spiking_vgg11_bn(pretrained=False, progress=True, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11', 'A', True, pretrained, progress, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg11_bn(pretrained=False, progress=True, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg11', 'A', True, pretrained, progress, norm_layer, T, multi_step_neuron, **kwargs)

def spiking_vgg13(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-13
        :rtype: torch.nn.Module

        A spiking version of VGG-13 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg13', 'B', False, pretrained, progress, None, single_step_neuron, **kwargs)


def multi_step_spiking_vgg13(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-13 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-13 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg13', 'B', False, pretrained, progress, None, T, multi_step_neuron, **kwargs)


def spiking_vgg13_bn(pretrained=False, progress=True, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-13-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg13', 'B', True, pretrained, progress, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg13_bn(pretrained=False, progress=True, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-13 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-13-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg13', 'B', True, pretrained, progress, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_vgg16(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16
        :rtype: torch.nn.Module

        A spiking version of VGG-16 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg16', 'D', False, pretrained, progress, None, single_step_neuron, **kwargs)


def multi_step_spiking_vgg16(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-16 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-16 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg16', 'D', False, pretrained, progress, None, T, multi_step_neuron, **kwargs)


def spiking_vgg16_bn(pretrained=False, progress=True, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-16-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg16', 'D', True, pretrained, progress, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg16_bn(pretrained=False, progress=True, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-16 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-16-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg16', 'D', True, pretrained, progress, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_vgg19(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param single_step_neuron: a single-step neuron
        :type single_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-19
        :rtype: torch.nn.Module

        A spiking version of VGG-19 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg19', 'E', False, pretrained, progress, None, single_step_neuron, **kwargs)


def multi_step_spiking_vgg19(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
:param T: total time-steps
:type T: int
        :param multi_step_neuron: a multi_step neuron
        :type multi_step_neuron: callable
        :param kwargs: kwargs for `single_step_neuron`
        :type kwargs: dict
        :return: Spiking VGG-19 with norm layer
        :rtype: torch.nn.Module

        A multi-step spiking version of VGG-19 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg19', 'E', False, pretrained, progress, None, T, multi_step_neuron, **kwargs)


def spiking_vgg19_bn(pretrained=False, progress=True, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with norm layer
    :rtype: torch.nn.Module

    A spiking version of VGG-19-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg19', 'E', True, pretrained, progress, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg19_bn(pretrained=False, progress=True, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with norm layer
    :rtype: torch.nn.Module

    A multi-step spiking version of VGG-19-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg19', 'E', True, pretrained, progress, norm_layer, T, multi_step_neuron, **kwargs)

