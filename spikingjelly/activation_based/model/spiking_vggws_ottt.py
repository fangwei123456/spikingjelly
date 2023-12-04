import torch
import torch.nn as nn
from copy import deepcopy
from .. import functional, neuron, layer

__all__ = [
    'OTTTSpikingVGG',
    'ottt_spiking_vggws', 
    'ottt_spiking_vgg11','ottt_spiking_vgg11_ws',
    'ottt_spiking_vgg13','ottt_spiking_vgg13_ws',
    'ottt_spiking_vgg16','ottt_spiking_vgg16_ws',
    'ottt_spiking_vgg19','ottt_spiking_vgg19_ws',
]

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class OTTTSpikingVGG(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, light_classifier=True, drop_rate=0., **kwargs):
        super(OTTTSpikingVGG, self).__init__()
        self.fc_hw = kwargs.get('fc_hw', 1)
        if weight_standardization:
            ws_scale = 2.74
        else:
            ws_scale = 1.
        self.neuron = spiking_neuron
        self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization,
                                         neuron=spiking_neuron, drop_rate=0., **kwargs)
        if light_classifier:
            self.classifier = layer.OTTTSequential(
                layer.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw)),
                layer.Flatten(1),
                layer.Linear(512*(self.fc_hw**2), num_classes),
            )
        else:
            Linear = layer.WSLinear if weight_standardization else layer.Linear
            self.classifier = layer.OTTTSequential(
                layer.AdaptiveAvgPool2d((7, 7)),
                layer.Flatten(1),
                Linear(512 * 7 * 7, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                Scale(ws_scale),
                layer.Dropout(),
                Linear(4096, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                Scale(ws_scale),
                layer.Dropout(),
                layer.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: callable = None, drop_rate=0., **kwargs):
        layers = []
        in_channels = 3
        Conv2d = layer.WSConv2d if weight_standardization else layer.Conv2d
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [layer.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, neuron(**deepcopy(kwargs))]
                if weight_standardization:
                    layers += [Scale(2.74)]
                in_channels = v
                if drop_rate > 0.:
                    layers += [layer.Dropout(drop_rate)]
        return layer.OTTTSequential(*layers)




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    'S': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512],
}


def _spiking_vgg(arch, cfg, weight_standardization, spiking_neuron: callable = None, **kwargs):
    model = OTTTSpikingVGG(cfg=cfgs[cfg], weight_standardization=weight_standardization, spiking_neuron=spiking_neuron, **kwargs)
    return model



def ottt_spiking_vggws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG (sWS), model used in 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vggws', 'S', True, spiking_neuron, light_classifier=True, **kwargs)




def ottt_spiking_vgg11(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg11', 'A', False, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg11_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg11_ws', 'A', True, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg13(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-13
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg13', 'B', False, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg13_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg13_ws', 'B', True, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg16(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg16', 'D', False, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg16_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg16_ws', 'D', True, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg19(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-19
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg19', 'E', False, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg19_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with weight standardization
    :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg19_ws', 'E', True, spiking_neuron, light_classifier=False, **kwargs)


