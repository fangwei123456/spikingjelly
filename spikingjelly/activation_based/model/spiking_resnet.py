import torch
import torch.nn as nn
from .. import functional
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['SpikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2',

           'MultiStepSpikingResNet', 'multi_step_spiking_resnet18', 'multi_step_spiking_resnet34', 'multi_step_spiking_resnet50', 'multi_step_spiking_resnet101',
           'multi_step_spiking_resnet152', 'multi_step_spiking_resnext50_32x4d', 'multi_step_spiking_resnext101_32x8d',
           'multi_step_spiking_wide_resnet50_2', 'multi_step_spiking_wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class MultiStepBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, multi_step_neuron: callable = None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer, multi_step_neuron, **kwargs)

    def forward(self, x_seq):
        identity = x_seq

        out = functional.seq_to_ann_forward(x_seq, [self.conv1, self.bn1])
        out = self.sn1(out)

        out = functional.seq_to_ann_forward(out, [self.conv2, self.bn2])

        if self.downsample is not None:
            identity = functional.seq_to_ann_forward(x_seq, self.downsample)

        out += identity
        out = self.sn2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = single_step_neuron(**kwargs)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out


class MultiStepBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, multi_step_neuron: callable = None, **kwargs):
        super(MultiStepBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer, multi_step_neuron, **kwargs)

    def forward(self, x_seq):
        identity = x_seq

        out = functional.seq_to_ann_forward(x_seq, [self.conv1, self.bn1])
        out = self.sn1(out)

        out = functional.seq_to_ann_forward(out, [self.conv2, self.bn2])
        out = self.sn2(out)

        out = functional.seq_to_ann_forward(out, [self.conv3, self.bn3])

        if self.downsample is not None:
            identity = functional.seq_to_ann_forward(x_seq, self.downsample)

        out += identity
        out = self.sn3(out)

        return out


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = single_step_neuron(**kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], single_step_neuron=single_step_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], single_step_neuron=single_step_neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, single_step_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, single_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, single_step_neuron=single_step_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class MultiStepSpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T: int = None, multi_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = multi_step_neuron(**kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], multi_step_neuron=multi_step_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], multi_step_neuron=multi_step_neuron,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], multi_step_neuron=multi_step_neuron,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], multi_step_neuron=multi_step_neuron,
                                       **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_step_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, multi_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, multi_step_neuron=multi_step_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        else:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
            # x.shape = [N, C, H, W]
            x = self.conv1(x)
            x = self.bn1(x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)

        x_seq = self.sn1(x_seq)
        x_seq = functional.seq_to_ann_forward(x_seq, self.maxpool)

        x_seq = self.layer1(x_seq)
        x_seq = self.layer2(x_seq)
        x_seq = self.layer3(x_seq)
        x_seq = self.layer4(x_seq)

        x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)

        x_seq = torch.flatten(x_seq, 2)
        # x_seq = self.fc(x_seq.mean(0))
        x_seq = functional.seq_to_ann_forward(x_seq, self.fc)
        return x_seq

    def forward(self, x):
        """
        :param x: the input with `shape=[N, C, H, W]` or `[*, N, C, H, W]`
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        return self._forward_impl(x)



def _spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
    model = SpikingResNet(block, layers, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def _multi_step_spiking_resnet(arch, block, layers, pretrained, progress, T, multi_step_neuron, **kwargs):
    model = MultiStepSpikingResNet(block, layers, T=T, multi_step_neuron=multi_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_resnet18(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    A spiking version of ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, single_step_neuron, **kwargs)


def multi_step_spiking_resnet18(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _multi_step_spiking_resnet('resnet18', MultiStepBasicBlock, [2, 2, 2, 2], pretrained, progress, T, multi_step_neuron, **kwargs)

def spiking_resnet34(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    A spiking version of ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnet34(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _multi_step_spiking_resnet('resnet34', MultiStepBasicBlock, [3, 4, 6, 3], pretrained, progress, T, multi_step_neuron, **kwargs)

def spiking_resnet50(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    A spiking version of ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnet50(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _multi_step_spiking_resnet('resnet50', MultiStepBottleneck, [3, 4, 6, 3], pretrained, progress, T, multi_step_neuron, **kwargs)

def spiking_resnet101(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    A spiking version of ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnet101(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _multi_step_spiking_resnet('resnet101', MultiStepBottleneck, [3, 4, 23, 3], pretrained, progress, T, multi_step_neuron, **kwargs)

def spiking_resnet152(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    A spiking version of ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnet152(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _multi_step_spiking_resnet('resnet152', MultiStepBottleneck, [3, 8, 36, 3], pretrained, progress, T, multi_step_neuron, **kwargs)

def spiking_resnext50_32x4d(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnext50_32x4d(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _multi_step_spiking_resnet('resnext50_32x4d', MultiStepBottleneck, [3, 4, 6, 3], pretrained, progress, T,  multi_step_neuron, **kwargs)


def spiking_resnext101_32x8d(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_resnext101_32x8d(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    A multi-step spiking version of ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _multi_step_spiking_resnet('resnext101_32x8d', MultiStepBottleneck, [3, 4, 23, 3], pretrained, progress, T,  multi_step_neuron, **kwargs)

def spiking_wide_resnet50_2(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_wide_resnet50_2(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    A multi-step spiking version of Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _multi_step_spiking_resnet('wide_resnet50_2', MultiStepBottleneck, [3, 4, 6, 3], pretrained, progress, T,  multi_step_neuron, **kwargs)

def spiking_wide_resnet101_2(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param single_step_neuron: a single step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

def multi_step_spiking_wide_resnet101_2(pretrained=False, progress=True, T: int = None, multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    A multi-step spiking version of Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _multi_step_spiking_resnet('wide_resnet101_2', MultiStepBottleneck, [3, 4, 23, 3], pretrained, progress, T,  multi_step_neuron, **kwargs)