from copy import deepcopy

import torch
import torch.nn as nn

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

from .. import layer

__all__ = [
    "SpikingResNet",
    "spiking_resnet18",
    "spiking_resnet34",
    "spiking_resnet50",
    "spiking_resnet101",
    "spiking_resnet152",
    "spiking_resnext50_32x4d",
    "spiking_resnext101_32x8d",
    "spiking_wide_resnet50_2",
    "spiking_wide_resnet101_2",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    **API Language** - :ref:`中文 <conv3x3-cn>` | :ref:`English <conv3x3-en>`

    ----

    .. _conv3x3-cn:

    * **中文**

    带 padding 的 3x3 卷积层构造函数。

    :param in_planes: 输入通道数
    :type in_planes: int
    :param out_planes: 输出通道数
    :type out_planes: int
    :param stride: 步幅，默认为 ``1``
    :type stride: int
    :param groups: 分组数，默认为 ``1``
    :type groups: int
    :param dilation: 膨胀率，默认为 ``1``
    :type dilation: int
    :return: 3x3 卷积层
    :rtype: layer.Conv2d

    ----

    .. _conv3x3-en:

    * **English**

    Construct a 3x3 convolution with padding.

    :param in_planes: Number of input channels
    :type in_planes: int
    :param out_planes: Number of output channels
    :type out_planes: int
    :param stride: Stride, default is ``1``
    :type stride: int
    :param groups: Number of groups, default is ``1``
    :type groups: int
    :param dilation: Dilation rate, default is ``1``
    :type dilation: int
    :return: 3x3 convolution layer
    :rtype: layer.Conv2d
    """
    return layer.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    **API Language** - :ref:`中文 <conv1x1-cn>` | :ref:`English <conv1x1-en>`

    ----

    .. _conv1x1-cn:

    * **中文**

    1x1 卷积层构造函数。

    :param in_planes: 输入通道数
    :type in_planes: int
    :param out_planes: 输出通道数
    :type out_planes: int
    :param stride: 步幅，默认为 ``1``
    :type stride: int
    :return: 1x1 卷积层
    :rtype: layer.Conv2d

    ----

    .. _conv1x1-en:

    * **English**

    Construct a 1x1 convolution.

    :param in_planes: Number of input channels
    :type in_planes: int
    :param out_planes: Number of output channels
    :type out_planes: int
    :param stride: Stride, default is ``1``
    :type stride: int
    :return: 1x1 convolution layer
    :rtype: layer.Conv2d
    """
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
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


class SpikingResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        """
        **API Language** - :ref:`中文 <SpikingResNet-cn>` | :ref:`English <SpikingResNet-en>`

        ----

        .. _SpikingResNet-cn:

        * **中文**

        脉冲 ResNet 网络。继承自 :class:`torchvision.models.ResNet`，将原网络的激活函数替换为脉冲神经元。

        :param block: 残差块的类型（``BasicBlock`` 或 ``Bottleneck``）
        :type block: type
        :param layers: 每个层的残差块数量
        :type layers: list
        :param num_classes: 分类任务的类别数
        :type num_classes: int
        :param zero_init_residual: 是否将最后一个 BN 初始化为零
        :type zero_init_residual: bool
        :param groups: 分组卷积的组数
        :type groups: int
        :param width_per_group: 每组的宽度
        :type width_per_group: int
        :param replace_stride_with_dilation: 是否用膨胀卷积替换步长
        :type replace_stride_with_dilation: Optional[List[bool]]
        :param norm_layer: 归一化层类型
        :type norm_layer: Optional[Callable]
        :param spiking_neuron: 脉冲神经元类
        :type spiking_neuron: callable
        :param kwargs: 传递给脉冲神经元的额外参数
        :type kwargs: dict

        ----

        .. _SpikingResNet-en:

        * **English**

        Spiking ResNet network. Inherits from :class:`torchvision.models.ResNet` with activations replaced by spiking neurons.

        :param block: Type of residual block (``BasicBlock`` or ``Bottleneck``)
        :type block: type
        :param layers: Number of residual blocks per layer
        :type layers: list
        :param num_classes: Number of classes for classification
        :type num_classes: int
        :param zero_init_residual: Whether to zero-initialize the last BN
        :type zero_init_residual: bool
        :param groups: Number of groups for grouped convolution
        :type groups: int
        :param width_per_group: Width per group
        :type width_per_group: int
        :param replace_stride_with_dilation: Replace stride with dilated convolution
        :type replace_stride_with_dilation: Optional[List[bool]]
        :param norm_layer: Normalization layer type
        :type norm_layer: Optional[Callable]
        :param spiking_neuron: Spiking neuron class
        :type spiking_neuron: callable
        :param kwargs: Extra arguments for the spiking neuron
        :type kwargs: dict
        """
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            spiking_neuron=spiking_neuron,
            **kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            spiking_neuron=spiking_neuron,
            **kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            spiking_neuron=spiking_neuron,
            **kwargs,
        )
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
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

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        spiking_neuron: callable = None,
        **kwargs,
    ):
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
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                spiking_neuron,
                **kwargs,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    spiking_neuron=spiking_neuron,
                    **kwargs,
                )
            )

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
        if self.avgpool.step_mode == "s":
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == "m":
            x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _spiking_resnet(
    arch, block, layers, pretrained, progress, spiking_neuron, **kwargs
):
    model = SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_resnet18(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnet18-cn>` | :ref:`English <spiking_resnet18-en>`

    ----

    .. _spiking_resnet18-cn:

    * **中文**

    构造 Spiking ResNet-18。

    :param pretrained: 若为 ``True``，加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnet18-en:

    * **English**

    Construct Spiking ResNet-18.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module
    """

    return _spiking_resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnet34(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnet34-cn>` | :ref:`English <spiking_resnet34-en>`

    ----

    .. _spiking_resnet34-cn:

    * **中文**

    构造 Spiking ResNet-34。

    :param pretrained: 若为 ``True``，加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnet34-en:

    * **English**

    Construct Spiking ResNet-34.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module
    """
    return _spiking_resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnet50(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnet50-cn>` | :ref:`English <spiking_resnet50-en>`

    ----

    .. _spiking_resnet50-cn:

    * **中文**

    构造 Spiking ResNet-50。

    :param pretrained: 若为 ``True``，加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnet50-en:

    * **English**

    Construct Spiking ResNet-50.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module
    """
    return _spiking_resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnet101(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnet101-cn>` | :ref:`English <spiking_resnet101-en>`

    ----

    .. _spiking_resnet101-cn:

    * **中文**

    构造 Spiking ResNet-101。

    :param pretrained: 若为 ``True``，加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnet101-en:

    * **English**

    Construct Spiking ResNet-101.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module
    """
    return _spiking_resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnet152(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnet152-cn>` | :ref:`English <spiking_resnet152-en>`

    ----

    .. _spiking_resnet152-cn:

    * **中文**

    构造 Spiking ResNet-152。

    :param pretrained: 若为 ``True``，加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnet152-en:

    * **English**

    Construct Spiking ResNet-152.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module
    """
    return _spiking_resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnext50_32x4d(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnext50_32x4d-cn>` | :ref:`English <spiking_resnext50_32x4d-en>`

    ----

    .. _spiking_resnext50_32x4d-cn:

    * **中文**

    构造 Spiking ResNeXt-50 32x4d。

    :param pretrained: 若为 ``True``, 加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnext50_32x4d-en:

    * **English**

    Construct Spiking ResNeXt-50 32x4d.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _spiking_resnet(
        "resnext50_32x4d",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_resnext101_32x8d(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_resnext101_32x8d-cn>` | :ref:`English <spiking_resnext101_32x8d-en>`

    ----

    .. _spiking_resnext101_32x8d-cn:

    * **中文**

    构造 Spiking ResNeXt-101 32x8d。

    :param pretrained: 若为 ``True``, 加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    ----

    .. _spiking_resnext101_32x8d-en:

    * **English**

    Construct Spiking ResNeXt-101 32x8d.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _spiking_resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_wide_resnet50_2(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_wide_resnet50_2-cn>` | :ref:`English <spiking_wide_resnet50_2-en>`

    ----

    .. _spiking_wide_resnet50_2-cn:

    * **中文**

    构造 Spiking Wide ResNet-50-2。

    该模型来自 `Wide Residual Networks <https://arxiv.org/pdf/1605.07146.pdf>`_
    的脉冲版本。

    :param pretrained: 若为 ``True``, 加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    ----

    .. _spiking_wide_resnet50_2-en:

    * **English**

    Construct Spiking Wide ResNet-50-2.

    This is the spiking version of `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module
    """
    kwargs["width_per_group"] = 64 * 2
    return _spiking_resnet(
        "wide_resnet50_2",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )


def spiking_wide_resnet101_2(
    pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs
):
    r"""
    **API Language** - :ref:`中文 <spiking_wide_resnet101_2-cn>` | :ref:`English <spiking_wide_resnet101_2-en>`

    ----

    .. _spiking_wide_resnet101_2-cn:

    * **中文**

    构造 Spiking Wide ResNet-101-2。

    该模型来自 `Wide Residual Networks <https://arxiv.org/pdf/1605.07146.pdf>`_
    的脉冲版本。

    :param pretrained: 若为 ``True``, 加载 ImageNet 预训练权重
    :type pretrained: bool
    :param progress: 是否显示下载进度
    :type progress: bool
    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    ----

    .. _spiking_wide_resnet101_2-en:

    * **English**

    Construct Spiking Wide ResNet-101-2.

    This is the spiking version of `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    :param pretrained: If ``True``, load ImageNet pretrained weights
    :type pretrained: bool
    :param progress: Whether to display download progress
    :type progress: bool
    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module
    """
    kwargs["width_per_group"] = 64 * 2
    return _spiking_resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        spiking_neuron,
        **kwargs,
    )
