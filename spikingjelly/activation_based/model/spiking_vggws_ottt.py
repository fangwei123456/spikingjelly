from copy import deepcopy

import torch.nn as nn

from .. import layer, neuron

__all__ = [
    "OTTTSpikingVGG",
    "ottt_spiking_vggws",
    "ottt_spiking_vgg11",
    "ottt_spiking_vgg11_ws",
    "ottt_spiking_vgg13",
    "ottt_spiking_vgg13_ws",
    "ottt_spiking_vgg16",
    "ottt_spiking_vgg16_ws",
    "ottt_spiking_vgg19",
    "ottt_spiking_vgg19_ws",
]

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class OTTTSpikingVGG(nn.Module):
    def __init__(
        self,
        cfg,
        weight_standardization=True,
        num_classes=1000,
        init_weights=True,
        spiking_neuron: callable = None,
        light_classifier=True,
        drop_rate=0.0,
        **kwargs,
    ):
        """
        **API Language:**
        :ref:`中文 <OTTTSpikingVGG-cn>` | :ref:`English <OTTTSpikingVGG-en>`

        ----

        .. _OTTTSpikingVGG-cn:

        * **中文**

        使用 OTTT（Online Training Through Time）训练的脉冲 VGG 网络。继承自 :class:`torchvision.models.VGG`。

        :param cfg: VGG 网络配置
        :type cfg: list
        :param weight_standardization: 是否使用权重标准化
        :type weight_standardization: bool
        :param num_classes: 分类类别数
        :type num_classes: int
        :param init_weights: 是否初始化权重
        :type init_weights: bool
        :param spiking_neuron: 脉冲神经元类
        :type spiking_neuron: callable
        :param light_classifier: 是否使用轻量分类器
        :type light_classifier: bool
        :param drop_rate: Dropout 比率
        :type drop_rate: float
        :param kwargs: 传递给父类的额外参数
        :type kwargs: dict

        ----

        .. _OTTTSpikingVGG-en:

        * **English**

        Spiking VGG network trained with OTTT (Online Training Through Time). Inherits from :class:`torchvision.models.VGG`.

        :param cfg: VGG network configuration
        :type cfg: list
        :param weight_standardization: Whether to use weight standardization
        :type weight_standardization: bool
        :param num_classes: Number of classes for classification
        :type num_classes: int
        :param init_weights: Whether to initialize weights
        :type init_weights: bool
        :param spiking_neuron: Spiking neuron class
        :type spiking_neuron: callable
        :param light_classifier: Whether to use a lightweight classifier
        :type light_classifier: bool
        :param drop_rate: Dropout rate
        :type drop_rate: float
        :param kwargs: Extra arguments for the parent class
        :type kwargs: dict
        """
        super(OTTTSpikingVGG, self).__init__()
        self.fc_hw = kwargs.get("fc_hw", 1)
        if weight_standardization:
            ws_scale = 2.74
        else:
            ws_scale = 1.0
        self.neuron = spiking_neuron
        self.features = self.make_layers(
            cfg=cfg,
            weight_standardization=weight_standardization,
            neuron=spiking_neuron,
            drop_rate=0.0,
            **kwargs,
        )
        if light_classifier:
            self.classifier = layer.OTTTSequential(
                layer.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw)),
                layer.Flatten(1),
                layer.Linear(512 * (self.fc_hw**2), num_classes),
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(
        cfg,
        weight_standardization=True,
        neuron: callable = None,
        drop_rate=0.0,
        **kwargs,
    ):
        layers = []
        in_channels = 3
        Conv2d = layer.WSConv2d if weight_standardization else layer.Conv2d
        for v in cfg:
            if v == "M":
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "A":
                layers += [layer.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, neuron(**deepcopy(kwargs))]
                if weight_standardization:
                    layers += [Scale(2.74)]
                in_channels = v
                if drop_rate > 0.0:
                    layers += [layer.Dropout(drop_rate)]
        return layer.OTTTSequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
    "S": [64, 128, "A", 256, 256, "A", 512, 512, "A", 512, 512],
}


def _spiking_vgg(
    arch, cfg, weight_standardization, spiking_neuron: callable = None, **kwargs
):
    model = OTTTSpikingVGG(
        cfg=cfgs[cfg],
        weight_standardization=weight_standardization,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def ottt_spiking_vggws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vggws-cn>` | :ref:`English <ottt_spiking_vggws-en>`

    ----

    .. _ottt_spiking_vggws-cn:

    * **中文**

    构造用于 OTTT 训练的带权重标准化 Spiking VGG。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking VGG（带权重标准化），用于 OTTT 训练
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vggws-en:

    * **English**

    Construct a weight-standardized Spiking VGG for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Weight-standardized Spiking VGG for OTTT
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vggws", "S", True, spiking_neuron, light_classifier=True, **kwargs
    )


def ottt_spiking_vgg11(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg11-cn>` | :ref:`English <ottt_spiking_vgg11-en>`

    ----

    .. _ottt_spiking_vgg11-cn:

    * **中文**

    构造用于 OTTT 训练的 Spiking VGG-11。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking VGG-11
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg11-en:

    * **English**

    Construct Spiking VGG-11 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking VGG-11
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg11", "A", False, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg11_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg11_ws-cn>` | :ref:`English <ottt_spiking_vgg11_ws-en>`

    ----

    .. _ottt_spiking_vgg11_ws-cn:

    * **中文**

    构造用于 OTTT 训练的带权重标准化 Spiking VGG-11。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: 带权重标准化的 Spiking VGG-11
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg11_ws-en:

    * **English**

    Construct a weight-standardized Spiking VGG-11 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Weight-standardized Spiking VGG-11
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg11_ws", "A", True, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg13(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg13-cn>` | :ref:`English <ottt_spiking_vgg13-en>`

    ----

    .. _ottt_spiking_vgg13-cn:

    * **中文**

    构造用于 OTTT 训练的 Spiking VGG-13。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking VGG-13
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg13-en:

    * **English**

    Construct Spiking VGG-13 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking VGG-13
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg13", "B", False, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg13_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg13_ws-cn>` | :ref:`English <ottt_spiking_vgg13_ws-en>`

    ----

    .. _ottt_spiking_vgg13_ws-cn:

    * **中文**

    构造用于 OTTT 训练的带权重标准化 Spiking VGG-13。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: 带权重标准化的 Spiking VGG-13
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg13_ws-en:

    * **English**

    Construct a weight-standardized Spiking VGG-13 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Weight-standardized Spiking VGG-13
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg13_ws", "B", True, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg16(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg16-cn>` | :ref:`English <ottt_spiking_vgg16-en>`

    ----

    .. _ottt_spiking_vgg16-cn:

    * **中文**

    构造用于 OTTT 训练的 Spiking VGG-16。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking VGG-16
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg16-en:

    * **English**

    Construct Spiking VGG-16 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking VGG-16
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg16", "D", False, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg16_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg16_ws-cn>` | :ref:`English <ottt_spiking_vgg16_ws-en>`

    ----

    .. _ottt_spiking_vgg16_ws-cn:

    * **中文**

    构造用于 OTTT 训练的带权重标准化 Spiking VGG-16。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: 带权重标准化的 Spiking VGG-16
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg16_ws-en:

    * **English**

    Construct a weight-standardized Spiking VGG-16 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Weight-standardized Spiking VGG-16
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg16_ws", "D", True, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg19(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg19-cn>` | :ref:`English <ottt_spiking_vgg19-en>`

    ----

    .. _ottt_spiking_vgg19-cn:

    * **中文**

    构造用于 OTTT 训练的 Spiking VGG-19。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: Spiking VGG-19
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg19-en:

    * **English**

    Construct Spiking VGG-19 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Spiking VGG-19
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg19", "E", False, spiking_neuron, light_classifier=False, **kwargs
    )


def ottt_spiking_vgg19_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <ottt_spiking_vgg19_ws-cn>` | :ref:`English <ottt_spiking_vgg19_ws-en>`

    ----

    .. _ottt_spiking_vgg19_ws-cn:

    * **中文**

    构造用于 OTTT 训练的带权重标准化 Spiking VGG-19。

    :param spiking_neuron: 脉冲神经元层
    :type spiking_neuron: callable
    :param kwargs: 传给 ``spiking_neuron`` 的关键字参数
    :type kwargs: dict
    :return: 带权重标准化的 Spiking VGG-19
    :rtype: torch.nn.Module

    ----

    .. _ottt_spiking_vgg19_ws-en:

    * **English**

    Construct a weight-standardized Spiking VGG-19 for OTTT training.

    :param spiking_neuron: Spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: Keyword arguments for ``spiking_neuron``
    :type kwargs: dict
    :return: Weight-standardized Spiking VGG-19
    :rtype: torch.nn.Module
    """

    return _spiking_vgg(
        "vgg19_ws", "E", True, spiking_neuron, light_classifier=False, **kwargs
    )
