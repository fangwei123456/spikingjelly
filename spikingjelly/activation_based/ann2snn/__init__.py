r"""
**API Language** - :ref:`中文 <ann2snn-cn>` | :ref:`English <ann2snn-en>`

----

.. _ann2snn-cn:

* **中文**

ANN 到 SNN 的转换模块。提供 FX graph 路径的 :class:`FXConverter` /
:class:`FXConversionRecipe`、module tree 路径的 :class:`ModuleConverter` /
:class:`ModuleConversionRecipe`，以及 ``HookFactory``、``NeuronFactory``、
``ReLURule``、``ThresholdOptimizer`` 等可扩展组件，并附带 ``download_url``
工具函数。兼容名 :class:`Converter` 等价于 :class:`FXConverter`，
:class:`ConversionRecipe` 等价于 :class:`FXConversionRecipe`。

----

.. _ann2snn-en:

* **English**

ANN-to-SNN conversion module. Provides the FX graph :class:`FXConverter` /
:class:`FXConversionRecipe`, the module-tree :class:`ModuleConverter` /
:class:`ModuleConversionRecipe`, extensible building blocks —
:class:`HookFactory`, :class:`NeuronFactory`, :class:`ReLURule` and
:class:`ThresholdOptimizer` — and a ``download_url`` helper for fetching
pretrained models. The compatibility name :class:`Converter` is equivalent to
:class:`FXConverter`, and :class:`ConversionRecipe` is equivalent to
:class:`FXConversionRecipe`.
"""

from .converter import Converter, FXConverter, ModuleConverter
from .delay import estimate_delay_start
from .factories import HookFactory, NeuronFactory
from .modules import ChannelVoltageScaler
from .recipes import (
    ConversionRecipe,
    FXConversionRecipe,
    LocalThresholdBalancingRecipe,
    ModuleConversionRecipe,
    RateCodingRecipe,
    SpikeZIPTFQANNRecipe,
    STATransformerRecipe,
    TransformerSpikeEquivalentRecipe,
)
from .rules import ReLURule
from .threshold import ThresholdOptimizer
from .utils import download_url

__all__ = [
    "Converter",
    "FXConverter",
    "ModuleConverter",
    "ConversionRecipe",
    "FXConversionRecipe",
    "ModuleConversionRecipe",
    "RateCodingRecipe",
    "LocalThresholdBalancingRecipe",
    "SpikeZIPTFQANNRecipe",
    "STATransformerRecipe",
    "TransformerSpikeEquivalentRecipe",
    "ChannelVoltageScaler",
    "estimate_delay_start",
    "download_url",
    "ReLURule",
    "NeuronFactory",
    "HookFactory",
    "ThresholdOptimizer",
]
