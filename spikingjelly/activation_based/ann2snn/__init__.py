r"""
**API Language** - :ref:`中文 <ann2snn-cn>` | :ref:`English <ann2snn-en>`

----

.. _ann2snn-cn:

* **中文**

ANN 到 SNN 的转换模块。提供 :class:`Converter` 转换器、转换
``Recipe``，以及 ``HookFactory``、``NeuronFactory``、``ReLURule``、
``ThresholdOptimizer`` 等可扩展组件，并附带 ``download_url`` 工具函数。

----

.. _ann2snn-en:

* **English**

ANN-to-SNN conversion module. Provides the :class:`Converter` driver,
conversion ``Recipe`` classes, extensible building blocks —
:class:`HookFactory`, :class:`NeuronFactory`, :class:`ReLURule` and
:class:`ThresholdOptimizer` — and a ``download_url`` helper for fetching
pretrained models.
"""

from .converter import Converter
from .delay import estimate_delay_start
from .factories import HookFactory, NeuronFactory
from .modules import ChannelVoltageScaler
from .recipes import (
    ConversionRecipe,
    LocalThresholdBalancingRecipe,
    RateCodingRecipe,
    SpikeZIPTFRecipe,
    STATransformerRecipe,
    TransformerSpikeEquivalentRecipe,
)
from .rules import ReLURule
from .threshold import ThresholdOptimizer
from .utils import download_url

__all__ = [
    "Converter",
    "ConversionRecipe",
    "RateCodingRecipe",
    "LocalThresholdBalancingRecipe",
    "SpikeZIPTFRecipe",
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
