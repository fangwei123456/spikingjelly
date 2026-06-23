"""
**API Language** - :ref:`中文 <model-cn>` | :ref:`English <model-en>`

----

.. _model-cn:

* **中文**

预定义SNN模型模块，包含Spikformer、SEW ResNet等模型。


----

.. _model-en:

* **English**

Pre-defined SNN model module including Spikformer, SEW ResNet, and more.
"""

from .spikformer import (
    Spikformer,
    SpikformerBlock,
    SpikformerConv2dBN,
    SpikformerConv2dBNLIF,
    SpikformerMLP,
    SpikformerPatchStem,
    spikformer_s,
    spikformer_ti,
)
from .train_classify import Trainer

__all__ = [
    "Spikformer",
    "SpikformerBlock",
    "SpikformerConv2dBN",
    "SpikformerConv2dBNLIF",
    "SpikformerMLP",
    "SpikformerPatchStem",
    "spikformer_s",
    "spikformer_ti",
    "Trainer",
]
