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
    spikformer_cifar10,
    spikformer_s,
    spikformer_ti,
)

__all__ = [
    "Spikformer",
    "spikformer_cifar10",
    "spikformer_s",
    "spikformer_ti",
]
