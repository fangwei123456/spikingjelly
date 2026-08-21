"""
**API Language** - :ref:`中文 <model-cn>` | :ref:`English <model-en>`

----

.. _model-cn:

* **中文**

预定义 SNN 模型及其 builder。模型内部组件仍从各自子模块导入。


----

.. _model-en:

* **English**

Pre-defined SNN models and their builders. Import implementation blocks from
their respective submodules.
"""

from .maxformer import MaxFormer, maxformer_10_384
from .ms_resnet import MaxResNet, MSResNet, max_resnet18, ms_resnet18, ms_resnet34
from .qkformer import QKFormer, qkformer_10_384
from .spike_driven_transformer import SpikeDrivenTransformer, sdt_8_384
from .spikformer import (
    Spikformer,
    spikformer_cifar10,
    spikformer_s,
    spikformer_ti,
)

__all__ = [
    "MSResNet",
    "MaxFormer",
    "MaxResNet",
    "QKFormer",
    "SpikeDrivenTransformer",
    "Spikformer",
    "max_resnet18",
    "maxformer_10_384",
    "ms_resnet18",
    "ms_resnet34",
    "qkformer_10_384",
    "sdt_8_384",
    "spikformer_cifar10",
    "spikformer_s",
    "spikformer_ti",
]
