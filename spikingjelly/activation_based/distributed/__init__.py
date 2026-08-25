r"""
**API Language** - :ref:`中文 <distributed-cn>` | :ref:`English <distributed-en>`

----

.. _distributed-cn:

* **中文**

SNN 分布式训练与离线推理模块。与 Megatron Core 和 SGLang 相关的语言模型能力位于
:mod:`spikingjelly.activation_based.distributed.llm`；不依赖 Megatron 的视觉模型
和底层张量并行能力由各自子模块提供。

----

.. _distributed-en:

* **English**

Distributed SNN training and offline inference modules. Language-model
functionality backed by Megatron Core and SGLang lives in
:mod:`spikingjelly.activation_based.distributed.llm`.
Vision and low-level tensor-parallel modules do not depend on Megatron.
"""

__all__ = ["llm", "tensor_parallel", "vision"]
