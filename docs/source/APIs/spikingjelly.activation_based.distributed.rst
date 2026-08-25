spikingjelly.activation_based.distributed package
=================================================

The root package contains SNN-specific tensor-parallel primitives. Training and
offline inference interfaces are grouped by workload: native PyTorch vision
lives in ``distributed.vision`` and Megatron Core language-model execution lives
in ``distributed.llm``. The package also provides an experimental SGLang offline
generation integration. Importing the root package imports neither optional
runtime.

Tensor parallel primitives
++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.tensor_parallel
    :members:
    :undoc-members:
    :show-inheritance:

Vision training and inference
+++++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.vision
    :members:
    :undoc-members:
    :show-inheritance:

LLM training and inference
++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.llm
    :members:
    :undoc-members:
    :show-inheritance:
