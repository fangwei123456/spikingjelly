spikingjelly.activation_based.distributed package
=================================================

The root package contains SNN-specific tensor-parallel primitives. Training
interfaces are grouped by workload: native PyTorch image training lives in
``distributed.vision`` and Megatron Core language-model training lives in
``distributed.llm``. Importing the root package does not import Megatron Core.

Tensor parallel primitives
++++++++++++++++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.tensor_parallel
    :members:
    :undoc-members:
    :show-inheritance:

Vision training
+++++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.vision
    :members:
    :undoc-members:
    :show-inheritance:

LLM training
++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.llm
    :members:
    :undoc-members:
    :show-inheritance:

LLM inference
+++++++++++++

.. automodule:: spikingjelly.activation_based.distributed.llm.inference
    :members:
    :undoc-members:
    :show-inheritance:
