"""
**API Language** - :ref:`中文 <cuda_kernel-cn>` | :ref:`English <cuda_kernel-en>`

----

.. _cuda_kernel-cn:

* **中文**

CUDA kernel 模块，提供代码生成、CuPy 神经元内核、脉冲算子和运行工具。


----

.. _cuda_kernel-en:

* **English**

CUDA kernel module providing code generation, CuPy neuron kernels, spike
operations, and runtime utilities.

The ``spike_linear`` submodule provides experimental hand-written CUDA
kernels for explicitly profiled binary SpikeLinear workloads. It does not
perform automatic density-based dispatch.
"""

from . import auto_cuda, neuron_kernel
from .spike_linear import (
    bit_pack_spike_dense,
    sparse_linear,
)

__all__ = [
    "auto_cuda",
    "bit_pack_spike_dense",
    "neuron_kernel",
    "sparse_linear",
]
