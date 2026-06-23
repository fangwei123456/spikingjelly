"""
**API Language** - :ref:`中文 <cuda_kernel-cn>` | :ref:`English <cuda_kernel-en>`

----

.. _cuda_kernel-cn:

* **中文**

CUDA kernel模块，提供GPU加速的神经元计算后端。


----

.. _cuda_kernel-en:

* **English**

CUDA kernel module providing GPU-accelerated neuron computation backends.
"""

from . import auto_cuda  # noqa
from .neuron_kernel import (
    multistep_eif_ptt,
    multistep_if_ptt,
    multistep_izhikevich_ptt,
    multistep_lif_ptt,
    multistep_plif_ptt,
    multistep_qif_ptt,
    save_cuda_codes,
)

__all__ = [
    "auto_cuda",
    "save_cuda_codes",
    "multistep_if_ptt",
    "multistep_lif_ptt",
    "multistep_plif_ptt",
    "multistep_qif_ptt",
    "multistep_izhikevich_ptt",
    "multistep_eif_ptt",
]
