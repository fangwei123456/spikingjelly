r"""SpikingJelly package-level configuration.

**API Language** - :ref:`中文 <configure-cn>` | :ref:`English <configure-en>`

----

.. _configure-cn:

* **中文**

SpikingJelly 的库级选项通过 ``SJ_*`` 环境变量配置。环境变量在本模块首次导入时读取，
因此必须在启动 Python 前设置。未设置的选项使用下列文档中给出的默认值。

.. code-block:: bash

    SJ_CUDA_THREADS=256 SJ_SAVE_DATASETS_COMPRESSED=0 python train.py

布尔环境变量遵循 PyTorch 的 ``0`` / ``1`` 风格，其中 ``0`` 表示 ``False``，
``1`` 表示 ``True``。

----

.. _configure-en:

* **English**

SpikingJelly package-level options are configured through ``SJ_*`` environment
variables. They are read when this module is first imported, so set them before
starting Python. Unset options use the defaults documented below.

.. code-block:: bash

    SJ_CUDA_THREADS=256 SJ_SAVE_DATASETS_COMPRESSED=0 python train.py

Boolean environment variables follow PyTorch's ``0`` / ``1`` convention:
``0`` means ``False`` and ``1`` means ``True``.
"""

import os


def _get_integer(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _get_boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {value!r}")
    return value == "1"


max_threads_number_for_datasets_preprocess = _get_integer(
    "SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS", 16
)
r"""
数据集预处理使用的最大线程数，由
``SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS`` 配置，默认值为 ``16``。

Maximum number of threads used for dataset preprocessing, configured by
``SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS``. The default is ``16``.
"""
if max_threads_number_for_datasets_preprocess <= 0:
    raise ValueError(
        "SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS must be a positive integer, "
        f"got {max_threads_number_for_datasets_preprocess!r}"
    )


cuda_threads = _get_integer("SJ_CUDA_THREADS", 512)
r"""
CUDA kernel 默认使用的线程数，由 ``SJ_CUDA_THREADS`` 配置，默认值为 ``512``。
建议使用 2 的幂。

Default number of threads used by CUDA kernels, configured by
``SJ_CUDA_THREADS``. The default is ``512``; a power of two is recommended.
"""
if cuda_threads <= 0:
    raise ValueError(
        f"SJ_CUDA_THREADS must be a positive integer, got {cuda_threads!r}"
    )


cuda_compiler_options = tuple(
    option.strip()
    for option in os.getenv("SJ_CUDA_COMPILER_OPTIONS", "-use_fast_math").split(",")
    if option.strip()
)
r"""
传递给 NVRTC 或 NVCC 的编译选项，由 ``SJ_CUDA_COMPILER_OPTIONS`` 以逗号分隔的
形式配置，默认值为 ``("-use_fast_math",)``。空字符串表示不传递选项。

Compiler options passed to NVRTC or NVCC, configured as a comma-separated value
in ``SJ_CUDA_COMPILER_OPTIONS``. The default is ``("-use_fast_math",)``; an empty
value produces an empty tuple.
"""


cuda_compiler_backend = os.getenv("SJ_CUDA_COMPILER_BACKEND", "nvrtc")
r"""
CuPy CUDA 编译后端，由 ``SJ_CUDA_COMPILER_BACKEND`` 配置，允许 ``nvrtc`` 或
``nvcc``，默认值为 ``nvrtc``。

CuPy CUDA compiler backend, configured by ``SJ_CUDA_COMPILER_BACKEND``. Valid
values are ``nvrtc`` and ``nvcc``; the default is ``nvrtc``.
"""
if cuda_compiler_backend not in ("nvcc", "nvrtc"):
    raise ValueError(
        "SJ_CUDA_COMPILER_BACKEND must be 'nvcc' or 'nvrtc', "
        f"got {cuda_compiler_backend!r}"
    )


save_datasets_compressed = _get_boolean("SJ_SAVE_DATASETS_COMPRESSED", True)
r"""
是否以压缩 NPZ 格式保存事件和帧，由 ``SJ_SAVE_DATASETS_COMPRESSED`` 配置，
默认值为 ``1``。压缩格式占用更少磁盘空间，但读取更慢。

Whether to save events and frames as compressed NPZ files, configured by
``SJ_SAVE_DATASETS_COMPRESSED``. The default is ``1``. Compression uses less disk
space but takes longer to read.
"""


save_spike_as_bool_in_neuron_kernel = _get_boolean(
    "SJ_SAVE_SPIKE_AS_BOOL_IN_NEURON_KERNEL", False
)
r"""
是否在神经元的 CuPy kernel 中以布尔形式保存反向传播所需的脉冲，由
``SJ_SAVE_SPIKE_AS_BOOL_IN_NEURON_KERNEL`` 配置，默认值为 ``0``。

Whether CuPy neuron kernels save spikes for backward as booleans, configured by
``SJ_SAVE_SPIKE_AS_BOOL_IN_NEURON_KERNEL``. The default is ``0``.
"""


save_bool_spike_level = _get_integer("SJ_SAVE_BOOL_SPIKE_LEVEL", 0)
r"""
布尔脉冲保存级别，由 ``SJ_SAVE_BOOL_SPIKE_LEVEL`` 配置，默认值为 ``0``。
``0`` 使用逐元素布尔值，``1`` 将每 8 个脉冲打包到一个 uint8 中。

Boolean spike storage level, configured by ``SJ_SAVE_BOOL_SPIKE_LEVEL``. The
default is ``0``. Level ``0`` stores element-wise booleans; level ``1`` packs
eight spikes into each uint8.
"""
if save_bool_spike_level not in (0, 1):
    raise ValueError(
        f"SJ_SAVE_BOOL_SPIKE_LEVEL must be 0 or 1, got {save_bool_spike_level!r}"
    )


triton_neuron_kernel_static_range_max_T = _get_integer(
    "SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T", 64
)
r"""
Triton 多步神经元 kernel 使用 ``tl.static_range`` 的最大序列长度，由
``SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T`` 配置，默认值为 ``64``。超过该值时
使用 ``tl.range``，以降低长序列的编译开销。

Maximum sequence length for which Triton multi-step neuron kernels use
``tl.static_range``, configured by
``SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T``. The default is ``64``. Larger
sequences use ``tl.range`` to reduce compilation overhead.
"""
if triton_neuron_kernel_static_range_max_T <= 0:
    raise ValueError(
        "SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T must be a positive integer, "
        f"got {triton_neuron_kernel_static_range_max_T!r}"
    )
