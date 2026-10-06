Binary spike operators / 二值脉冲算子
======================================

以下接口使用根目录 ``ops`` 中的注册实现。

The following interfaces use registered implementations in root ``ops``.

``spike_linear`` 和 ``spike_conv1d/2d/3d`` 的数值计算继续由 PyTorch 分发至
CPU、cuBLAS 或 cuDNN；新反向通过 ``save_for_backward`` 保存 bool/位压缩输入，
支持重复反向，不使用全局消耗式张量缓存，也不需要导入时编译 C++ 包装层。
输入须为二值 0/1。``configure.save_bool_spike_level`` 为 0 时保存 bool，
为 1 时保存位压缩数据。字符串卷积 padding 遵循 PyTorch 原生路径。

``spike_linear`` and ``spike_conv1d/2d/3d`` keep PyTorch's CPU/cuBLAS/cuDNN
arithmetic. Their new backward saves bool/packed inputs with ``save_for_backward``,
supports repeated backward, and requires neither a destructive global tensor
cache nor an import-time C++ wrapper build. Inputs must be binary 0/1.
``configure.save_bool_spike_level=0`` saves bool tensors; level 1 saves packed
bits. String convolution padding follows the native PyTorch path.

``if_linear`` / ``lif_linear`` 保留现有 CuPy 融合 kernel 的 CUDA FP32 约束和
自定义替代梯度能力；二者分别位于 ``ops/if_linear`` 和 ``ops/lif_linear``。
``sparse_linear(strategy="sparse")`` 与 ``packed_spike_linear`` 保留现有
CuPy CUDA FP32/FP16/BF16 实现，未宣称新增 Triton 或原生 AOT 实现。

``if_linear`` / ``lif_linear`` retain the existing CuPy fused kernels' CUDA FP32
constraints and custom-surrogate support, in separate ``ops/if_linear`` and
``ops/lif_linear`` packages. ``sparse_linear(strategy="sparse")`` and
``packed_spike_linear`` retain their CuPy CUDA FP32/FP16/BF16 implementations;
no new Triton or native AOT implementation is claimed for these operations.

普通位压缩为一维、最低位优先的格式；按行打包逐行补零，二者仅在列数为 8 的倍数时
可以直接互换。解压默认返回 ``uint8``，可显式指定 dtype。
``SJ_SPIKE_COMPRESS_CUDA_IMPLEMENTATION=auto|triton|cupy`` 在启动前设置；
CPU 不依赖 GPU 可选包。此算子目前没有原生 AOT CUDA 实现，强制 ``cuda`` 会报错。

Flat packing is one-dimensional and least-significant-bit first. Row packing
zero-pads each row, so the formats are interchangeable only when the column count
is a multiple of eight. Decompression returns ``uint8`` by default and accepts an
explicit dtype. Set ``SJ_SPIKE_COMPRESS_CUDA_IMPLEMENTATION=auto|triton|cupy`` before
startup. CPU needs no optional GPU packages. This operator currently has no native
AOT CUDA implementation; forcing ``cuda`` raises an error.

.. code-block:: python

   import torch
   from spikingjelly.activation_based import functional

   x = torch.tensor([[0., 1., 1.], [1., 0., 1.]], requires_grad=True)
   weight = torch.randn(4, 3, requires_grad=True)
   y = functional.spike_linear(x, weight)
   y.sum().backward(retain_graph=True)
   y.sum().backward()
   packed = functional.bit_spike_compress(x.detach())
   restored = functional.bit_spike_decompress(packed, tuple(x.shape), x.dtype)
   assert torch.equal(restored, x.detach())

.. automodule:: spikingjelly.activation_based.functional.spike
   :members:
   :show-inheritance:
