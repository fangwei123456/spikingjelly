Binary spike projections / 二值脉冲投影
==========================================

普通 Linear/Conv 使用 PyTorch 层，省显存训练通过 ``memopt`` 应用压缩和 checkpoint。
旧 ``SpikeLinear/SpikeConv*`` 与 ``spike_linear/spike_conv*`` 已删除。

Use ordinary PyTorch Linear/Conv layers and ``memopt`` compression/checkpointing
for memory optimization. The legacy ``SpikeLinear/SpikeConv*`` and
``spike_linear/spike_conv*`` interfaces have been removed.

``if_linear`` / ``lif_linear`` 保留 CUDA FP32、初态梯度和自定义替代梯度语义。
原生扩展可用时执行融合前向并在反向重计算；否则使用 Torch 参考路径，后者不保证
不物化中间脉冲。``packed_spike_linear`` 和 ``sparse_linear(strategy="sparse")``
支持 CUDA FP32/FP16/BF16，以原生 kernel 或 Torch 参考执行。

``if_linear`` / ``lif_linear`` preserve CUDA FP32, initial-state gradients and
custom surrogates. A built native extension runs fused forward and rematerializes
in backward. Otherwise they use Torch reference execution, which does not promise
elimination of intermediate spikes. ``packed_spike_linear`` and
``sparse_linear(strategy="sparse")`` support CUDA FP32/FP16/BF16 using native
kernels or Torch reference execution.

普通位压缩是一维、最低位优先的格式；projection 按行打包会逐行补零，不能直接
用全局打包替代。memopt 的压缩器独立保留。普通压缩 CUDA 自动选择 Triton/Torch；
CPU 使用 Torch，不依赖 GPU 包。没有 CuPy 依赖。

Flat packing is one-dimensional, least-significant-bit first. Projection packing
zero-pads each row and cannot generally be replaced with flat packing. memopt
compressors remain independent. Ordinary packing selects Triton/Torch on CUDA
and Torch on CPU; no CuPy dependency is needed.

.. code-block:: python

   import torch
   from spikingjelly.activation_based import functional

   x = torch.tensor([[0., 1., 1.], [1., 0., 1.]])
   packed = functional.bit_spike_compress(x)
   restored = functional.bit_spike_decompress(packed, tuple(x.shape), x.dtype)
   assert torch.equal(restored, x)

.. automodule:: spikingjelly.activation_based.functional.spike
   :members:
   :show-inheritance:
