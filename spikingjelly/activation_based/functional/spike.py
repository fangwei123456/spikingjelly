from typing import Literal, Optional

import torch

from .. import surrogate

__all__ = [
    "bit_spike_compress",
    "bit_spike_decompress",
    "bit_pack_spike_dense",
    "packed_spike_linear",
    "sparse_linear",
    "if_linear",
    "lif_linear",
]


def bit_spike_compress(spike: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <registered-bit_spike_compress-cn>` | :ref:`English <registered-bit_spike_compress-en>`

    ----

    .. _registered-bit_spike_compress-cn:

    * **中文**

    将任意形状的二值输入按最低位优先打包为一维 uint8；尾部补零。CPU 使用 Torch，CUDA 自动选择 Triton/Torch。没有梯度。

    :param spike: 二值（0/1）输入；支持非连续张量，按行打包与 sparse 策略要求连续二维 CUDA FP32/FP16/BF16。
    :type spike: torch.Tensor
    :return: 压缩为一维 ceil(numel/8) 个 uint8；按行打包为 [M, ceil(K/8)] uint8；解压为 shape/dtype 指定的张量，均保持设备。
    :rtype: torch.Tensor
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-bit_spike_compress-en:

    * **English**

    Pack binary input of any shape into flat uint8, least-significant-bit first with zero-padded tails. CPU uses Torch; CUDA selects Triton/Torch. This operation is nondifferentiable.

    :param spike: Binary (0/1) input; noncontiguous tensors are supported except row packing and sparse strategy, which require contiguous 2D CUDA FP32/FP16/BF16.
    :type spike: torch.Tensor
    :return: Flat compression returns ceil(numel/8) uint8 bytes; row packing returns [M, ceil(K/8)] uint8; decompression returns the requested shape/dtype. Device is preserved.
    :rtype: torch.Tensor
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.spike_compress import _pack

    return _pack(spike)


def bit_spike_decompress(
    packed: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <registered-bit_spike_decompress-cn>` | :ref:`English <registered-bit_spike_decompress-en>`

    ----

    .. _registered-bit_spike_decompress-cn:

    * **中文**

    将一维位打包输入解压，恢复指定 shape 和 dtype；设备保持不变。CPU 使用 Torch，CUDA 自动选择 Triton/Torch。

    :param packed: 同设备 uint8 打包数据；解压要求一维，packed Linear 要求 [M, ceil(K/8)] 连续二维 CUDA 数据。
    :type packed: torch.Tensor
    :param shape: 原始形状，各维非负；元素数必须与打包字节数匹配。
    :type shape: tuple[int, ...]
    :param dtype: 解压输出 dtype，默认 uint8；编码为 0/1。 默认 ``torch.uint8``.
    :type dtype: torch.dtype
    :return: 压缩为一维 ceil(numel/8) 个 uint8；按行打包为 [M, ceil(K/8)] uint8；解压为 shape/dtype 指定的张量，均保持设备。
    :rtype: torch.Tensor
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-bit_spike_decompress-en:

    * **English**

    Unpack flat packed bits into the requested shape and dtype on the same device. CPU uses Torch; CUDA selects Triton/Torch.

    :param packed: Packed uint8 data; decompression requires 1D input, packed Linear requires contiguous 2D CUDA [M, ceil(K/8)] data.
    :type packed: torch.Tensor
    :param shape: Original shape with nonnegative dimensions; element count must match the packed byte count.
    :type shape: tuple[int, ...]
    :param dtype: Output dtype for decompression, default uint8, with values 0/1. Default: ``torch.uint8``.
    :type dtype: torch.dtype
    :return: Flat compression returns ceil(numel/8) uint8 bytes; row packing returns [M, ceil(K/8)] uint8; decompression returns the requested shape/dtype. Device is preserved.
    :rtype: torch.Tensor
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.spike_compress import _unpack

    return _unpack(packed, shape, dtype)


def bit_pack_spike_dense(spike: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <registered-bit_pack_spike_dense-cn>` | :ref:`English <registered-bit_pack_spike_dense-en>`

    ----

    .. _registered-bit_pack_spike_dense-cn:

    * **中文**

    将连续二维 CUDA FP32/FP16/BF16 二值输入逐行打包为 uint8，各行尾部单独补零。原生扩展可用时执行 CUDA kernel，否则使用 Torch。没有梯度。

    :param spike: 二值（0/1）输入；支持非连续张量，按行打包与 sparse 策略要求连续二维 CUDA FP32/FP16/BF16。
    :type spike: torch.Tensor
    :return: 压缩为一维 ceil(numel/8) 个 uint8；按行打包为 [M, ceil(K/8)] uint8；解压为 shape/dtype 指定的张量，均保持设备。
    :rtype: torch.Tensor
    :raises ValueError: 形状、设备或配置无效。
    :raises TypeError: dtype 不受支持。
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-bit_pack_spike_dense-en:

    * **English**

    Pack contiguous 2D CUDA FP32/FP16/BF16 binary input into uint8 rows, padding each row independently. Use a native CUDA kernel when built, otherwise Torch. This operation is nondifferentiable.

    :param spike: Binary (0/1) input; noncontiguous tensors are supported except row packing and sparse strategy, which require contiguous 2D CUDA FP32/FP16/BF16.
    :type spike: torch.Tensor
    :return: Flat compression returns ceil(numel/8) uint8 bytes; row packing returns [M, ceil(K/8)] uint8; decompression returns the requested shape/dtype. Device is preserved.
    :rtype: torch.Tensor
    :raises ValueError: Invalid shape, device, or configuration.
    :raises TypeError: Unsupported dtype.
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.spike_linear.sparse import bit_pack_spike_dense as pack

    return pack(spike)


def packed_spike_linear(
    packed: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <registered-packed_spike_linear-cn>` | :ref:`English <registered-packed_spike_linear-en>`

    ----

    .. _registered-packed_spike_linear-cn:

    * **中文**

    对逐行打包的二值输入执行 Linear，支持 CUDA FP32/FP16/BF16 权重。原生扩展不可用时使用 Torch 参考路径。仅权重和偏置可微。

    :param packed: 同设备 uint8 打包数据；解压要求一维，packed Linear 要求 [M, ceil(K/8)] 连续二维 CUDA 数据。
    :type packed: torch.Tensor
    :param weight: 与输入同设备的权重；Linear 形状 [N, K]，卷积形状遵循对应 PyTorch conv。
    :type weight: torch.Tensor
    :param bias: 同设备可选 [N] 偏置；None 不加偏置。 默认 ``None``.
    :type bias: Optional[torch.Tensor]
    :return: 同设备、同权重 dtype 的 Linear 输出，最后一维为 N。
    :rtype: torch.Tensor
    :raises ValueError: 形状、设备或配置无效。
    :raises TypeError: dtype 不受支持。
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-packed_spike_linear-en:

    * **English**

    Apply Linear to row-packed binary input with CUDA FP32/FP16/BF16 weights. Use Torch reference execution when the native extension is unavailable. Only weights and bias are differentiable.

    :param packed: Packed uint8 data; decompression requires 1D input, packed Linear requires contiguous 2D CUDA [M, ceil(K/8)] data.
    :type packed: torch.Tensor
    :param weight: Weight on the input device; [N, K] for Linear, or the corresponding PyTorch conv shape.
    :type weight: torch.Tensor
    :param bias: Optional [N] bias on the input device; None omits bias. Default: ``None``.
    :type bias: Optional[torch.Tensor]
    :return: Linear output on the same device and in the weight dtype, with last dimension N.
    :rtype: torch.Tensor
    :raises ValueError: Invalid shape, device, or configuration.
    :raises TypeError: Unsupported dtype.
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.spike_linear.sparse import _packed_forward

    return _packed_forward(packed, weight, bias)


def sparse_linear(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    strategy: Literal["torch", "sparse"] = "torch",
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <registered-sparse_linear-cn>` | :ref:`English <registered-sparse_linear-en>`

    ----

    .. _registered-sparse_linear-cn:

    * **中文**

    对未打包的二值输入执行 Linear。torch 策略调用 PyTorch；sparse 策略使用原生 CUDA 稀疏 kernel，未构建时使用 Torch 参考路径。支持输入、权重及偏置梯度。

    :param spike: 二值（0/1）输入；支持非连续张量，按行打包与 sparse 策略要求连续二维 CUDA FP32/FP16/BF16。
    :type spike: torch.Tensor
    :param weight: 与输入同设备的权重；Linear 形状 [N, K]，卷积形状遵循对应 PyTorch conv。
    :type weight: torch.Tensor
    :param bias: 同设备可选 [N] 偏置；None 不加偏置。 默认 ``None``.
    :type bias: Optional[torch.Tensor]
    :param strategy: torch 直接调用 PyTorch Linear；sparse 使用原生 CUDA 或 Torch 参考路径。 默认 ``'torch'``.
    :type strategy: Literal['torch', 'sparse']
    :return: 同设备、同权重 dtype 的 Linear 输出，最后一维为 N。
    :rtype: torch.Tensor
    :raises ValueError: 形状、设备或配置无效。
    :raises TypeError: dtype 不受支持。
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-sparse_linear-en:

    * **English**

    Apply Linear to unpacked binary input. The torch strategy calls PyTorch; sparse uses the native sparse CUDA kernel, with Torch reference execution when not built. Input, weight and bias gradients are supported.

    :param spike: Binary (0/1) input; noncontiguous tensors are supported except row packing and sparse strategy, which require contiguous 2D CUDA FP32/FP16/BF16.
    :type spike: torch.Tensor
    :param weight: Weight on the input device; [N, K] for Linear, or the corresponding PyTorch conv shape.
    :type weight: torch.Tensor
    :param bias: Optional [N] bias on the input device; None omits bias. Default: ``None``.
    :type bias: Optional[torch.Tensor]
    :param strategy: torch directly calls PyTorch Linear; sparse uses native CUDA or Torch reference execution. Default: ``'torch'``.
    :type strategy: Literal['torch', 'sparse']
    :return: Linear output on the same device and in the weight dtype, with last dimension N.
    :rtype: torch.Tensor
    :raises ValueError: Invalid shape, device, or configuration.
    :raises TypeError: Unsupported dtype.
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.spike_linear.sparse import sparse_linear as linear

    return linear(spike, weight, bias, strategy)


def if_linear(
    x: torch.Tensor,
    v: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    detach_reset: bool = False,
    surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    threads: Literal[128, 256, 512] = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <registered-if_linear-cn>` | :ref:`English <registered-if_linear-en>`

    ----

    .. _registered-if_linear-cn:

    * **中文**

    执行 IF/LIF 后接 Linear；已构建的原生 CUDA 融合前向不物化中间脉冲，未构建时使用 Torch 参考路径。保持输入与初态，返回输出及最终电位。支持一阶梯度，反向重新计算脉冲。

    :param x: CUDA FP32 输入 [M, K] 或 [T, M, K]，维度非空；支持非连续存储，在入口转为连续。
    :type x: torch.Tensor
    :param v: 同设备 FP32 初始电位 [M, K]，不原地修改。
    :type v: torch.Tensor
    :param weight_t: 同设备 FP32 转置权重 [K, N]；入口转为连续。
    :type weight_t: torch.Tensor
    :param bias: 同设备可选 [N] 偏置；None 不加偏置。 默认 ``None``.
    :type bias: Optional[torch.Tensor]
    :param v_threshold: 发放阈值。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 硬重置电位；None 使用软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param detach_reset: 是否分离重置分支的脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param surrogate_function: spiking=True 的逐元素阶跃替代梯度；None 使用 Sigmoid。反向使用显式 CUDA 内核；自定义替代梯度通过其 PyTorch 导数参与反向。七种内置替代梯度支持 fullgraph 编译训练，自定义替代梯度仅支持 eager 训练。 默认 ``None``.
    :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
    :param threads: CUDA block 线程数，仅允许 128、256、512。 默认 ``256``.
    :type threads: Literal[128, 256, 512]
    :return: (y, v_final)：y 形状 [M, N] 或 [T, M, N]，v_final 形状 [M, K]；均为同设备 FP32。
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: 形状、设备或配置无效。
    :raises TypeError: dtype 不受支持。
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-if_linear-en:

    * **English**

    Run IF/LIF followed by Linear. The built native CUDA fused forward avoids intermediate spikes; without the extension, use Torch reference execution. Preserve inputs and initial state; return output and final voltage. Supports first-order gradients by recomputing spikes in backward.

    :param x: CUDA FP32 input [M, K] or [T, M, K], with nonempty dimensions; noncontiguous storage is made contiguous at entry.
    :type x: torch.Tensor
    :param v: FP32 initial voltage [M, K] on the input device; not mutated.
    :type v: torch.Tensor
    :param weight_t: FP32 transposed weight [K, N] on the input device; made contiguous at entry.
    :type weight_t: torch.Tensor
    :param bias: Optional [N] bias on the input device; None omits bias. Default: ``None``.
    :type bias: Optional[torch.Tensor]
    :param v_threshold: Firing threshold. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach spike gradients in the reset branch. Default: ``False``.
    :type detach_reset: bool
    :param surrogate_function: Elementwise Heaviside surrogate with spiking=True; None selects Sigmoid. Backward uses explicit CUDA kernels; custom surrogates supply their PyTorch derivative. The seven built-in surrogates support fullgraph compiled training; custom surrogates support eager training only. Default: ``None``.
    :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
    :param threads: CUDA threads per block; one of 128, 256, or 512. Default: ``256``.
    :type threads: Literal[128, 256, 512]
    :return: (y, v_final): y has shape [M, N] or [T, M, N], v_final has shape [M, K]; both FP32 on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: Invalid shape, device, or configuration.
    :raises TypeError: Unsupported dtype.
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.if_linear import if_linear as fused

    function = surrogate.Sigmoid() if surrogate_function is None else surrogate_function
    return fused(
        x,
        v,
        weight_t,
        bias,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_function=function,
        threads=threads,
    )


def lif_linear(
    x: torch.Tensor,
    v: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    tau: float = 2.0,
    decay_input: bool = True,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    detach_reset: bool = False,
    surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    threads: Literal[128, 256, 512] = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <registered-lif_linear-cn>` | :ref:`English <registered-lif_linear-en>`

    ----

    .. _registered-lif_linear-cn:

    * **中文**

    执行 IF/LIF 后接 Linear；已构建的原生 CUDA 融合前向不物化中间脉冲，未构建时使用 Torch 参考路径。保持输入与初态，返回输出及最终电位。支持一阶梯度，反向重新计算脉冲。

    :param x: CUDA FP32 输入 [M, K] 或 [T, M, K]，维度非空；支持非连续存储，在入口转为连续。
    :type x: torch.Tensor
    :param v: 同设备 FP32 初始电位 [M, K]，不原地修改。
    :type v: torch.Tensor
    :param weight_t: 同设备 FP32 转置权重 [K, N]；入口转为连续。
    :type weight_t: torch.Tensor
    :param bias: 同设备可选 [N] 偏置；None 不加偏置。 默认 ``None``.
    :type bias: Optional[torch.Tensor]
    :param tau: 有限时间常数，大于 1，以时间步为单位。 默认 ``2.0``.
    :type tau: float
    :param decay_input: 是否衰减输入。 默认 ``True``.
    :type decay_input: bool
    :param v_threshold: 发放阈值。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 硬重置电位；None 使用软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param detach_reset: 是否分离重置分支的脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param surrogate_function: spiking=True 的逐元素阶跃替代梯度；None 使用 Sigmoid。反向使用显式 CUDA 内核；自定义替代梯度通过其 PyTorch 导数参与反向。七种内置替代梯度支持 fullgraph 编译训练，自定义替代梯度仅支持 eager 训练。 默认 ``None``.
    :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
    :param threads: CUDA block 线程数，仅允许 128、256、512。 默认 ``256``.
    :type threads: Literal[128, 256, 512]
    :return: (y, v_final)：y 形状 [M, N] 或 [T, M, N]，v_final 形状 [M, K]；均为同设备 FP32。
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: 形状、设备或配置无效。
    :raises TypeError: dtype 不受支持。
    :raises RuntimeError: 张量约束无效、所需实现不可用或算子执行失败。

    ----

    .. _registered-lif_linear-en:

    * **English**

    Run IF/LIF followed by Linear. The built native CUDA fused forward avoids intermediate spikes; without the extension, use Torch reference execution. Preserve inputs and initial state; return output and final voltage. Supports first-order gradients by recomputing spikes in backward.

    :param x: CUDA FP32 input [M, K] or [T, M, K], with nonempty dimensions; noncontiguous storage is made contiguous at entry.
    :type x: torch.Tensor
    :param v: FP32 initial voltage [M, K] on the input device; not mutated.
    :type v: torch.Tensor
    :param weight_t: FP32 transposed weight [K, N] on the input device; made contiguous at entry.
    :type weight_t: torch.Tensor
    :param bias: Optional [N] bias on the input device; None omits bias. Default: ``None``.
    :type bias: Optional[torch.Tensor]
    :param tau: Finite time constant greater than one, in time steps. Default: ``2.0``.
    :type tau: float
    :param decay_input: Whether to decay the input. Default: ``True``.
    :type decay_input: bool
    :param v_threshold: Firing threshold. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach spike gradients in the reset branch. Default: ``False``.
    :type detach_reset: bool
    :param surrogate_function: Elementwise Heaviside surrogate with spiking=True; None selects Sigmoid. Backward uses explicit CUDA kernels; custom surrogates supply their PyTorch derivative. The seven built-in surrogates support fullgraph compiled training; custom surrogates support eager training only. Default: ``None``.
    :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
    :param threads: CUDA threads per block; one of 128, 256, or 512. Default: ``256``.
    :type threads: Literal[128, 256, 512]
    :return: (y, v_final): y has shape [M, N] or [T, M, N], v_final has shape [M, K]; both FP32 on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: Invalid shape, device, or configuration.
    :raises TypeError: Unsupported dtype.
    :raises RuntimeError: Invalid tensor constraints, unavailable implementation, or execution failure.
    """
    from ..._ops.lif_linear import lif_linear as fused

    function = surrogate.Sigmoid() if surrogate_function is None else surrogate_function
    return fused(
        x,
        v,
        weight_t,
        bias,
        tau=tau,
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_function=function,
        threads=threads,
    )
