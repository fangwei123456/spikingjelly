"""Experimental hand-written CUDA kernels for binary SpikeLinear.

``sparse_linear`` exposes a row-index sparse kernel and a cuBLAS fallback.
The slower v3 kernel remains available only as a low-level custom op for
pre-packed spike tensors. Both kernels accept contiguous CUDA FP32,
FP16, or BF16 tensors, with fake/autograd registrations. Missing native
extensions use Torch reference execution.
"""

from typing import Literal, Optional

import torch


from ..native_loader import _native_available
from ..surrogate import _DTYPES

__all__ = [
    "bit_pack_spike_dense",
    "sparse_linear",
]


# ----------------------------------------------------------------------
# Projection validation and dtype contracts
# ----------------------------------------------------------------------

_MAX_CUDA_ELEMENTS = 2**31 - 1


def _check_dtype(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype not in _DTYPES:
        raise TypeError(f"{name} must have dtype float32, float16, or bfloat16")


def _check_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> None:
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() > _MAX_CUDA_ELEMENTS:
        raise ValueError(
            f"{name} exceeds the {_MAX_CUDA_ELEMENTS}-element CUDA kernel limit"
        )


def _check_weight_bias(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _check_cuda_tensor(weight, "weight", dtype, 2)
    if weight.shape[1] != K:
        raise ValueError(f"weight.shape[1] must equal {K}, got {weight.shape[1]}")
    if weight.device != device:
        raise ValueError("spike and weight must be on the same CUDA device")
    if bias is not None:
        _check_cuda_tensor(bias, "bias", dtype, 1)
        if bias.shape[0] != weight.shape[0]:
            raise ValueError("bias.shape[0] must equal weight.shape[0]")
        if bias.device != device:
            raise ValueError("spike and bias must be on the same CUDA device")


# ----------------------------------------------------------------------
# Bit-pack helper (also used directly as a public utility)
# ----------------------------------------------------------------------


@torch.library.custom_op(
    "sj_spike_linear::pack_rows", mutates_args=(), device_types="cuda"
)
def bit_pack_spike_dense(spike: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <bit_pack_spike_dense-cn>` | :ref:`English <bit_pack_spike_dense-en>`

    ----

    .. _bit_pack_spike_dense-cn:

    * **中文**

    将二维二值 CUDA 张量按行打包为 ``uint8``。每个输出字节的第 ``i`` 位
    对应输入的第 ``b * 8 + i`` 列；末尾不足 8 位时补零。输入支持
    ``float32``、``float16`` 和 ``bfloat16``，大于 ``0.5`` 的值编码为 1。

    :param spike: 形状为 ``[M, K]`` 的连续 CUDA 张量。调用方必须保证输入满足
        二值脉冲契约
    :type spike: torch.Tensor
    :return: 形状为 ``[M, ceil(K / 8)]``、与输入位于同一设备的 ``uint8`` 张量
    :rtype: torch.Tensor
    :raises TypeError: ``spike`` 的 dtype 不受支持
    :raises ValueError: ``spike`` 不是二维、连续 CUDA 张量，或元素数超过
        CUDA kernel 的 ``int32`` 索引上限

    ----

    .. _bit_pack_spike_dense-en:

    * **English**

    Pack a two-dimensional binary CUDA tensor into row-major ``uint8`` bytes.
    Bit ``i`` of output byte ``b`` represents input column ``b * 8 + i``;
    a trailing partial byte is zero-padded. The input may use ``float32``,
    ``float16``, or ``bfloat16``. Values greater than ``0.5`` encode as 1.

    :param spike: Contiguous CUDA tensor shaped ``[M, K]``. The caller must
        guarantee the binary-spike contract
    :type spike: torch.Tensor
    :return: A ``uint8`` tensor shaped ``[M, ceil(K / 8)]`` on the input device
    :rtype: torch.Tensor
    :raises TypeError: If ``spike`` has an unsupported dtype
    :raises ValueError: If ``spike`` is not a two-dimensional contiguous CUDA
        tensor, or exceeds the CUDA kernel's ``int32`` indexing limit
    """
    if spike.dim() != 2:
        raise ValueError("spike must be 2D")
    _check_dtype(spike, "spike")
    if not spike.is_cuda:
        raise ValueError("spike must be a CUDA tensor")
    if not spike.is_contiguous():
        raise ValueError("spike must be contiguous")
    if spike.numel() > _MAX_CUDA_ELEMENTS:
        raise ValueError(
            f"spike exceeds the {_MAX_CUDA_ELEMENTS}-element CUDA kernel limit"
        )

    M, K = spike.shape
    K_PACKED = (K + 7) // 8
    out = torch.empty((M, K_PACKED), dtype=torch.uint8, device=spike.device)
    if M == 0 or K_PACKED == 0:
        return out

    if _native_available(__package__, spike.get_device()):
        return torch.ops.sj_spike_linear.kernel_pack_rows(spike)
    values = torch.nn.functional.pad(spike > 0.5, (0, K_PACKED * 8 - K))
    shifts = torch.arange(8, device=spike.device, dtype=torch.int64)
    return (
        (values.reshape(M, K_PACKED, 8).to(torch.int64) << shifts)
        .sum(-1)
        .to(torch.uint8)
    )


# ----------------------------------------------------------------------
# v3 (dense, bit-packed) — custom_op
# ----------------------------------------------------------------------


@torch.library.custom_op("sj_spike_linear::packed", mutates_args=())
def _packed_forward(
    spike_packed: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <_packed_forward-cn>` | :ref:`English <_packed_forward-en>`

    ----

    .. __packed_forward-cn:

    * **中文**

    对预先按位打包的二维脉冲执行实验性 v3 CUDA Linear kernel。低精度输入
    矩阵乘法使用 FP32 累加，并将结果转换回权重 dtype；可选偏置随后以
    输出 dtype 相加。

    :param spike_packed: 形状为 ``[M, ceil(K / 8)]`` 的连续 ``uint8`` CUDA 张量
    :type spike_packed: torch.Tensor
    :param weight: 形状为 ``[N, K]`` 的连续 CUDA 张量，dtype 为
        ``float32``、``float16`` 或 ``bfloat16``
    :type weight: torch.Tensor
    :param bias: 可选的连续 ``[N]`` CUDA 张量，dtype 和设备必须与 ``weight`` 一致
    :type bias: Optional[torch.Tensor]
    :return: 形状为 ``[M, N]``、dtype 与 ``weight`` 相同的 CUDA 张量
    :rtype: torch.Tensor
    :raises TypeError: 输入 dtype 不满足约束
    :raises ValueError: 输入 shape、连续性、设备或元素数量不满足约束

    ----

    .. __packed_forward-en:

    * **English**

    Apply the experimental v3 CUDA Linear kernel to a pre-packed binary spike
    matrix. Low-precision matrix products accumulate in FP32 and are converted
    back to the weight dtype before the optional bias is added in that dtype.

    :param spike_packed: Contiguous ``uint8`` CUDA tensor shaped
        ``[M, ceil(K / 8)]``
    :type spike_packed: torch.Tensor
    :param weight: Contiguous ``[N, K]`` CUDA tensor with ``float32``,
        ``float16``, or ``bfloat16`` dtype
    :type weight: torch.Tensor
    :param bias: Optional contiguous ``[N]`` CUDA tensor on the same device and
        with the same dtype as ``weight``
    :type bias: Optional[torch.Tensor]
    :return: CUDA tensor shaped ``[M, N]`` with the weight dtype
    :rtype: torch.Tensor
    :raises TypeError: If an input dtype violates the contract
    :raises ValueError: If an input shape, layout, device, or element count
        violates the contract
    """
    _check_cuda_tensor(spike_packed, "spike_packed", torch.uint8, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K_PACKED = spike_packed.shape
    N, K = weight.shape
    _check_dtype(weight, "weight")
    _check_weight_bias(weight, bias, K, spike_packed.device, weight.dtype)
    if K_PACKED != (K + 7) // 8:
        raise ValueError("spike_packed.shape[1] must equal ceil(weight.shape[1] / 8)")
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    if _native_available(__package__, weight.get_device()):
        Y = torch.ops.sj_spike_linear.kernel_packed(spike_packed, weight)
    else:
        shifts = torch.arange(8, device=spike_packed.device, dtype=torch.uint8)
        values = ((spike_packed.unsqueeze(-1) >> shifts) & 1).reshape(M, K_PACKED * 8)[
            :, :K
        ]
        # The CUDA kernel accumulates in FP32, then casts before adding bias.
        Y = (values.float() @ weight.float().t()).to(weight.dtype)
    if bias is not None:
        Y = Y + bias
    return Y


@torch.library.register_fake("sj_spike_linear::packed")
def __packed_forward_fake(
    spike_packed: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if spike_packed.dtype != torch.uint8:
        raise TypeError("spike_packed must have dtype uint8")
    if weight.dtype not in _DTYPES:
        raise TypeError("weight must have dtype float32, float16, or bfloat16")
    if weight.device != spike_packed.device:
        raise ValueError("spike_packed and weight must be on the same device")
    torch._check(spike_packed.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike_packed.shape[1] == (weight.shape[1] + 7) // 8,
        lambda: "packed width must equal ceil(K / 8)",
    )
    if bias is not None:
        if bias.dtype != weight.dtype:
            raise TypeError("bias must have the same dtype as weight")
        if bias.device != spike_packed.device:
            raise ValueError("spike_packed and bias must be on the same device")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike_packed.shape[0], weight.shape[0]),
        dtype=weight.dtype,
        device=spike_packed.device,
    )


def _setup_v3_context(ctx, inputs, output):
    del output
    spike_packed, weight, bias = inputs
    ctx.save_for_backward(spike_packed, weight, bias)


def _v3_backward(ctx, grad_output):
    spike_packed, weight, bias = ctx.saved_tensors
    M, K_PACKED = spike_packed.shape
    K = weight.shape[1]
    with torch.cuda.device(weight.device):
        bits = (
            spike_packed.unsqueeze(-1)
            >> torch.arange(8, dtype=torch.uint8, device=spike_packed.device)
        ) & 1
        spike = bits.reshape(M, K_PACKED * 8)[:, :K].to(grad_output.dtype)
        grad_weight = torch.mm(grad_output.t(), spike)
        grad_bias = grad_output.sum(0) if bias is not None else None
    return None, grad_weight, grad_bias


torch.library.register_autograd(
    "sj_spike_linear::packed",
    _v3_backward,
    setup_context=_setup_v3_context,
)


# ----------------------------------------------------------------------
# v15 (true-sparse, W transposed) — custom_op
# ----------------------------------------------------------------------


@torch.library.custom_op("sj_spike_linear::sparse", mutates_args=())
def _sparse_forward(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <_sparse_forward-cn>` | :ref:`English <_sparse_forward-en>`

    ----

    .. __sparse_forward-cn:

    * **中文**

    使用实验性稀疏行索引 CUDA kernel 对二维二值脉冲执行 Linear。kernel 使用
    固定容量的 ``int32 [M, K]`` 索引工作区，避免为数据相关的紧凑 CSR 缓冲区
    执行 host 同步。低精度矩阵乘法使用 FP32 累加并转换回输入 dtype；可选偏置
    随后以输出 dtype 相加。

    :param spike: 形状为 ``[M, K]`` 的连续二值 CUDA 张量，dtype 为
        ``float32``、``float16`` 或 ``bfloat16``
    :type spike: torch.Tensor
    :param weight: 与 ``spike`` 同 dtype、同设备的连续 ``[N, K]`` CUDA 张量
    :type weight: torch.Tensor
    :param bias: 可选的连续 ``[N]`` CUDA 张量，dtype 和设备与 ``spike`` 一致
    :type bias: Optional[torch.Tensor]
    :return: 形状为 ``[M, N]``、dtype 与 ``spike`` 相同的 CUDA 张量
    :rtype: torch.Tensor
    :raises TypeError: 输入 dtype 不满足约束
    :raises ValueError: 输入 shape、连续性、设备或元素数量不满足约束

    ----

    .. __sparse_forward-en:

    * **English**

    Apply Linear to a two-dimensional binary spike tensor with the experimental
    sparse row-index CUDA kernel. A fixed-capacity ``int32 [M, K]`` index
    workspace avoids the host synchronization required by a data-dependent
    compact CSR allocation. Low-precision matrix products accumulate in FP32
    and are converted back to the input dtype before the optional bias is added
    in that dtype.

    :param spike: Contiguous binary ``[M, K]`` CUDA tensor with ``float32``,
        ``float16``, or ``bfloat16`` dtype
    :type spike: torch.Tensor
    :param weight: Contiguous ``[N, K]`` CUDA tensor on the same device and with
        the same dtype as ``spike``
    :type weight: torch.Tensor
    :param bias: Optional contiguous ``[N]`` CUDA tensor on the same device and
        with the same dtype as ``spike``
    :type bias: Optional[torch.Tensor]
    :return: CUDA tensor shaped ``[M, N]`` with the spike dtype
    :rtype: torch.Tensor
    :raises TypeError: If an input dtype violates the contract
    :raises ValueError: If an input shape, layout, device, or element count
        violates the contract
    """
    _check_dtype(spike, "spike")
    _check_cuda_tensor(spike, "spike", spike.dtype, 2)
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    M, K = spike.shape
    N = weight.shape[0]
    if weight.dtype != spike.dtype:
        raise TypeError("spike and weight must have the same dtype")
    _check_weight_bias(weight, bias, K, spike.device, spike.dtype)
    if M * N > _MAX_CUDA_ELEMENTS:
        raise ValueError("output exceeds the CUDA kernel element limit")

    if _native_available(__package__, spike.get_device()):
        Y = torch.ops.sj_spike_linear.kernel_sparse(spike, weight)
    else:
        Y = ((spike > 0.5).float() @ weight.float().t()).to(spike.dtype)
    if bias is not None:
        Y = Y + bias
    return Y


@torch.library.register_fake("sj_spike_linear::sparse")
def __sparse_forward_fake(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if spike.dtype not in _DTYPES:
        raise TypeError("spike must have dtype float32, float16, or bfloat16")
    if weight.dtype != spike.dtype:
        raise TypeError("spike and weight must have the same dtype")
    if weight.device != spike.device:
        raise ValueError("spike and weight must be on the same device")
    torch._check(spike.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(
        spike.shape[1] == weight.shape[1],
        lambda: "spike and weight K dimensions must match",
    )
    if bias is not None:
        if bias.dtype != spike.dtype:
            raise TypeError("bias must have the same dtype as spike")
        if bias.device != spike.device:
            raise ValueError("spike and bias must be on the same device")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == weight.shape[0])
    return torch.empty(
        (spike.shape[0], weight.shape[0]),
        dtype=spike.dtype,
        device=spike.device,
    )


def _setup_v15_context(ctx, inputs, output):
    del output
    spike, weight, bias = inputs
    ctx.save_for_backward(spike, weight, bias)


def _v15_backward(ctx, grad_output):
    spike, weight, bias = ctx.saved_tensors
    with torch.cuda.device(weight.device):
        grad_spike = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), spike.to(grad_output.dtype))
        grad_bias = grad_output.sum(0) if bias is not None else None
    return grad_spike, grad_weight, grad_bias


torch.library.register_autograd(
    "sj_spike_linear::sparse",
    _v15_backward,
    setup_context=_setup_v15_context,
)


# ----------------------------------------------------------------------
# Public API: sparse_linear
# ----------------------------------------------------------------------


def sparse_linear(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    strategy: Literal["torch", "sparse"] = "torch",
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <sparse_linear-cn>` | :ref:`English <sparse_linear-en>`

    ----

    .. _sparse_linear-cn:

    * **中文**

    对未按位打包的二值脉冲执行 Linear。``strategy="torch"`` 直接调用
    :func:`torch.nn.functional.linear`。``strategy="sparse"`` 使用实验性稀疏
    CUDA kernel，要求二维、同设备且同 dtype 的输入，并使用
    ``4 * M * K`` 字节的 ``int32`` 索引工作区。稀疏 kernel 以 ``> 0.5``
    判断脉冲，调用方必须保证二值输入契约，因为运行时检查数值会同步设备。

    :param spike: 输入脉冲。稀疏策略要求形状为 ``[M, K]`` 的 CUDA 张量，dtype
        为 ``float32``、``float16`` 或 ``bfloat16``
    :type spike: torch.Tensor
    :param weight: 形状为 ``[N, K]`` 的权重；稀疏策略要求与 ``spike`` 同 dtype
        和设备
    :type weight: torch.Tensor
    :param bias: 可选的 ``[N]`` 偏置；稀疏策略要求与 ``spike`` 同 dtype 和设备
    :type bias: Optional[torch.Tensor]
    :param strategy: ``"torch"`` 或 ``"sparse"``，默认 ``"torch"``
    :type strategy: Literal["torch", "sparse"]
    :return: Linear 输出；稀疏策略返回 ``[M, N]`` 且保持输入 dtype
    :rtype: torch.Tensor
    :raises ValueError: ``strategy`` 未知，或稀疏策略的 shape、设备、元素数量不满足约束
    :raises TypeError: 稀疏策略的 dtype 不满足约束

    ----

    .. _sparse_linear-en:

    * **English**

    Apply Linear to an unpacked binary spike tensor. ``strategy="torch"`` calls
    :func:`torch.nn.functional.linear` directly. ``strategy="sparse"`` uses the
    experimental sparse CUDA kernel and requires two-dimensional inputs with
    matching devices and dtypes. It allocates an ``int32`` index
    workspace of ``4 * M * K`` bytes. The sparse kernel thresholds at ``> 0.5``;
    the caller must guarantee binary values because validating them would
    synchronize the device.

    :param spike: Input spikes. The sparse strategy requires a ``[M, K]`` CUDA
        tensor with ``float32``, ``float16``, or ``bfloat16`` dtype
    :type spike: torch.Tensor
    :param weight: Weight shaped ``[N, K]``. The sparse strategy requires the
        same dtype and device as ``spike``
    :type weight: torch.Tensor
    :param bias: Optional ``[N]`` bias. The sparse strategy requires the same
        dtype and device as ``spike``
    :type bias: Optional[torch.Tensor]
    :param strategy: ``"torch"`` or ``"sparse"``. Defaults to ``"torch"``
    :type strategy: Literal["torch", "sparse"]
    :return: Linear output. The sparse strategy returns ``[M, N]`` and preserves
        the input dtype
    :rtype: torch.Tensor
    :raises ValueError: If ``strategy`` is unknown, or a sparse input shape,
        device, or element count violates the contract
    :raises TypeError: If a sparse input dtype violates the contract
    """
    if strategy not in ("torch", "sparse"):
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from: 'torch', 'sparse'."
        )
    if strategy == "torch":
        return torch.nn.functional.linear(spike, weight, bias)
    return _sparse_forward(
        spike.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
    )


@torch.library.register_fake("sj_spike_linear::pack_rows")
def _pack_rows_fake(spike):
    _check_dtype(spike, "spike")
    torch._check(spike.ndim == 2, lambda: "spike must be 2D")
    torch._check(spike.device.type in ("cuda", "meta"), lambda: "spike must be CUDA")
    torch._check(spike.is_contiguous(), lambda: "spike must be contiguous")
    torch._check(
        spike.numel() <= _MAX_CUDA_ELEMENTS,
        lambda: "spike exceeds the CUDA kernel element limit",
    )
    return torch.empty(
        (spike.shape[0], (spike.shape[1] + 7) // 8),
        dtype=torch.uint8,
        device=spike.device,
    )
