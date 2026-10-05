from __future__ import annotations

import torch

from spikingjelly.logger import logger

from ..layout import _empty_like, _layout_args
from ..triton_layout import _spatial_offsets
from ..triton_runtime import (
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except (ImportError, OSError) as e:
    from .. import triton_missing as dummy

    logger.info("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()
    libdevice = dummy.DummyImport()

__all__ = ["single_step_stbif", "multi_step_stbif"]

_STBIF_STATIC_RANGE_MAX_T = 16


@triton.jit
def _single_step_stbif_kernel(
    x_ptr,
    q_init_ptr,
    acc_q_init_ptr,
    q_threshold_ptr,
    pos_max_ptr,
    neg_min_ptr,
    out_ptr,
    q_final_ptr,
    acc_q_final_ptr,
    cur_output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 0)
    q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 1)
    acc_q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 2)
    out_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 3)
    q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 4)
    acc_q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 5)
    cur_output_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 6)
    x = tl.load(x_ptr + x_ptr_offsets, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_init_ptr + q_init_ptr_offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + acc_q_init_ptr_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    q_threshold = tl.load(q_threshold_ptr).to(tl.float32)
    pos_max = tl.load(pos_max_ptr).to(tl.float32)
    neg_min = tl.load(neg_min_ptr).to(tl.float32)

    normalized = x / q_threshold
    q = q + normalized
    acc_q = libdevice.rint(acc_q)
    pos = (q >= 1.0) & (acc_q < pos_max)
    neg = (q < 0.0) & (acc_q > neg_min)
    cur = pos.to(tl.float32) - neg.to(tl.float32)
    acc_q = acc_q + cur
    q = q - pos.to(tl.float32) + neg.to(tl.float32)
    out = cur * q_threshold

    tl.store(out_ptr + out_ptr_offsets, out.to(dtype), mask=mask)
    tl.store(q_final_ptr + q_final_ptr_offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + acc_q_final_ptr_offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + cur_output_ptr_offsets, cur, mask=mask)


@triton.autotune(
    do_bench=triton.testing.do_bench,
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["T", "N", "dtype"],
)
@triton.jit
def _multi_step_stbif_kernel_static(
    x_seq_ptr,
    q_init_ptr,
    acc_q_init_ptr,
    out_seq_ptr,
    q_final_ptr,
    acc_q_final_ptr,
    cur_output_ptr,
    q_threshold_ptr,
    pos_max_ptr,
    neg_min_ptr,
    T: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 0)
    q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 1)
    acc_q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 2)
    out_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 3)
    q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 4)
    acc_q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 5)
    cur_output_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 6)
    q = tl.load(q_init_ptr + q_init_ptr_offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + acc_q_init_ptr_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    q_threshold = tl.load(q_threshold_ptr).to(tl.float32)
    pos_max = tl.load(pos_max_ptr).to(tl.float32)
    neg_min = tl.load(neg_min_ptr).to(tl.float32)

    for t in tl.static_range(0, T, 1):
        x = tl.load(
            x_seq_ptr
            + x_seq_ptr_offsets
            + t * tl.full((), STRIDES.value[0][0], tl.int64),
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = libdevice.rint(acc_q)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(
            out_seq_ptr
            + out_seq_ptr_offsets
            + t * tl.full((), STRIDES.value[3][0], tl.int64),
            (cur * q_threshold).to(dtype),
            mask=mask,
        )

    tl.store(q_final_ptr + q_final_ptr_offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + acc_q_final_ptr_offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + cur_output_ptr_offsets, cur, mask=mask)


@triton.autotune(
    do_bench=triton.testing.do_bench,
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["N", "dtype"],
)
@triton.jit
def _multi_step_stbif_kernel_dynamic(
    x_seq_ptr,
    q_init_ptr,
    acc_q_init_ptr,
    out_seq_ptr,
    q_final_ptr,
    acc_q_final_ptr,
    cur_output_ptr,
    q_threshold_ptr,
    pos_max_ptr,
    neg_min_ptr,
    T,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 0)
    q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 1)
    acc_q_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 2)
    out_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 3)
    q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 4)
    acc_q_final_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 5)
    cur_output_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 6)
    q = tl.load(q_init_ptr + q_init_ptr_offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + acc_q_init_ptr_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    q_threshold = tl.load(q_threshold_ptr).to(tl.float32)
    pos_max = tl.load(pos_max_ptr).to(tl.float32)
    neg_min = tl.load(neg_min_ptr).to(tl.float32)

    for t in tl.range(0, T, 1):
        x = tl.load(
            x_seq_ptr
            + x_seq_ptr_offsets
            + t * tl.full((), STRIDES.value[0][0], tl.int64),
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = libdevice.rint(acc_q)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(
            out_seq_ptr
            + out_seq_ptr_offsets
            + t * tl.full((), STRIDES.value[3][0], tl.int64),
            (cur * q_threshold).to(dtype),
            mask=mask,
        )

    tl.store(q_final_ptr + q_final_ptr_offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + acc_q_final_ptr_offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + cur_output_ptr_offsets, cur, mask=mask)


def _select_stbif_kernel(T: int):
    if T <= _STBIF_STATIC_RANGE_MAX_T and use_static_range_for_triton_neuron_kernel(T):
        return _multi_step_stbif_kernel_static
    return _multi_step_stbif_kernel_dynamic


@register_op("sj_stbif::single_step_stbif")
def single_step_stbif(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <single_step_stbif-cn>` | :ref:`English <single_step_stbif-en>`

    ----

    .. _single_step_stbif-cn:

    * **中文**

    使用专用 Triton kernel 执行单步 STBIF 状态转移。``x``、``q`` 和
    ``acc_q`` 必须是 shape、dtype 和 device 相同的 CUDA FP32、FP16 或
    BF16 张量。本函数仅提供离散推理状态转移，不支持 autograd。

    :param x: 单步 CUDA 输入
    :type x: torch.Tensor
    :param q: 与 ``x`` 同形状的量化残差状态
    :type q: torch.Tensor
    :param acc_q: 与 ``x`` 同形状的累计释放量状态
    :type acc_q: torch.Tensor
    :param q_threshold: 单元素量化尺度张量
    :type q_threshold: torch.Tensor
    :param pos_max: 单元素正向累计上界张量
    :type pos_max: torch.Tensor
    :param neg_min: 单元素负向累计下界张量
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: 当 ``x``、``q`` 和 ``acc_q`` 的 shape、dtype 或 device
        不一致，或任一标量参数不是单元素张量时
    :raises NotImplementedError: 当 dtype 不受 Triton 后端支持时
    :raises ImportError: 当未安装 Triton 时

    ----

    .. _single_step_stbif-en:

    * **English**

    Run one STBIF state transition with the dedicated Triton kernel. ``x``,
    ``q``, and ``acc_q`` must be CUDA FP32, FP16, or BF16 tensors with matching
    shape, dtype, and device. This function implements a discrete inference
    transition and does not support autograd.

    :param x: Single-step CUDA input
    :type x: torch.Tensor
    :param q: Quantized-residual state with the same shape as ``x``
    :type q: torch.Tensor
    :param acc_q: Accumulated released-quantity state with the same shape as ``x``
    :type acc_q: torch.Tensor
    :param q_threshold: Scalar quantization-scale tensor
    :type q_threshold: torch.Tensor
    :param pos_max: Scalar positive accumulated bound
    :type pos_max: torch.Tensor
    :param neg_min: Scalar negative accumulated bound
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: If ``x``, ``q``, and ``acc_q`` differ in shape, dtype,
        or device, or if a scalar parameter does not contain exactly one element
    :raises NotImplementedError: If the dtype is not supported by the Triton
        backend
    :raises ImportError: If Triton is not installed
    """
    if (
        q.shape != x.shape
        or acc_q.shape != x.shape
        or q.dtype != x.dtype
        or acc_q.dtype != x.dtype
        or q.device != x.device
        or acc_q.device != x.device
    ):
        raise ValueError("x, q, and acc_q must have the same shape, dtype, and device.")
    scalar_inputs = (q_threshold, pos_max, neg_min)
    if any(value.numel() != 1 for value in scalar_inputs):
        raise ValueError("q_threshold, pos_max, and neg_min must be scalar tensors.")
    q_threshold = q_threshold.to(device=x.device, dtype=x.dtype)
    pos_max = pos_max.to(device=x.device, dtype=x.dtype)
    neg_min = neg_min.to(device=x.device, dtype=x.dtype)
    N = x.numel()
    dtype = x.dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(dtype)

    out = _empty_like(x, sequence=False)
    q_final = _empty_like(q, sequence=False)
    acc_q_final = _empty_like(acc_q, sequence=False)
    cur_output = _empty_like(q, sequence=False)
    block_n = 256
    grid = (triton.cdiv(N, block_n),)
    sizes, strides = _layout_args(
        x, x, q, acc_q, out, q_final, acc_q_final, cur_output, sequence=False
    )
    with torch.cuda.device(x.device):
        wrap_triton(_single_step_stbif_kernel)[grid](
            x,
            q,
            acc_q,
            q_threshold,
            pos_max,
            neg_min,
            out,
            q_final,
            acc_q_final,
            cur_output,
            N=N,
            SIZES=sizes,
            STRIDES=strides,
            BLOCK_N=block_n,
            dtype=type_dict[dtype],
        )
    return out, q_final, acc_q_final, cur_output


@torch.library.register_fake("sj_stbif::single_step_stbif")
def _single_step_stbif_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
    del q_threshold, pos_max, neg_min
    return (
        _empty_like(x, sequence=False),
        _empty_like(q, sequence=False),
        _empty_like(acc_q, sequence=False),
        _empty_like(q, sequence=False),
    )


@register_op("sj_stbif::multi_step_stbif")
def multi_step_stbif(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <multi_step_stbif-cn>` | :ref:`English <multi_step_stbif-en>`

    ----

    .. _multi_step_stbif-cn:

    * **中文**

    使用专用 Triton kernel 执行多步 STBIF 状态转移。``x_seq`` 的第 0 维
    是时间维；``q`` 和 ``acc_q`` 必须与单个时间步的输入具有相同的
    shape、dtype 和 device。张量必须位于 CUDA，dtype 为 FP32、FP16 或
    BF16。本函数仅提供离散推理状态转移，不支持 autograd。

    :param x_seq: 形状为 ``[T, *]`` 的多步 CUDA 输入，且 ``T > 0``
    :type x_seq: torch.Tensor
    :param q: 形状为 ``x_seq.shape[1:]`` 的量化残差初始状态
    :type q: torch.Tensor
    :param acc_q: 形状为 ``x_seq.shape[1:]`` 的累计释放量初始状态
    :type acc_q: torch.Tensor
    :param q_threshold: 单元素量化尺度张量
    :type q_threshold: torch.Tensor
    :param pos_max: 单元素正向累计上界张量
    :type pos_max: torch.Tensor
    :param neg_min: 单元素负向累计下界张量
    :type neg_min: torch.Tensor
    :return: ``(out_seq, q_final, acc_q_final, cur_output)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: 任一标量参数不是单元素张量
    :raises NotImplementedError: dtype 不受 Triton 后端支持
    :raises ImportError: 未安装 Triton

    ----

    .. _multi_step_stbif-en:

    * **English**

    Run a multi-step STBIF state transition with the dedicated Triton kernel.
    Dimension 0 of ``x_seq`` is time; ``q`` and ``acc_q`` must match one input
    step in shape, dtype, and device. Tensors must be CUDA FP32, FP16, or BF16.
    This function implements a discrete inference transition and does not
    support autograd.

    :param x_seq: Multi-step CUDA input shaped ``[T, *]`` with ``T > 0``
    :type x_seq: torch.Tensor
    :param q: Initial quantized-residual state shaped ``x_seq.shape[1:]``
    :type q: torch.Tensor
    :param acc_q: Initial accumulated released-quantity state shaped
        ``x_seq.shape[1:]``
    :type acc_q: torch.Tensor
    :param q_threshold: Scalar quantization-scale tensor
    :type q_threshold: torch.Tensor
    :param pos_max: Scalar positive accumulated bound
    :type pos_max: torch.Tensor
    :param neg_min: Scalar negative accumulated bound
    :type neg_min: torch.Tensor
    :return: ``(out_seq, q_final, acc_q_final, cur_output)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: If a scalar parameter does not contain exactly one element
    :raises NotImplementedError: If the dtype is not supported by the Triton
        backend
    :raises ImportError: If Triton is not installed
    """
    state_shape = x_seq.shape[1:]
    if any(
        state.shape != state_shape
        or state.dtype != x_seq.dtype
        or state.device != x_seq.device
        for state in (q, acc_q)
    ):
        raise ValueError(
            "q and acc_q must match one x_seq step in shape, dtype, and device."
        )
    if any(value.numel() != 1 for value in (q_threshold, pos_max, neg_min)):
        raise ValueError("q_threshold, pos_max, and neg_min must be scalar tensors.")
    T = x_seq.shape[0]
    N = x_seq[0].numel()
    dtype = x_seq.dtype
    if dtype not in type_dict:
        raise NotImplementedError(dtype)
    out_seq = _empty_like(x_seq)
    q_final = _empty_like(q, sequence=False)
    acc_q_final = _empty_like(acc_q, sequence=False)
    cur_output = _empty_like(q, sequence=False)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    q_threshold = q_threshold.to(device=x_seq.device, dtype=x_seq.dtype)
    pos_max = pos_max.to(device=x_seq.device, dtype=x_seq.dtype)
    neg_min = neg_min.to(device=x_seq.device, dtype=x_seq.dtype)

    sizes, strides = _layout_args(
        x_seq, x_seq, q, acc_q, out_seq, q_final, acc_q_final, cur_output, sequence=True
    )
    with torch.cuda.device(x_seq.device):
        wrap_triton(_select_stbif_kernel(T))[grid](
            x_seq,
            q,
            acc_q,
            out_seq,
            q_final,
            acc_q_final,
            cur_output,
            q_threshold,
            pos_max,
            neg_min,
            T=T,
            N=N,
            SIZES=sizes,
            STRIDES=strides,
            dtype=type_dict[dtype],
        )
    return out_seq, q_final, acc_q_final, cur_output


@torch.library.register_fake("sj_stbif::multi_step_stbif")
def _multi_step_stbif_fake(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
    return (
        _empty_like(x_seq),
        _empty_like(q, sequence=False),
        _empty_like(acc_q, sequence=False),
        _empty_like(q, sequence=False),
    )
