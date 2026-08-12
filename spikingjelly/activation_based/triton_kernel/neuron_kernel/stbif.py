from __future__ import annotations

import torch

from spikingjelly.logger import logger

from ..triton_utils import (
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
    from .. import dummy

    logger.debug(
        "spikingjelly.activation_based.triton_kernel.neuron_kernel.stbif: %s", e
    )
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
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
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

    tl.store(out_ptr + offsets, out.to(dtype), mask=mask)
    tl.store(q_final_ptr + offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + offsets, cur, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["T", "N", "dtype"],
    restore_value=["out_seq_ptr", "q_final_ptr", "acc_q_final_ptr", "cur_output_ptr"],
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
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    q = tl.load(q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    q_threshold = tl.load(q_threshold_ptr).to(tl.float32)
    pos_max = tl.load(pos_max_ptr).to(tl.float32)
    neg_min = tl.load(neg_min_ptr).to(tl.float32)

    for t in tl.static_range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = libdevice.rint(acc_q)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(
            out_seq_ptr + t * N + offsets, (cur * q_threshold).to(dtype), mask=mask
        )

    tl.store(q_final_ptr + offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + offsets, cur, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["N", "dtype"],
    restore_value=["out_seq_ptr", "q_final_ptr", "acc_q_final_ptr", "cur_output_ptr"],
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
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    q = tl.load(q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    q_threshold = tl.load(q_threshold_ptr).to(tl.float32)
    pos_max = tl.load(pos_max_ptr).to(tl.float32)
    neg_min = tl.load(neg_min_ptr).to(tl.float32)

    for t in tl.range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = libdevice.rint(acc_q)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(
            out_seq_ptr + t * N + offsets, (cur * q_threshold).to(dtype), mask=mask
        )

    tl.store(q_final_ptr + offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + offsets, cur, mask=mask)


def _select_stbif_kernel(T: int):
    if T <= _STBIF_STATIC_RANGE_MAX_T and use_static_range_for_triton_neuron_kernel(T):
        return _multi_step_stbif_kernel_static
    return _multi_step_stbif_kernel_dynamic


@register_op("sj::single_step_stbif")
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
    x = x.contiguous()
    q = q.contiguous()
    acc_q = acc_q.contiguous()
    q_threshold = q_threshold.to(device=x.device, dtype=x.dtype).contiguous()
    pos_max = pos_max.to(device=x.device, dtype=x.dtype).contiguous()
    neg_min = neg_min.to(device=x.device, dtype=x.dtype).contiguous()
    N = x.numel()
    dtype = x.dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(dtype)

    out = torch.empty_like(x)
    q_final = torch.empty_like(q)
    acc_q_final = torch.empty_like(acc_q)
    cur_output = torch.empty_like(q)
    block_n = 256
    grid = (triton.cdiv(N, block_n),)
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
            BLOCK_N=block_n,
            dtype=type_dict[dtype],
        )
    return out, q_final, acc_q_final, cur_output


@torch.library.register_fake("sj::single_step_stbif")
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
        torch.empty_like(x),
        torch.empty_like(q),
        torch.empty_like(acc_q),
        torch.empty_like(q),
    )


@register_op("sj::multi_step_stbif")
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
    """
    if any(value.numel() != 1 for value in (q_threshold, pos_max, neg_min)):
        raise ValueError("q_threshold, pos_max, and neg_min must be scalar tensors.")
    x_shape = x_seq.shape
    x_seq = x_seq.contiguous()
    q = q.contiguous()
    acc_q = acc_q.contiguous()
    T = x_seq.shape[0]
    N = x_seq[0].numel()
    dtype = x_seq.dtype
    if dtype not in type_dict:
        raise NotImplementedError(dtype)
    out_seq = torch.empty_like(x_seq)
    q_final = torch.empty_like(q)
    acc_q_final = torch.empty_like(acc_q)
    cur_output = torch.empty_like(q)

    x_seq_flat = x_seq.reshape(T, N)
    q_flat = q.reshape(N)
    acc_q_flat = acc_q.reshape(N)
    out_seq_flat = out_seq.reshape(T, N)
    q_final_flat = q_final.reshape(N)
    acc_q_final_flat = acc_q_final.reshape(N)
    cur_output_flat = cur_output.reshape(N)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    q_threshold = q_threshold.to(device=x_seq.device, dtype=x_seq.dtype).contiguous()
    pos_max = pos_max.to(device=x_seq.device, dtype=x_seq.dtype).contiguous()
    neg_min = neg_min.to(device=x_seq.device, dtype=x_seq.dtype).contiguous()

    with torch.cuda.device(x_seq.device):
        wrap_triton(_select_stbif_kernel(T))[grid](
            x_seq_flat,
            q_flat,
            acc_q_flat,
            out_seq_flat,
            q_final_flat,
            acc_q_final_flat,
            cur_output_flat,
            q_threshold,
            pos_max,
            neg_min,
            T=T,
            N=N,
            dtype=type_dict[dtype],
        )
    return out_seq.reshape(x_shape), q_final, acc_q_final, cur_output


@torch.library.register_fake("sj::multi_step_stbif")
def _multi_step_stbif_fake(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
    return (
        x_seq.new_empty(x_seq.shape),
        q.new_empty(q.shape),
        acc_q.new_empty(acc_q.shape),
        q.new_empty(q.shape),
    )
