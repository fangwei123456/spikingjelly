from __future__ import annotations

from collections.abc import Callable

import torch


__all__ = [
    "spikezip_release_bias_single_step",
    "spikezip_release_bias_multi_step",
    "spikezip_embedding_single_step",
    "spikezip_embedding_multi_step",
    "spikezip_matmul_delta",
    "spikezip_matmul_sequence_delta",
    "spikezip_vit_single_tokens",
    "spikezip_vit_token_sequence",
    "sta_spike_encoder_single_step",
    "sta_constant_single_step",
    "sta_constant_multi_step",
]


def _temporal_difference(y_cum: torch.Tensor) -> torch.Tensor:
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


def spikezip_release_bias_single_step(
    y: torch.Tensor,
    bias: torch.Tensor | None,
    realize_time: int,
    bias_steps: int,
    bias_view_shape: tuple[int, ...],
) -> tuple[torch.Tensor, int, bool]:
    r"""
    **API Language** - :ref:`中文 <spikezip_release_bias_single_step-cn>` | :ref:`English <spikezip_release_bias_single_step-en>`

    ----

    .. _spikezip_release_bias_single_step-cn:

    * **中文**

    执行 SpikeZIP bias 的单步释放。函数接收当前输出 ``y``、显式剩余释放步数
    ``realize_time``、总释放步数 ``bias_steps`` 和调用方已选定的 bias view shape；
    若仍需释放 bias，则返回 ``y + bias / bias_steps``、递减后的时间和
    ``True``。否则返回原输出、原时间和 ``False``。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode``、
    ``training/eval`` 或子模块状态。

    :param y: 当前输出 tensor
    :type y: torch.Tensor
    :param bias: 可选 bias tensor
    :type bias: torch.Tensor or None
    :param realize_time: 剩余 bias 释放步数
    :type realize_time: int
    :param bias_steps: bias 总释放步数
    :type bias_steps: int
    :param bias_view_shape: bias 广播到 ``y`` 时使用的 view shape
    :type bias_view_shape: Tuple[int, ...]
    :return: ``(y_next, realize_time_next, released)``
    :rtype: Tuple[torch.Tensor, int, bool]

    ----

    .. _spikezip_release_bias_single_step-en:

    * **English**

    Run one SpikeZIP bias-release step. The function receives the current output
    ``y``, explicit remaining release steps ``realize_time``, total
    ``bias_steps``, and the caller-selected bias view shape. If a bias step is
    still pending, it returns ``y + bias / bias_steps``, the decremented time and
    ``True``. Otherwise it returns the original output, original time and
    ``False``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode``, ``training/eval``, or child-module state.

    :param y: Current output tensor
    :type y: torch.Tensor
    :param bias: Optional bias tensor
    :type bias: torch.Tensor or None
    :param realize_time: Remaining bias release steps
    :type realize_time: int
    :param bias_steps: Total bias release steps
    :type bias_steps: int
    :param bias_view_shape: View shape used to broadcast bias to ``y``
    :type bias_view_shape: Tuple[int, ...]
    :return: ``(y_next, realize_time_next, released)``
    :rtype: Tuple[torch.Tensor, int, bool]
    """
    if bias is None or realize_time <= 0:
        return y, realize_time, False
    bias = bias.to(device=y.device, dtype=y.dtype).view(bias_view_shape)
    return y + bias / bias_steps, realize_time - 1, True


def spikezip_release_bias_multi_step(
    y_seq: torch.Tensor,
    bias: torch.Tensor | None,
    realize_time: int,
    bias_steps: int,
    bias_view_shape: tuple[int, ...],
) -> tuple[torch.Tensor, int, int]:
    r"""
    **API Language** - :ref:`中文 <spikezip_release_bias_multi_step-cn>` | :ref:`English <spikezip_release_bias_multi_step-en>`

    ----

    .. _spikezip_release_bias_multi_step-cn:

    * **中文**

    执行 SpikeZIP bias 的多步释放。函数只在序列前 ``min(T, realize_time)`` 步
    加入 ``bias / bias_steps``，并返回更新后的序列、剩余释放时间和本次实际释放
    步数。若没有 bias 或无需释放，则返回原序列且不原地修改输入。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode``、
    ``training/eval`` 或子模块状态。

    :param y_seq: 当前输出序列，第 0 维为时间
    :type y_seq: torch.Tensor
    :param bias: 可选 bias tensor
    :type bias: torch.Tensor or None
    :param realize_time: 剩余 bias 释放步数
    :type realize_time: int
    :param bias_steps: bias 总释放步数
    :type bias_steps: int
    :param bias_view_shape: bias 广播到 ``y_seq`` 切片时使用的 view shape
    :type bias_view_shape: Tuple[int, ...]
    :return: ``(y_seq_next, realize_time_next, released_steps)``
    :rtype: Tuple[torch.Tensor, int, int]

    ----

    .. _spikezip_release_bias_multi_step-en:

    * **English**

    Run multi-step SpikeZIP bias release. The function adds
    ``bias / bias_steps`` only to the first ``min(T, realize_time)`` sequence
    steps, and returns the updated sequence, remaining release time, and the
    number of steps released in this call. If there is no bias or no pending
    release, it returns the original sequence without in-place mutation.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode``, ``training/eval``, or child-module state.

    :param y_seq: Current output sequence with time at dimension 0
    :type y_seq: torch.Tensor
    :param bias: Optional bias tensor
    :type bias: torch.Tensor or None
    :param realize_time: Remaining bias release steps
    :type realize_time: int
    :param bias_steps: Total bias release steps
    :type bias_steps: int
    :param bias_view_shape: View shape used to broadcast bias to ``y_seq`` slices
    :type bias_view_shape: Tuple[int, ...]
    :return: ``(y_seq_next, realize_time_next, released_steps)``
    :rtype: Tuple[torch.Tensor, int, int]
    """
    released_steps = min(y_seq.shape[0], realize_time)
    if bias is None or released_steps <= 0:
        return y_seq, realize_time, 0
    bias = bias.to(device=y_seq.device, dtype=y_seq.dtype).view(bias_view_shape)
    y_next = y_seq.clone()
    y_next[:released_steps] = y_next[:released_steps] + bias / bias_steps
    return y_next, realize_time - released_steps, released_steps


def spikezip_embedding_single_step(
    x: torch.Tensor,
    t: int,
    embedding_forward: Callable[[torch.Tensor], torch.Tensor],
    embedding_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <spikezip_embedding_single_step-cn>` | :ref:`English <spikezip_embedding_single_step-en>`

    ----

    .. _spikezip_embedding_single_step-cn:

    * **中文**

    执行 SpikeZIP embedding 的单步显式时间状态转移。``t == 0`` 时调用已由模块
    选定的 ``embedding_forward(x)``；之后返回与 embedding 输出同形的零张量。
    返回值为 ``(output, t_next)``。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode`` 或
    ``training/eval``。

    :param x: token index tensor
    :type x: torch.Tensor
    :param t: 当前时间索引
    :type t: int
    :param embedding_forward: 已由调用方选择的 embedding 数值路径
    :type embedding_forward: Callable[[torch.Tensor], torch.Tensor]
    :param embedding_dim: embedding 输出尾维
    :type embedding_dim: int
    :param dtype: 零输出 dtype
    :type dtype: torch.dtype
    :return: ``(output, t_next)``
    :rtype: Tuple[torch.Tensor, int]

    ----

    .. _spikezip_embedding_single_step-en:

    * **English**

    Run one explicit time-state transition for SpikeZIP embedding. When
    ``t == 0`` it calls the caller-selected ``embedding_forward(x)``; later
    steps return zeros with the embedding output shape. The return value is
    ``(output, t_next)``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode`` or ``training/eval``.

    :param x: Token index tensor
    :type x: torch.Tensor
    :param t: Current time index
    :type t: int
    :param embedding_forward: Embedding numeric path selected by the caller
    :type embedding_forward: Callable[[torch.Tensor], torch.Tensor]
    :param embedding_dim: Last dimension of the embedding output
    :type embedding_dim: int
    :param dtype: dtype for zero outputs
    :type dtype: torch.dtype
    :return: ``(output, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    """
    if t == 0:
        return embedding_forward(x), t + 1
    return torch.zeros(*x.shape, embedding_dim, device=x.device, dtype=dtype), t + 1


def spikezip_embedding_multi_step(
    x_seq: torch.Tensor,
    embedding_forward: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <spikezip_embedding_multi_step-cn>` | :ref:`English <spikezip_embedding_multi_step-en>`

    ----

    .. _spikezip_embedding_multi_step-cn:

    * **中文**

    执行 SpikeZIP embedding 的多步显式时间状态转移。输入第 0 维为时间维且必须
    非空。第一步输出 ``embedding_forward(x_seq[0])``，后续为零；返回
    ``(output_seq, t_next)``，其中 ``t_next`` 等于输入序列长度。该语义与
    ``SpikeZIPEmbedding.multi_step_forward`` 一致：每次多步调用都从序列首步发放
    embedding，并覆盖 module 的时间索引。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode`` 或
    ``training/eval``。

    :param x_seq: token index 序列，第 0 维为时间
    :type x_seq: torch.Tensor
    :param embedding_forward: 已由调用方选择的 embedding 数值路径
    :type embedding_forward: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(output_seq, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    :raises ValueError: 若输入时间维为空

    ----

    .. _spikezip_embedding_multi_step-en:

    * **English**

    Run a multi-step explicit time-state transition for SpikeZIP embedding. The
    input uses dimension 0 as time and must be non-empty. The first output step
    is ``embedding_forward(x_seq[0])`` and later steps are zeros. It returns
    ``(output_seq, t_next)``, where ``t_next`` equals the input sequence length.
    This matches ``SpikeZIPEmbedding.multi_step_forward``: every multi-step call
    emits the embedding at its first step and overwrites the module time index.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode`` or ``training/eval``.

    :param x_seq: Token index sequence with time at dimension 0
    :type x_seq: torch.Tensor
    :param embedding_forward: Embedding numeric path selected by the caller
    :type embedding_forward: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(output_seq, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    :raises ValueError: If the input time dimension is empty
    """
    if x_seq.shape[0] == 0:
        raise ValueError("SpikeZIPEmbedding expects a non-empty sequence.")
    y0 = embedding_forward(x_seq[0])
    y_seq = torch.zeros(x_seq.shape[0], *y0.shape, device=y0.device, dtype=y0.dtype)
    y_seq[0] = y0
    return y_seq, x_seq.shape[0]


def spikezip_matmul_delta(
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    a_sum: torch.Tensor,
    b_sum: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <spikezip_matmul_delta-cn>` | :ref:`English <spikezip_matmul_delta-en>`

    ----

    .. _spikezip_matmul_delta-cn:

    * **中文**

    执行 SpikeZIP attention 中 ``A_sum @ B_sum`` 的单步差分展开：
    ``a_sum @ b_t + a_t @ b_sum - a_t @ b_t``。``transpose_b=True`` 时先转置
    ``b_t`` 和 ``b_sum`` 的最后两个维度。函数只描述已选定 matmul 片段，不管理
    ``MemoryModule`` state、``step_mode`` 或 ``training/eval``。

    :param a_t: 当前左输入差分
    :type a_t: torch.Tensor
    :param b_t: 当前右输入差分
    :type b_t: torch.Tensor
    :param a_sum: 当前左输入累计
    :type a_sum: torch.Tensor
    :param b_sum: 当前右输入累计
    :type b_sum: torch.Tensor
    :param transpose_b: 是否转置右输入最后两个维度
    :type transpose_b: bool
    :return: 当前 matmul 差分输出
    :rtype: torch.Tensor

    ----

    .. _spikezip_matmul_delta-en:

    * **English**

    Run the single-step expansion for ``A_sum @ B_sum`` used by SpikeZIP
    attention: ``a_sum @ b_t + a_t @ b_sum - a_t @ b_t``. When
    ``transpose_b=True``, the last two dimensions of ``b_t`` and ``b_sum`` are
    transposed first. The function only describes the selected matmul fragment
    and does not manage ``MemoryModule`` state, ``step_mode``, or
    ``training/eval``.

    :param a_t: Current left-input difference
    :type a_t: torch.Tensor
    :param b_t: Current right-input difference
    :type b_t: torch.Tensor
    :param a_sum: Current accumulated left input
    :type a_sum: torch.Tensor
    :param b_sum: Current accumulated right input
    :type b_sum: torch.Tensor
    :param transpose_b: Whether to transpose the last two dimensions of the right input
    :type transpose_b: bool
    :return: Current matmul difference output
    :rtype: torch.Tensor
    """
    b_t_arg = b_t.transpose(-2, -1) if transpose_b else b_t
    b_sum_arg = b_sum.transpose(-2, -1) if transpose_b else b_sum
    return a_sum @ b_t_arg + a_t @ b_sum_arg - a_t @ b_t_arg


def spikezip_matmul_sequence_delta(
    a_seq: torch.Tensor,
    b_seq: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <spikezip_matmul_sequence_delta-cn>` | :ref:`English <spikezip_matmul_sequence_delta-en>`

    ----

    .. _spikezip_matmul_sequence_delta-cn:

    * **中文**

    执行 SpikeZIP attention 中 ``A.cumsum(0) @ B.cumsum(0)`` 的多步时间差分。
    ``transpose_b=True`` 时先转置累计右输入的最后两个维度。函数不读取或写入
    ``MemoryModule`` memory，不管理 ``step_mode`` 或 ``training/eval``。

    :param a_seq: 左输入序列，第 0 维为时间
    :type a_seq: torch.Tensor
    :param b_seq: 右输入序列，第 0 维为时间
    :type b_seq: torch.Tensor
    :param transpose_b: 是否转置右输入最后两个维度
    :type transpose_b: bool
    :return: matmul 的时间差分输出序列
    :rtype: torch.Tensor

    ----

    .. _spikezip_matmul_sequence_delta-en:

    * **English**

    Run the multi-step temporal difference of
    ``A.cumsum(0) @ B.cumsum(0)`` used by SpikeZIP attention. When
    ``transpose_b=True``, the last two dimensions of the accumulated right input
    are transposed first. The function does not read or write ``MemoryModule``
    memory and does not manage ``step_mode`` or ``training/eval``.

    :param a_seq: Left input sequence with time at dimension 0
    :type a_seq: torch.Tensor
    :param b_seq: Right input sequence with time at dimension 0
    :type b_seq: torch.Tensor
    :param transpose_b: Whether to transpose the last two dimensions of the right input
    :type transpose_b: bool
    :return: Temporal-difference matmul output sequence
    :rtype: torch.Tensor
    """
    a_cum = a_seq.cumsum(dim=0)
    b_cum = b_seq.cumsum(dim=0)
    if transpose_b:
        b_cum = b_cum.transpose(-2, -1)
    return _temporal_difference(a_cum @ b_cum)


def spikezip_vit_single_tokens(
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    batch_size: int,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <spikezip_vit_single_tokens-cn>` | :ref:`English <spikezip_vit_single_tokens-en>`

    ----

    .. _spikezip_vit_single_tokens-cn:

    * **中文**

    生成 SpikeZIP ViT 单步 cls token 与 position embedding。``t == 0`` 时返回
    batch-expanded ``cls_token`` 和 ``pos_embed``；之后返回同形零张量。返回值包含
    递增后的时间 ``t_next``。

    :param cls_token: ViT cls token 参数
    :type cls_token: torch.Tensor
    :param pos_embed: ViT position embedding 参数
    :type pos_embed: torch.Tensor
    :param batch_size: batch 大小
    :type batch_size: int
    :param t: 当前时间索引
    :type t: int
    :return: ``(cls, pos, t_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, int]

    ----

    .. _spikezip_vit_single_tokens-en:

    * **English**

    Generate single-step SpikeZIP ViT cls token and position embedding. When
    ``t == 0``, it returns the batch-expanded ``cls_token`` and ``pos_embed``;
    later steps return zeros with the same shapes. The return value includes the
    incremented ``t_next``.

    :param cls_token: ViT cls token parameter
    :type cls_token: torch.Tensor
    :param pos_embed: ViT position embedding parameter
    :type pos_embed: torch.Tensor
    :param batch_size: Batch size
    :type batch_size: int
    :param t: Current time index
    :type t: int
    :return: ``(cls, pos, t_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, int]
    """
    if t == 0:
        cls = cls_token.expand(batch_size, -1, -1)
        pos = pos_embed
    else:
        cls = torch.zeros(
            batch_size,
            *cls_token.shape[1:],
            device=cls_token.device,
            dtype=cls_token.dtype,
        )
        pos = torch.zeros_like(pos_embed)
    return cls, pos, t + 1


def spikezip_vit_token_sequence(
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    time_steps: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <spikezip_vit_token_sequence-cn>` | :ref:`English <spikezip_vit_token_sequence-en>`

    ----

    .. _spikezip_vit_token_sequence-cn:

    * **中文**

    生成 SpikeZIP ViT 多步 cls token 与 position embedding 序列。第一步为
    batch-expanded ``cls_token`` 和 ``pos_embed``，后续时间步为零。

    :param cls_token: ViT cls token 参数
    :type cls_token: torch.Tensor
    :param pos_embed: ViT position embedding 参数
    :type pos_embed: torch.Tensor
    :param time_steps: 输出时间步数
    :type time_steps: int
    :param batch_size: batch 大小
    :type batch_size: int
    :return: ``(cls_seq, pos_seq)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _spikezip_vit_token_sequence-en:

    * **English**

    Generate multi-step SpikeZIP ViT cls token and position-embedding sequences.
    The first step contains the batch-expanded ``cls_token`` and ``pos_embed``;
    later timesteps are zeros.

    :param cls_token: ViT cls token parameter
    :type cls_token: torch.Tensor
    :param pos_embed: ViT position embedding parameter
    :type pos_embed: torch.Tensor
    :param time_steps: Number of output timesteps
    :type time_steps: int
    :param batch_size: Batch size
    :type batch_size: int
    :return: ``(cls_seq, pos_seq)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    cls = cls_token.expand(batch_size, -1, -1)
    cls_seq = torch.zeros(
        time_steps,
        *cls.shape,
        device=cls.device,
        dtype=cls.dtype,
    )
    pos_seq = torch.zeros(
        time_steps,
        *pos_embed.shape,
        device=pos_embed.device,
        dtype=pos_embed.dtype,
    )
    cls_seq[0] = cls
    pos_seq[0] = pos_embed
    return cls_seq, pos_seq


def sta_spike_encoder_single_step(
    x: torch.Tensor,
    mem: torch.Tensor,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <sta_spike_encoder_single_step-cn>` | :ref:`English <sta_spike_encoder_single_step-en>`

    ----

    .. _sta_spike_encoder_single_step-cn:

    * **中文**

    执行 STA spike encoder 的单步显式 ``mem`` 状态转移。函数接收当前输入 ``x``、
    已物化的残差 ``mem`` 和已广播/裁剪到当前输入 dtype/device 的正阈值
    ``threshold``，返回 ``(spike, mem_next)``。

    函数不读取或写入 ``MemoryModule`` memory，不负责 state 物化、shape/device/dtype
    不匹配时的重建、channel-wise 阈值广播、``step_mode`` 或 ``training/eval``。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param mem: 已物化的残差 state，shape 与 ``x`` 相同
    :type mem: torch.Tensor
    :param threshold: 已广播并 clamp 到正值的阈值 tensor，可与 ``x`` 广播
    :type threshold: torch.Tensor
    :return: ``(spike, mem_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _sta_spike_encoder_single_step-en:

    * **English**

    Run one explicit ``mem`` state transition for the STA spike encoder. The
    function receives current input ``x``, materialized residual ``mem``, and a
    positive ``threshold`` already broadcast/clamped on the current dtype/device,
    and returns ``(spike, mem_next)``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage state materialization, shape/device/dtype mismatch rebuilding,
    channel-wise threshold broadcasting, ``step_mode``, or ``training/eval``.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param mem: Materialized residual state with the same shape as ``x``
    :type mem: torch.Tensor
    :param threshold: Positive threshold tensor already broadcast and clamped;
        it must be broadcastable to ``x``
    :type threshold: torch.Tensor
    :return: ``(spike, mem_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    mem_next = mem + x
    spike_count = torch.trunc(mem_next / threshold)
    spike = spike_count * threshold
    mem_next = mem_next - spike
    return spike, mem_next


def sta_constant_single_step(
    value: torch.Tensor,
    t: int,
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <sta_constant_single_step-cn>` | :ref:`English <sta_constant_single_step-en>`

    ----

    .. _sta_constant_single_step-cn:

    * **中文**

    执行 STA constant 的单步显式时间状态转移。``t == 0`` 时输出 ``value``，
    否则输出 ``torch.zeros_like(value)``；返回 ``(output, t_next)``。

    :param value: 常量 tensor
    :type value: torch.Tensor
    :param t: 当前时间索引
    :type t: int
    :return: ``(output, t_next)``
    :rtype: Tuple[torch.Tensor, int]

    ----

    .. _sta_constant_single_step-en:

    * **English**

    Run one explicit time-state transition for STA constant. When ``t == 0`` the
    function outputs ``value``; otherwise it outputs ``torch.zeros_like(value)``.
    It returns ``(output, t_next)``.

    :param value: Constant tensor
    :type value: torch.Tensor
    :param t: Current time index
    :type t: int
    :return: ``(output, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    """
    output = value if t == 0 else torch.zeros_like(value)
    return output, t + 1


def sta_constant_multi_step(
    value: torch.Tensor,
    t: int,
    time_steps: int,
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <sta_constant_multi_step-cn>` | :ref:`English <sta_constant_multi_step-en>`

    ----

    .. _sta_constant_multi_step-cn:

    * **中文**

    执行 STA constant 的多步显式时间状态转移。若 ``t == 0``，输出第一步为
    ``value``、后续为零的序列；否则输出全零序列。返回 ``(output_seq, t_next)``。

    :param value: 常量 tensor
    :type value: torch.Tensor
    :param t: 当前时间索引
    :type t: int
    :param time_steps: 输出时间步数
    :type time_steps: int
    :return: ``(output_seq, t_next)``
    :rtype: Tuple[torch.Tensor, int]

    ----

    .. _sta_constant_multi_step-en:

    * **English**

    Run the explicit multi-step time-state transition for STA constant. If
    ``t == 0``, the first output step is ``value`` and the rest are zeros;
    otherwise the whole output sequence is zeros. It returns
    ``(output_seq, t_next)``.

    :param value: Constant tensor
    :type value: torch.Tensor
    :param t: Current time index
    :type t: int
    :param time_steps: Number of output time steps
    :type time_steps: int
    :return: ``(output_seq, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    """
    if t == 0:
        zeros = torch.zeros_like(value).expand(time_steps - 1, *value.shape)
        output = torch.cat((value.unsqueeze(0), zeros), dim=0)
    else:
        output = torch.zeros_like(value).expand(time_steps, *value.shape)
    return output, t + time_steps
