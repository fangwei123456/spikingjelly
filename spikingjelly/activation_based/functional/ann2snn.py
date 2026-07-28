from __future__ import annotations

import torch


__all__ = [
    "spikezip_release_bias_single_step",
    "spikezip_release_bias_multi_step",
    "spikezip_matmul_delta",
    "spikezip_matmul_sequence_delta",
    "sta_spike_encoder_single_step",
]


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
    y_cum = a_cum @ b_cum
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


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
