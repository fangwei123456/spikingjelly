from __future__ import annotations

import torch


__all__ = [
    "delay_single_step",
    "neunorm_single_step",
    "synapse_filter_single_step",
]


def delay_single_step(
    x: torch.Tensor,
    queue: tuple[torch.Tensor, ...],
    delay_steps: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    r"""
    **API Language** - :ref:`中文 <delay_single_step-cn>` | :ref:`English <delay_single_step-en>`

    ----

    .. _delay_single_step-cn:

    * **中文**

    执行 Delay 的单步显式状态转移。``queue`` 是按时间从旧到新排列的 tensor tuple。
    函数返回 ``(y, queue_next)``，不原地修改输入 ``queue``。当 ``delay_steps=0``
    且 ``queue`` 为空时，输出 ``y`` 与输入 ``x`` alias；当已有 queue 被消费时，输出
    ``y`` 与被弹出的 queue 元素 alias。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param queue: 当前延迟队列状态，元素为 tensor，按旧到新排列
    :type queue: Tuple[torch.Tensor, ...]
    :param delay_steps: 延迟时间步数，必须是非负整数
    :type delay_steps: int
    :return: ``(y, queue_next)``，其中 ``y`` 是当前输出，``queue_next`` 是下一状态
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
    :raises ValueError: ``delay_steps`` 不是非负整数时抛出

    ----

    .. _delay_single_step-en:

    * **English**

    Run one explicit Delay state transition. ``queue`` is a tuple of tensors
    ordered from oldest to newest. The function returns ``(y, queue_next)`` and
    does not mutate the input ``queue`` in place. When ``delay_steps=0`` and
    ``queue`` is empty, output ``y`` aliases ``x``; when an existing queue item is
    consumed, ``y`` aliases the popped queue item.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param queue: Current delay-queue state with tensor elements ordered from
        oldest to newest
    :type queue: Tuple[torch.Tensor, ...]
    :param delay_steps: Number of delayed time steps; must be a non-negative
        integer
    :type delay_steps: int
    :return: ``(y, queue_next)``, where ``y`` is the current output and
        ``queue_next`` is the next state
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
    :raises ValueError: If ``delay_steps`` is not a non-negative integer
    """
    if not isinstance(delay_steps, int) or delay_steps < 0:
        raise ValueError("delay_steps must be a non-negative integer")

    queue_with_x = (*queue, x)
    if len(queue_with_x) > delay_steps:
        return queue_with_x[0], queue_with_x[1:]
    return torch.zeros_like(x), queue_with_x


def neunorm_single_step(
    in_spikes: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    k0: float,
    k1: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <neunorm_single_step-cn>` | :ref:`English <neunorm_single_step-en>`

    ----

    .. _neunorm_single_step-cn:

    * **中文**

    执行 ``NeuNorm`` 的单步显式状态转移，返回 ``(out, x_next)``。函数接收已物化的
    归一化状态 ``x``、权重 ``w`` 和系数 ``k0``/``k1``，不读取或修改
    ``MemoryModule`` memory。

    :param in_spikes: 当前输入脉冲张量，shape 为 ``[N, C, H, W]``
    :type in_spikes: torch.Tensor
    :param x: 已物化的当前归一化 tensor state，shape 为 ``[N, 1, H, W]``
    :type x: torch.Tensor
    :param w: NeuNorm 权重，shape 为 ``[C, H, W]`` 或 ``[1, H, W]``
    :type w: torch.Tensor
    :param k0: state 动量系数
    :type k0: float
    :param k1: 当前输入累加项系数
    :type k1: float
    :return: ``(out, x_next)``，其中 ``out`` 是归一化后的输出，``x_next`` 是下一状态
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _neunorm_single_step-en:

    * **English**

    Run one explicit ``NeuNorm`` state transition and return ``(out, x_next)``.
    The function receives the materialized normalization state ``x``, weight
    ``w``, and coefficients ``k0``/``k1`` explicitly, and does not read or mutate
    ``MemoryModule`` memory.

    :param in_spikes: Current input spike tensor shaped ``[N, C, H, W]``
    :type in_spikes: torch.Tensor
    :param x: Materialized current normalization tensor state shaped
        ``[N, 1, H, W]``
    :type x: torch.Tensor
    :param w: NeuNorm weight shaped ``[C, H, W]`` or ``[1, H, W]``
    :type w: torch.Tensor
    :param k0: State momentum coefficient
    :type k0: float
    :param k1: Current-input accumulation coefficient
    :type k1: float
    :return: ``(out, x_next)``, where ``out`` is the normalized output and
        ``x_next`` is the next state
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    x_next = k0 * x + k1 * in_spikes.sum(dim=1, keepdim=True)
    return in_spikes - w * x_next, x_next


def synapse_filter_single_step(
    x: torch.Tensor,
    out_i: torch.Tensor,
    reciprocal_tau: float | torch.Tensor,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <synapse_filter_single_step-cn>` | :ref:`English <synapse_filter_single_step-en>`

    ----

    .. _synapse_filter_single_step-cn:

    * **中文**

    执行 SynapseFilter 的单步显式状态转移。函数接收已物化的输出电流状态
    ``out_i`` 和确定的 ``reciprocal_tau = 1 / tau``，返回下一时刻输出电流。
    该函数不读取或修改 ``MemoryModule`` memory，也不原地修改 ``x`` 或 ``out_i``。

    :param x: 当前输入脉冲或输入电流张量
    :type x: torch.Tensor
    :param out_i: 已物化的当前输出电流 tensor state，shape/dtype/device 与 ``x`` 兼容
    :type out_i: torch.Tensor
    :param reciprocal_tau: 时间常数倒数；非 learnable module 传入 ``1 / tau``，
        learnable module 传入 ``w.sigmoid()``
    :type reciprocal_tau: float | torch.Tensor
    :return: 下一时刻输出电流
    :rtype: torch.Tensor

    ----

    .. _synapse_filter_single_step-en:

    * **English**

    Run one explicit SynapseFilter state transition. The function receives a
    materialized output-current state ``out_i`` and the selected
    ``reciprocal_tau = 1 / tau``, then returns the next output current. It does
    not read or mutate ``MemoryModule`` memory and does not mutate ``x`` or
    ``out_i`` in place.

    :param x: Current input spike or input-current tensor
    :type x: torch.Tensor
    :param out_i: Materialized current output-current tensor state compatible
        with ``x`` in shape, dtype, and device
    :type out_i: torch.Tensor
    :param reciprocal_tau: Reciprocal time constant; non-learnable modules pass
        ``1 / tau`` and learnable modules pass ``w.sigmoid()``
    :type reciprocal_tau: float | torch.Tensor
    :return: Next output current
    :rtype: torch.Tensor
    """
    return out_i - (1.0 - x) * out_i * reciprocal_tau + x
