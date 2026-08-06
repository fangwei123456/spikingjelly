from __future__ import annotations

import torch


__all__ = [
    "delay_step",
    "neunorm_step",
    "synapse_filter_step",
]


def neunorm_step(
    in_spikes: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    momentum: float,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <neunorm_step-cn>` | :ref:`English <neunorm_step-en>`

    ----

    .. _neunorm_step-cn:

    * **中文**

    执行一次 NeuNorm 状态转移，返回归一化输出和下一状态。函数不读取或修改
    module memory。

    :param in_spikes: 当前输入脉冲，shape 为 ``[N, C, H, W]``
    :type in_spikes: torch.Tensor
    :param state: 已物化的 NeuNorm 状态，shape 为 ``[N, 1, H, W]``
    :type state: torch.Tensor
    :param weight: 可广播到 ``in_spikes`` 的 NeuNorm 权重
    :type weight: torch.Tensor
    :param momentum: 旧状态的系数
    :type momentum: float
    :param input_scale: 通道求和结果的系数
    :type input_scale: float
    :return: ``(output, state_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _neunorm_step-en:

    * **English**

    Run one NeuNorm state transition and return its normalized output and next
    state. The function does not read or mutate module memory.

    :param in_spikes: Current input spikes shaped ``[N, C, H, W]``
    :type in_spikes: torch.Tensor
    :param state: Materialized NeuNorm state shaped ``[N, 1, H, W]``
    :type state: torch.Tensor
    :param weight: NeuNorm weight broadcastable to ``in_spikes``
    :type weight: torch.Tensor
    :param momentum: Coefficient applied to the previous state
    :type momentum: float
    :param input_scale: Coefficient applied to the channel sum
    :type input_scale: float
    :return: ``(output, state_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    state_next = momentum * state + input_scale * in_spikes.sum(dim=1, keepdim=True)
    return in_spikes - weight * state_next, state_next


def delay_step(
    x: torch.Tensor,
    queue: tuple[torch.Tensor, ...],
    delay_steps: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    r"""
    **API Language** - :ref:`中文 <delay_step-cn>` | :ref:`English <delay_step-en>`

    ----

    .. _delay_step-cn:

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

    .. _delay_step-en:

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    if not isinstance(delay_steps, int) or delay_steps < 0:
        raise ValueError("delay_steps must be a non-negative integer")

    queue_with_x = (*queue, x)
    if len(queue_with_x) > delay_steps:
        return queue_with_x[0], queue_with_x[1:]
    return torch.zeros_like(x), queue_with_x


def synapse_filter_step(
    x: torch.Tensor,
    out_i: torch.Tensor,
    reciprocal_tau: float | torch.Tensor,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <synapse_filter_step-cn>` | :ref:`English <synapse_filter_step-en>`

    ----

    .. _synapse_filter_step-cn:

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

    .. _synapse_filter_step-en:

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    return out_i - (1.0 - x) * out_i * reciprocal_tau + x
