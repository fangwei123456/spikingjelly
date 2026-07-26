from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch


__all__ = [
    "delay_single_step",
    "element_wise_recurrent_single_step",
    "linear_recurrent_single_step",
    "batch_norm_through_time_single_step",
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


def element_wise_recurrent_single_step(
    x: torch.Tensor,
    y: torch.Tensor,
    element_wise_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sub_module: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <element_wise_recurrent_single_step-cn>` | :ref:`English <element_wise_recurrent_single_step-en>`

    ----

    .. _element_wise_recurrent_single_step-cn:

    * **中文**

    执行 ``ElementWiseRecurrentContainer`` 的单步局部状态转移。函数显式接收当前
    输出状态 ``y``，按现有 module 语义调用 ``element_wise_function(y, x)``，再将
    结果传给 ``sub_module``，返回下一输出状态。该函数只管理当前容器自己的 ``y``，
    不递归展开 ``sub_module`` 的参数、buffer 或 hidden state。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param y: 已物化的当前输出 tensor state
    :type y: torch.Tensor
    :param element_wise_function: 逐元素合成函数，按 ``element_wise_function(y, x)``
        调用
    :type element_wise_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    :param sub_module: 接收合成结果并返回下一 ``y`` 的 callable
    :type sub_module: Callable[[torch.Tensor], torch.Tensor]
    :return: 下一输出状态
    :rtype: torch.Tensor

    ----

    .. _element_wise_recurrent_single_step-en:

    * **English**

    Run one local state transition for ``ElementWiseRecurrentContainer``. The
    function receives the current output state ``y`` explicitly, calls
    ``element_wise_function(y, x)`` to match the existing module semantics, feeds
    the result to ``sub_module``, and returns the next output state. It only
    manages this container's own ``y`` and does not recursively expand
    ``sub_module`` parameters, buffers, or hidden state.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param y: Materialized current output tensor state
    :type y: torch.Tensor
    :param element_wise_function: Element-wise merge function called as
        ``element_wise_function(y, x)``
    :type element_wise_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    :param sub_module: Callable that receives the merged tensor and returns the
        next ``y``
    :type sub_module: Callable[[torch.Tensor], torch.Tensor]
    :return: Next output state
    :rtype: torch.Tensor
    """
    return sub_module(element_wise_function(y, x))


def linear_recurrent_single_step(
    x: torch.Tensor,
    y: torch.Tensor,
    recurrent_module: Callable[[torch.Tensor], torch.Tensor],
    sub_module: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <linear_recurrent_single_step-cn>` | :ref:`English <linear_recurrent_single_step-en>`

    ----

    .. _linear_recurrent_single_step-cn:

    * **中文**

    执行 ``LinearRecurrentContainer`` 的单步局部状态转移。函数显式接收当前输出状态
    ``y`` 和线性自连接模块，依次调用
    ``recurrent_module(torch.cat((x, y), dim=-1))`` 与 ``sub_module``，返回下一
    ``y``。该函数不物化 ``None`` state，也不递归展开两个 module 的内部状态。

    :param x: 当前输入张量，最后一维为输入特征
    :type x: torch.Tensor
    :param y: 已物化的当前输出 tensor state，最后一维为输出特征
    :type y: torch.Tensor
    :param recurrent_module: 接收拼接后的 ``(x, y)`` 并产生自连接输入的 callable
    :type recurrent_module: Callable[[torch.Tensor], torch.Tensor]
    :param sub_module: 接收线性自连接结果并返回下一 ``y`` 的 callable
    :type sub_module: Callable[[torch.Tensor], torch.Tensor]
    :return: 下一输出状态
    :rtype: torch.Tensor

    ----

    .. _linear_recurrent_single_step-en:

    * **English**

    Run one local state transition for ``LinearRecurrentContainer``. The function
    receives the current output state ``y`` and the recurrent module, calls
    ``recurrent_module(torch.cat((x, y), dim=-1))`` followed by ``sub_module``,
    and returns the next ``y``. It does not materialize ``None`` state and does
    not recursively expand either module's internal state.

    :param x: Current input tensor with input features on the last dimension
    :type x: torch.Tensor
    :param y: Materialized current output tensor state with output features on
        the last dimension
    :type y: torch.Tensor
    :param recurrent_module: Callable that receives concatenated ``(x, y)`` and
        produces the recurrent input
    :type recurrent_module: Callable[[torch.Tensor], torch.Tensor]
    :param sub_module: Callable that receives the linear recurrent output and
        returns the next ``y``
    :type sub_module: Callable[[torch.Tensor], torch.Tensor]
    :return: Next output state
    :rtype: torch.Tensor
    """
    return sub_module(recurrent_module(torch.cat((x, y), dim=-1)))


def batch_norm_through_time_single_step(
    x: torch.Tensor,
    t: int,
    bn_modules: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    disable_running_stats: bool = False,
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <batch_norm_through_time_single_step-cn>` | :ref:`English <batch_norm_through_time_single_step-en>`

    ----

    .. _batch_norm_through_time_single_step-cn:

    * **中文**

    执行 Batch Normalization Through Time (BNTT) 的单步时间状态转移。函数接收当前
    Python 时间索引 ``t``，选择 ``bn_modules[t + 1]``，返回 ``(out, t_next)``。
    BatchNorm 的参数、buffer、``training/eval`` 和 running stats 仍由被调用的 BN
    module 管理；本函数不把这些状态展开为 functional state。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param t: 当前时间索引；初始值通常为 ``-1``
    :type t: int
    :param bn_modules: 每个时间步对应的 BatchNorm callable
    :type bn_modules: Sequence[Callable[[torch.Tensor], torch.Tensor]]
    :param disable_running_stats: 是否在本次调用期间临时关闭所选 BN 的
        ``track_running_stats``
    :type disable_running_stats: bool
    :return: ``(out, t_next)``
    :rtype: Tuple[torch.Tensor, int]

    ----

    .. _batch_norm_through_time_single_step-en:

    * **English**

    Run one Batch Normalization Through Time (BNTT) time-state transition. The
    function receives the current Python time index ``t``, selects
    ``bn_modules[t + 1]``, and returns ``(out, t_next)``. BatchNorm parameters,
    buffers, ``training/eval``, and running stats remain owned by the selected BN
    module; this function does not expand them as functional state.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param t: Current time index, usually initialized as ``-1``
    :type t: int
    :param bn_modules: BatchNorm callables for each time step
    :type bn_modules: Sequence[Callable[[torch.Tensor], torch.Tensor]]
    :param disable_running_stats: Whether to temporarily disable
        ``track_running_stats`` on the selected BN for this call
    :type disable_running_stats: bool
    :return: ``(out, t_next)``
    :rtype: Tuple[torch.Tensor, int]
    """
    t_next = t + 1
    bn = bn_modules[t_next]
    original_track_running_stats = getattr(bn, "track_running_stats", None)
    if disable_running_stats and original_track_running_stats is not None:
        bn.track_running_stats = False
    try:
        return bn(x), t_next
    finally:
        if disable_running_stats and original_track_running_stats is not None:
            bn.track_running_stats = original_track_running_stats


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
