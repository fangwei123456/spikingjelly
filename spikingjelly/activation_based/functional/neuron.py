"""Explicit neuron updates.

``*_step`` consumes one time step. ``*_multi_step`` consumes a time-major sequence
and is exposed only when the sequence path has its own implementation rather
than being a Python loop over ``*_step``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


__all__ = [
    "lava_cuba_lif_step",
    "if_step",
    "qif_step",
    "eif_step",
    "lif_step",
    "plif_step",
    "izhikevich_step",
    "klif_step",
    "cuba_lif_step",
    "if_multi_step_inductor",
    "lif_multi_step_inductor",
    "plif_multi_step_inductor",
    "if_step_cupy",
    "lif_step_cupy",
    "if_multi_step_cupy",
    "lif_multi_step_cupy",
    "plif_multi_step_cupy",
    "qif_multi_step_cupy",
    "eif_multi_step_cupy",
    "izhikevich_multi_step_cupy",
    "if_multi_step_triton",
    "lif_multi_step_triton",
    "plif_multi_step_triton",
    "sliding_psn_step",
    "gated_lif_multi_step",
    "stbif_step",
    "stbif_multi_step_torch",
    "activation_aware_if_step",
    "activation_aware_if_multi_step_triton",
]


SurrogateFunction = Callable[[torch.Tensor], torch.Tensor]


def lava_cuba_lif_step(
    x: torch.Tensor,
    current_state: torch.Tensor,
    voltage_state: torch.Tensor,
    current_decay: torch.Tensor,
    voltage_decay: torch.Tensor,
    s_scale: float,
    v_threshold: float,
    v_threshold_eps: float,
    v_reset: float,
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lava_cuba_lif_step-cn>` | :ref:`English <lava_cuba_lif_step-en>`

    ----

    .. _lava_cuba_lif_step-cn:

    * **中文**

    执行 ``lava_exchange.CubaLIFNode`` 不含可选 norm 的 Torch 路径的一次状态
    更新，返回脉冲、下一电流状态和 reset 后的下一电压状态。函数不物化 state，
    不读取 module，也不判断 ``training/eval``。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param current_state: 当前电流状态张量
    :type current_state: torch.Tensor
    :param voltage_state: 当前电压状态张量
    :type voltage_state: torch.Tensor
    :param current_decay: 电流衰减 tensor
    :type current_decay: torch.Tensor
    :param voltage_decay: 电压衰减 tensor
    :type voltage_decay: torch.Tensor
    :param s_scale: Lava 突触缩放因子
    :type s_scale: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_threshold_eps: Lava 阈值近似 epsilon
    :type v_threshold_eps: float
    :param v_reset: hard reset 电压
    :type v_reset: float
    :param surrogate_function: 已选定替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否在 reset 分支中分离 ``spike`` 的计算图
    :type detach_reset: bool
    :return: ``(spike, current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _lava_cuba_lif_step-en:

    * **English**

    Run one state update for the norm-free Torch path of
    ``lava_exchange.CubaLIFNode`` and return spikes, next current state, and
    reset next voltage state. The function does not materialize state, read a
    module, or inspect ``training/eval``.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param current_state: Current current-state tensor
    :type current_state: torch.Tensor
    :param voltage_state: Current voltage-state tensor
    :type voltage_state: torch.Tensor
    :param current_decay: Current-decay tensor
    :type current_decay: torch.Tensor
    :param voltage_decay: Voltage-decay tensor
    :type voltage_decay: torch.Tensor
    :param s_scale: Lava synaptic scale
    :type s_scale: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_threshold_eps: Lava threshold approximation epsilon
    :type v_threshold_eps: float
    :param v_reset: Hard-reset voltage
    :type v_reset: float
    :param surrogate_function: Selected surrogate function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach ``spike`` in the reset branch
    :type detach_reset: bool
    :return: ``(spike, current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    from ..lava_exchange import LeakyIntegratorStep, step_quantize

    current_next = LeakyIntegratorStep.apply(
        x,
        step_quantize(current_decay),
        current_state.contiguous(),
        s_scale,
    )
    voltage_charged = LeakyIntegratorStep.apply(
        current_next,
        step_quantize(voltage_decay),
        voltage_state.contiguous(),
        s_scale,
    )
    spike = surrogate_function(voltage_charged - (v_threshold + v_threshold_eps))
    spike_d = spike.detach() if detach_reset else spike
    voltage_next = spike_d * v_reset + (1.0 - spike_d) * voltage_charged
    return spike, current_next, voltage_next


def _reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
) -> torch.Tensor:
    spike_d = spike.detach() if detach_reset else spike
    if v_reset is None:
        return v - spike_d * v_threshold
    return spike_d * v_reset + (1.0 - spike_d) * v


def _normalize_multi_step_output(
    out: tuple[torch.Tensor, ...],
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if store_v_seq:
        spike_seq, v_next, v_seq = out
        return spike_seq, v_next, v_seq
    spike_seq, v_next = out
    return spike_seq, v_next, None


def if_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <if_step-cn>` | :ref:`English <if_step-en>`

    ----

    .. _if_step-cn:

    * **中文**

    执行一条已确定 Torch 路径的 IF 单步状态转移，返回 ``(spike, v_next)``。
    该函数不读取 module memory，不管理 ``training/eval``，也不原地修改 ``x`` 或
    ``v``。

    :param x: 当前输入张量，shape 通常为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位 tensor state，shape/dtype/device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _if_step-en:

    * **English**

    Run one IF single-step state transition for a selected Torch path and return
    ``(spike, v_next)``. This function does not read module memory, does not manage
    ``training/eval``, and does not mutate ``x`` or ``v`` in place.

    :param x: Current input tensor, conventionally shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Materialized current membrane-voltage tensor state compatible with
        ``x`` in shape, dtype, and device
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    v_charged = v + x
    spike = surrogate_function(v_charged - v_threshold)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def qif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    a0: float,
    v_rest: float,
    v_c: float,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <qif_step-cn>` | :ref:`English <qif_step-en>`

    ----

    .. _qif_step-cn:

    * **中文**

    执行 QIF 神经元的一次显式状态更新。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 当前膜电位，shape、dtype 和 device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param a0: 二次项系数
    :type a0: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_c: 临界电位
    :type v_c: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _qif_step-en:

    * **English**

    Run one explicit QIF neuron state update.

    :param x: Current input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Current voltage compatible with ``x`` in shape, dtype, and device
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param a0: Quadratic coefficient
    :type a0: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_c: Critical voltage
    :type v_c: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    v_charged = v + (x + a0 * (v - v_rest) * (v - v_c)) / tau
    spike = surrogate_function(v_charged - v_threshold)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def eif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    delta_t: float,
    theta_rh: float,
    v_rest: float,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <eif_step-cn>` | :ref:`English <eif_step-en>`

    ----

    .. _eif_step-cn:

    * **中文**

    执行 EIF 神经元的一次显式状态更新。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 当前膜电位，shape、dtype 和 device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param delta_t: 指数项陡峭度
    :type delta_t: float
    :param theta_rh: 基强度阈值
    :type theta_rh: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _eif_step-en:

    * **English**

    Run one explicit EIF neuron state update.

    :param x: Current input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Current voltage compatible with ``x`` in shape, dtype, and device
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param delta_t: Exponential sharpness
    :type delta_t: float
    :param theta_rh: Rheobase threshold
    :type theta_rh: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    v_charged = (
        v + (x + v_rest - v + delta_t * torch.exp((v - theta_rh) / delta_t)) / tau
    )
    spike = surrogate_function(v_charged - v_threshold)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def activation_aware_if_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_offset: torch.Tensor,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <activation_aware_if_step-cn>` | :ref:`English <activation_aware_if_step-en>`

    ----

    .. _activation_aware_if_step-cn:

    * **中文**

    执行一条已确定 Torch 路径上的 activation-aware IF 单步状态转移。函数接收当前
    输入 ``x``、已物化膜电位 ``v``、已广播到当前输入形状的 ``v_threshold`` 和
    ``v_offset``，返回 ``(spike, v_next)``。

    函数不读取或写入 ``MemoryModule`` memory，不负责 ``training/eval``、
    ``step_mode``、backend dispatch 或 channel-wise 参数广播。module 必须先完成
    state 物化和参数广播，再调用该函数。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位 tensor state，shape 与 ``x`` 相同
    :type v: torch.Tensor
    :param v_threshold: 已广播的发放阈值，可为 scalar tensor 或 shape 与 ``x``
        可广播的 tensor
    :type v_threshold: torch.Tensor
    :param v_offset: 已广播的膜电位偏移，可为 scalar tensor 或 shape 与 ``x``
        可广播的 tensor
    :type v_offset: torch.Tensor
    :param v_reset: 硬复位电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _activation_aware_if_step-en:

    * **English**

    Run one activation-aware IF single-step state transition on an already
    selected Torch path. The function receives current input ``x``, materialized
    membrane voltage ``v``, and ``v_threshold`` / ``v_offset`` already broadcast
    for the current input shape, and returns ``(spike, v_next)``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``training/eval``, ``step_mode``, backend dispatch, or channel-wise
    parameter broadcasting. The module must materialize state and broadcast
    parameters before calling this function.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Materialized current membrane-voltage tensor state with the same
        shape as ``x``
    :type v: torch.Tensor
    :param v_threshold: Broadcast threshold, either a scalar tensor or a tensor
        broadcastable to ``x``
    :type v_threshold: torch.Tensor
    :param v_offset: Broadcast membrane offset, either a scalar tensor or a
        tensor broadcastable to ``x``
    :type v_offset: torch.Tensor
    :param v_reset: Hard-reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    h = v + x
    spike = surrogate_function(h + v_offset - v_threshold)
    spike_d = spike.detach() if detach_reset else spike
    if v_reset is None:
        v_next = h - spike_d * v_threshold
    else:
        v_next = spike_d * v_reset + (1.0 - spike_d) * h
    return spike, v_next


def activation_aware_if_multi_step_triton(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_offset: torch.Tensor,
    channel_size: int,
    inner_size: int,
    v_reset: Optional[float],
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <activation_aware_if_multi_step_triton-cn>` | :ref:`English <activation_aware_if_multi_step_triton-en>`

    ----

    .. _activation_aware_if_multi_step_triton-cn:

    * **中文**

    使用已选定的 Triton kernel 执行 activation-aware IF 多步状态转移。调用方必须
    传入已物化的膜电位、已规范化的阈值和偏移张量，以及由输入布局计算得到的
    ``channel_size`` 和 ``inner_size``。函数不负责 backend、``training/eval``、
    surrogate 或 autograd 分支选择。

    :param x_seq: CUDA 输入序列，shape 为 ``[T, N, *]``，dtype 为
        ``torch.float32`` 或 ``torch.bfloat16``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state
    :type v: torch.Tensor
    :param v_threshold: scalar 或逐通道阈值 tensor
    :type v_threshold: torch.Tensor
    :param v_offset: scalar 或逐通道偏移 tensor
    :type v_offset: torch.Tensor
    :param channel_size: 逐通道参数对应的 channel 数；scalar 参数时为 ``1``
    :type channel_size: int
    :param inner_size: 每个 channel 对应的连续元素数
    :type inner_size: int
    :param v_reset: 硬复位电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param store_v_seq: 是否返回 reset 后的完整膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ImportError: Triton kernel 不可用

    ----

    .. _activation_aware_if_multi_step_triton-en:

    * **English**

    Run an activation-aware IF multi-step state transition with the already
    selected Triton kernel. The caller must provide a materialized membrane
    voltage, normalized threshold and offset tensors, and ``channel_size`` /
    ``inner_size`` derived from the input layout. This function does not select
    backend, ``training/eval``, surrogate, or autograd branches.

    :param x_seq: CUDA input sequence shaped ``[T, N, *]`` with dtype
        ``torch.float32`` or ``torch.bfloat16``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane-voltage tensor state
    :type v: torch.Tensor
    :param v_threshold: Scalar or channel-wise threshold tensor
    :type v_threshold: torch.Tensor
    :param v_offset: Scalar or channel-wise offset tensor
    :type v_offset: torch.Tensor
    :param channel_size: Channel count for channel-wise parameters, or ``1``
        for scalar parameters
    :type channel_size: int
    :param inner_size: Number of contiguous elements per channel
    :type inner_size: int
    :param v_reset: Hard-reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param store_v_seq: Whether to return the full post-reset voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ImportError: If the Triton kernel is unavailable
    """
    try:
        from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
            activation_aware_if,
        )
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "activation_aware_if_multi_step_triton requires the Triton backend."
        ) from exc
    if activation_aware_if is None:
        raise ImportError(
            "activation_aware_if_multi_step_triton requires the Triton backend."
        )

    spike_seq, v_out = activation_aware_if._multistep_activation_aware_if(
        x_seq,
        v,
        v_threshold,
        v_offset,
        channel_size=channel_size,
        inner_size=inner_size,
        v_reset=v_reset,
        store_v_seq=store_v_seq,
    )
    if store_v_seq:
        return spike_seq, v_out[-1].clone(), v_out
    return spike_seq, v_out, None


def if_multi_step_inductor(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <if_multi_step_inductor-cn>` | :ref:`English <if_multi_step_inductor-en>`

    ----

    .. _if_multi_step_inductor-cn:

    * **中文**

    使用 ``torch.compile(..., backend="inductor")`` 执行一条已确定的 IF 多步状态
    转移路径。该函数不管理 module memory、``training/eval``、``step_mode`` 或通用
    ``backend`` 分支；调用者必须传入已物化的初始膜电位。

    :param x_seq: 输入序列张量，shape 通常为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回各时间步 reset 后的膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _if_multi_step_inductor-en:

    * **English**

    Run a selected IF multi-step state transition path with
    ``torch.compile(..., backend="inductor")``. This function does not manage
    module memory, ``training/eval``, ``step_mode``, or generic ``backend``
    dispatch; the caller must pass a materialized initial membrane voltage.

    :param x_seq: Input sequence tensor, conventionally shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane-voltage tensor state
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the post-reset membrane voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..neuron import inductor_cache

    x_seq = x_seq.contiguous()
    v = v.contiguous()
    surrogate_key = inductor_cache.surrogate_key(surrogate_function)
    graph = inductor_cache.compile_graph(
        None
        if surrogate_key is None
        else (
            "if",
            store_v_seq,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            inductor_cache.runtime_key(x_seq, v),
        ),
        inductor_cache._build_if_multi_step(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalize_multi_step_output(graph(x_seq, v), store_v_seq)


def lif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lif_step-cn>` | :ref:`English <lif_step-en>`

    ----

    .. _lif_step-cn:

    * **中文**

    执行 LIF 单步状态转移，返回 ``(spike, v_next)``。函数只描述已选定的
    Torch 执行路径，不管理 module 的 ``training/eval`` 分支。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位 tensor state
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _lif_step-en:

    * **English**

    Run one LIF single-step state transition and return ``(spike, v_next)``. This
    function only describes a selected Torch execution path and does not manage
    the module ``training/eval`` branch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Materialized current membrane-voltage tensor state
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        v_charged = v + (x - (v - v_reset_value)) / tau
    else:
        v_charged = v - (v - v_reset_value) / tau + x
    spike = surrogate_function(v_charged - v_threshold)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def lif_multi_step_inductor(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <lif_multi_step_inductor-cn>` | :ref:`English <lif_multi_step_inductor-en>`

    ----

    .. _lif_multi_step_inductor-cn:

    * **中文**

    使用 ``torch.compile(..., backend="inductor")`` 执行一条已确定的 LIF 多步状态
    转移路径。该函数只描述执行过程，不根据 ``training/eval`` 或通用 ``backend``
    参数选择分支。

    :param x_seq: 输入序列张量，shape 通常为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回各时间步 reset 后的膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _lif_multi_step_inductor-en:

    * **English**

    Run a selected LIF multi-step state transition path with
    ``torch.compile(..., backend="inductor")``. This function only describes the
    execution and does not choose branches from ``training/eval`` or a generic
    ``backend`` argument.

    :param x_seq: Input sequence tensor, conventionally shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane-voltage tensor state
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the post-reset membrane voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..neuron import inductor_cache

    x_seq = x_seq.contiguous()
    v = v.contiguous()
    surrogate_key = inductor_cache.surrogate_key(surrogate_function)
    graph = inductor_cache.compile_graph(
        None
        if surrogate_key is None
        else (
            "lif",
            store_v_seq,
            decay_input,
            tau,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            inductor_cache.runtime_key(x_seq, v),
        ),
        inductor_cache._build_lif_multi_step(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalize_multi_step_output(graph(x_seq, v), store_v_seq)


def plif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <plif_step-cn>` | :ref:`English <plif_step-en>`

    ----

    .. _plif_step-cn:

    * **中文**

    执行 PLIF 单步状态转移，返回 ``(spike, v_next)``。``w`` 是显式参数，不从
    module 读取。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位 tensor state
    :type v: torch.Tensor
    :param w: PLIF 的可学习参数
    :type w: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _plif_step-en:

    * **English**

    Run one PLIF single-step state transition and return ``(spike, v_next)``.
    ``w`` is an explicit parameter and is not read from a module.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Materialized current membrane-voltage tensor state
    :type v: torch.Tensor
    :param w: Learnable PLIF parameter
    :type w: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    reciprocal_tau = w.sigmoid()
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        v_charged = v + (x - (v - v_reset_value)) * reciprocal_tau
    else:
        v_charged = v - (v - v_reset_value) * reciprocal_tau + x
    spike = surrogate_function(v_charged - v_threshold)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def plif_multi_step_inductor(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <plif_multi_step_inductor-cn>` | :ref:`English <plif_multi_step_inductor-en>`

    ----

    .. _plif_multi_step_inductor-cn:

    * **中文**

    使用 ``torch.compile(..., backend="inductor")`` 执行一条已确定的 PLIF 多步状态
    转移路径。``w`` 是显式参数，函数不会从 module 读取参数或状态，也不管理
    ``training/eval`` 分支。

    :param x_seq: 输入序列张量，shape 通常为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state
    :type v: torch.Tensor
    :param w: PLIF 的可学习参数
    :type w: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回各时间步 reset 后的膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _plif_multi_step_inductor-en:

    * **English**

    Run a selected PLIF multi-step state transition path with
    ``torch.compile(..., backend="inductor")``. ``w`` is an explicit parameter;
    this function does not read parameters or state from a module and does not
    manage the ``training/eval`` branch.

    :param x_seq: Input sequence tensor, conventionally shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane-voltage tensor state
    :type v: torch.Tensor
    :param w: Learnable PLIF parameter
    :type w: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate function for the selected execution path
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the post-reset membrane voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..neuron import inductor_cache

    x_seq = x_seq.contiguous()
    v = v.contiguous()
    reciprocal_tau = w.sigmoid().to(x_seq).contiguous()
    surrogate_key = inductor_cache.surrogate_key(surrogate_function)
    graph = inductor_cache.compile_graph(
        None
        if surrogate_key is None
        else (
            "plif",
            store_v_seq,
            decay_input,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            inductor_cache.runtime_key(x_seq, v, reciprocal_tau),
        ),
        inductor_cache._build_plif_multi_step(
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalize_multi_step_output(graph(x_seq, v, reciprocal_tau), store_v_seq)


def izhikevich_step(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    a0: float,
    v_rest: float,
    v_c: float,
    tau_w: float,
    a: float,
    b: float,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <izhikevich_step-cn>` | :ref:`English <izhikevich_step-en>`

    ----

    .. _izhikevich_step-cn:

    * **中文**

    执行 Izhikevich 神经元的一次完整状态更新。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param w: 当前适应电流
    :type w: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param a0: 膜电位二次项系数
    :type a0: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_c: 临界电位
    :type v_c: float
    :param tau_w: 适应电流时间常数
    :type tau_w: float
    :param a: 阈下耦合系数
    :type a: float
    :param b: 脉冲触发的适应电流增量
    :type b: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离膜电位 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next, w_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _izhikevich_step-en:

    * **English**

    Run one complete Izhikevich neuron state update.

    :param x: Current input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param w: Current adaptation current
    :type w: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param a0: Quadratic voltage coefficient
    :type a0: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_c: Critical voltage
    :type v_c: float
    :param tau_w: Adaptation-current time constant
    :type tau_w: float
    :param a: Subthreshold coupling coefficient
    :type a: float
    :param b: Spike-triggered adaptation increment
    :type b: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the voltage-reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next, w_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    v_charged = v + (x + a0 * (v - v_rest) * (v - v_c) - w) / tau
    w_next = w + (a * (v_charged - v_rest) - w) / tau_w
    spike = surrogate_function(v_charged - v_threshold)
    spike_d = spike.detach() if detach_reset else spike
    if v_reset is None:
        v_next = v_charged - spike_d * v_threshold
    else:
        v_next = (1.0 - spike_d) * v_charged + spike * v_reset
    return spike, v_next, w_next + b * spike


def klif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    tau: float,
    decay_input: bool,
    scale_reset: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <klif_step-cn>` | :ref:`English <klif_step-en>`

    ----

    .. _klif_step-cn:

    * **中文**

    执行 KLIF 神经元的一次显式状态更新。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param k: 可学习缩放参数
    :type k: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param scale_reset: reset 是否在除以 ``k`` 后的电位域执行
    :type scale_reset: bool
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _klif_step-en:

    * **English**

    Run one explicit KLIF neuron state update.

    :param x: Current input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param k: Learnable scaling parameter
    :type k: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param scale_reset: Whether reset operates after dividing voltage by ``k``
    :type scale_reset: bool
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        v_charged = v + (x - (v - v_reset_value)) / tau
    else:
        v_charged = v - (v - v_reset_value) / tau + x
    v_charged = torch.relu(k * v_charged)
    spike = surrogate_function(v_charged - v_threshold)
    if scale_reset:
        return spike, _reset(
            v_charged / k,
            spike,
            v_threshold / k,
            v_reset,
            detach_reset,
        )
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def cuba_lif_step(
    x: torch.Tensor,
    current: torch.Tensor,
    v: torch.Tensor,
    current_decay: float,
    voltage_decay: float,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <cuba_lif_step-cn>` | :ref:`English <cuba_lif_step-en>`

    ----

    .. _cuba_lif_step-cn:

    * **中文**

    执行 current-based LIF 神经元的一次显式状态更新。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param current: 当前突触电流
    :type current: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param current_decay: 突触电流衰减系数
    :type current_decay: float
    :param voltage_decay: 膜电位衰减系数
    :type voltage_decay: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, current_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _cuba_lif_step-en:

    * **English**

    Run one explicit current-based LIF neuron state update.

    :param x: Current input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param current: Current synaptic current
    :type current: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param current_decay: Synaptic-current decay
    :type current_decay: float
    :param voltage_decay: Membrane-voltage decay
    :type voltage_decay: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, current_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    current_next = current * current_decay + x
    v_charged = v * voltage_decay + current_next
    spike = surrogate_function(v_charged - v_threshold)
    v_next = _reset(v_charged, spike, v_threshold, v_reset, detach_reset)
    return spike, current_next, v_next


def if_step_cupy(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    forward_kernel: Any,
    backward_kernel: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <if_step_cupy-cn>` | :ref:`English <if_step_cupy-en>`

    ----

    .. _if_step_cupy-cn:

    * **中文**

    使用调用方已选定的 CuPy kernel 执行 IF 单步状态转移。函数接受标准 shape
    的 tensor，并在调用底层 kernel 时自行展平和恢复 shape；它不负责 backend
    选择或 kernel 缓存。

    :param x: 当前 CUDA 输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 已物化且与 ``x`` shape 相同的膜电位
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param forward_kernel: 调用方已创建的 IF CuPy forward kernel
    :type forward_kernel: Any
    :param backward_kernel: 调用方已创建的 IF CuPy backward kernel
    :type backward_kernel: Any
    :return: ``(spike, v_next)``，两者 shape 均与 ``x`` 相同
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _if_step_cupy-en:

    * **English**

    Run one IF state transition with caller-selected CuPy kernels. The function
    accepts tensors in their standard shape and handles flattening and shape
    restoration around the low-level kernel. It does not select a backend or
    cache kernels.

    :param x: Current CUDA input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with the same shape as ``x``
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param forward_kernel: Caller-created IF CuPy forward kernel
    :type forward_kernel: Any
    :param backward_kernel: Caller-created IF CuPy backward kernel
    :type backward_kernel: Any
    :return: ``(spike, v_next)``, both shaped like ``x``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    from ..cuda_kernel.auto_cuda import ss_neuron_kernel

    spike, v_next = ss_neuron_kernel.ss_if_step(
        x.flatten(),
        v.flatten(),
        v_threshold,
        v_reset,
        forward_kernel,
        backward_kernel,
    )
    return spike.reshape_as(x), v_next.reshape_as(v)


def lif_step_cupy(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    forward_kernel: Any,
    backward_kernel: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lif_step_cupy-cn>` | :ref:`English <lif_step_cupy-en>`

    ----

    .. _lif_step_cupy-cn:

    * **中文**

    使用调用方已选定的 CuPy kernel 执行 LIF 单步状态转移。``forward_kernel``
    已包含 ``decay_input`` 配置；函数接受标准 shape 的 tensor，并在调用底层
    kernel 时自行展平和恢复 shape。

    :param x: 当前 CUDA 输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 已物化且与 ``x`` shape 相同的膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param forward_kernel: 调用方已创建的 LIF CuPy forward kernel
    :type forward_kernel: Any
    :param backward_kernel: 调用方已创建的 LIF CuPy backward kernel
    :type backward_kernel: Any
    :return: ``(spike, v_next)``，两者 shape 均与 ``x`` 相同
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _lif_step_cupy-en:

    * **English**

    Run one LIF state transition with caller-selected CuPy kernels. The
    ``forward_kernel`` already contains the ``decay_input`` configuration. The
    function accepts tensors in their standard shape and handles flattening and
    shape restoration around the low-level kernel.

    :param x: Current CUDA input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with the same shape as ``x``
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param forward_kernel: Caller-created LIF CuPy forward kernel
    :type forward_kernel: Any
    :param backward_kernel: Caller-created LIF CuPy backward kernel
    :type backward_kernel: Any
    :return: ``(spike, v_next)``, both shaped like ``x``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    from ..cuda_kernel.auto_cuda import ss_neuron_kernel

    spike, v_next = ss_neuron_kernel.ss_lif_step(
        x.flatten(),
        v.flatten(),
        v_threshold,
        v_reset,
        1.0 / tau,
        forward_kernel,
        backward_kernel,
    )
    return spike.reshape_as(x), v_next.reshape_as(v)


def if_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <if_multi_step_cupy-cn>` | :ref:`English <if_multi_step_cupy-en>`

    ----

    .. _if_multi_step_cupy-cn:

    * **中文**

    使用 CuPy kernel 执行 IF 多步状态转移，不进行通用 backend 分发。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _if_multi_step_cupy-en:

    * **English**

    Run an IF multi-step state transition with the CuPy kernel without generic
    backend dispatch.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..cuda_kernel.auto_cuda import neuron_kernel

    spike_seq, v_seq = neuron_kernel.multistep_if(
        x_seq.flatten(1),
        v.flatten(),
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape_as(x_seq)
    v_seq = v_seq.reshape_as(x_seq)
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def lif_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <lif_multi_step_cupy-cn>` | :ref:`English <lif_multi_step_cupy-en>`

    ----

    .. _lif_multi_step_cupy-cn:

    * **中文**

    使用 CuPy kernel 执行 LIF 多步状态转移，不进行通用 backend 分发。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _lif_multi_step_cupy-en:

    * **English**

    Run a LIF multi-step state transition with the CuPy kernel without generic
    backend dispatch.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..cuda_kernel.auto_cuda import neuron_kernel

    spike_seq, v_seq = neuron_kernel.multistep_lif(
        x_seq.flatten(1),
        v.flatten(),
        decay_input,
        tau,
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape_as(x_seq)
    v_seq = v_seq.reshape_as(x_seq)
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def plif_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <plif_multi_step_cupy-cn>` | :ref:`English <plif_multi_step_cupy-en>`

    ----

    .. _plif_multi_step_cupy-cn:

    * **中文**

    使用 CuPy kernel 执行 PLIF 多步状态转移；``w`` 与 torch/Inductor 接口语义一致。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param w: PLIF 可学习参数
    :type w: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _plif_multi_step_cupy-en:

    * **English**

    Run a PLIF multi-step state transition with the CuPy kernel. ``w`` has the
    same semantics as in the torch and Inductor interfaces.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param w: Learnable PLIF parameter
    :type w: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..cuda_kernel.auto_cuda import neuron_kernel

    spike_seq, v_seq = neuron_kernel.multistep_plif(
        x_seq.flatten(1),
        v.flatten(),
        w.sigmoid().to(x_seq),
        decay_input,
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape_as(x_seq)
    v_seq = v_seq.reshape_as(x_seq)
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def qif_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    v_rest: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    surrogate_function: SurrogateFunction,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <qif_multi_step_cupy-cn>` | :ref:`English <qif_multi_step_cupy-en>`

    ----

    .. _qif_multi_step_cupy-cn:

    * **中文**

    使用已选定的 CuPy kernel 执行 QIF 多步状态转移。函数接收已物化的膜电位，
    不负责 backend 或 ``training/eval`` 分支选择。

    :param x_seq: CUDA 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_c: 临界电位
    :type v_c: float
    :param a0: 二次项系数
    :type a0: float
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param surrogate_function: CuPy kernel 使用的替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_v_seq: 是否返回完整膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _qif_multi_step_cupy-en:

    * **English**

    Run a QIF multi-step state transition with the already selected CuPy
    kernel. The function receives a materialized membrane voltage and does not
    select backend or ``training/eval`` branches.

    :param x_seq: CUDA input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_c: Critical voltage
    :type v_c: float
    :param a0: Quadratic coefficient
    :type a0: float
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param surrogate_function: Surrogate-gradient function used by the CuPy kernel
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_v_seq: Whether to return the full membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from .. import cuda_kernel

    spike_seq, v_seq = cuda_kernel.multistep_qif_ptt(
        x_seq.flatten(1),
        v.flatten(0),
        tau,
        v_threshold,
        v_reset,
        v_rest,
        v_c,
        a0,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape(x_seq.shape)
    v_seq = v_seq.reshape(x_seq.shape)
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def eif_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    v_rest: float,
    theta_rh: float,
    delta_t: float,
    detach_reset: bool,
    surrogate_function: SurrogateFunction,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <eif_multi_step_cupy-cn>` | :ref:`English <eif_multi_step_cupy-en>`

    ----

    .. _eif_multi_step_cupy-cn:

    * **中文**

    使用已选定的 CuPy kernel 执行 EIF 多步状态转移。函数接收已物化的膜电位，
    不负责 backend 或 ``training/eval`` 分支选择。

    :param x_seq: CUDA 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param v_rest: 静息电位
    :type v_rest: float
    :param theta_rh: 基强度阈值
    :type theta_rh: float
    :param delta_t: 指数项陡峭度
    :type delta_t: float
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param surrogate_function: CuPy kernel 使用的替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_v_seq: 是否返回完整膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _eif_multi_step_cupy-en:

    * **English**

    Run an EIF multi-step state transition with the already selected CuPy
    kernel. The function receives a materialized membrane voltage and does not
    select backend or ``training/eval`` branches.

    :param x_seq: CUDA input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param v_rest: Resting voltage
    :type v_rest: float
    :param theta_rh: Rheobase threshold
    :type theta_rh: float
    :param delta_t: Exponential slope factor
    :type delta_t: float
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param surrogate_function: Surrogate-gradient function used by the CuPy kernel
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_v_seq: Whether to return the full membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from .. import cuda_kernel

    spike_seq, v_seq = cuda_kernel.multistep_eif_ptt(
        x_seq.flatten(1),
        v.flatten(0),
        tau,
        v_threshold,
        v_reset,
        v_rest,
        theta_rh,
        delta_t,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape(x_seq.shape)
    v_seq = v_seq.reshape(x_seq.shape)
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def izhikevich_multi_step_cupy(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    v_rest: float,
    a: float,
    b: float,
    tau_w: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    surrogate_function: SurrogateFunction,
    store_state_seq: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    r"""
    **API Language** - :ref:`中文 <izhikevich_multi_step_cupy-cn>` | :ref:`English <izhikevich_multi_step_cupy-en>`

    ----

    .. _izhikevich_multi_step_cupy-cn:

    * **中文**

    使用已选定的 CuPy kernel 执行 Izhikevich 多步状态转移，显式接收和返回膜电位
    ``v`` 与适应变量 ``w``。

    :param x_seq: CUDA 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param w: 已物化的初始适应变量
    :type w: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param v_rest: 静息电位
    :type v_rest: float
    :param a: 适应变量的恢复系数
    :type a: float
    :param b: 适应变量对膜电位的敏感系数
    :type b: float
    :param tau_w: 适应变量时间常数
    :type tau_w: float
    :param v_c: 临界电位
    :type v_c: float
    :param a0: reset 时适应变量的增量
    :type a0: float
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param surrogate_function: CuPy kernel 使用的替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_state_seq: 是否返回完整 ``v``、``w`` 序列
    :type store_state_seq: bool
    :return: ``(spike_seq, v_next, w_next, v_seq_or_none, w_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]

    ----

    .. _izhikevich_multi_step_cupy-en:

    * **English**

    Run an Izhikevich multi-step state transition with the already selected
    CuPy kernel, explicitly receiving and returning membrane voltage ``v`` and
    adaptation state ``w``.

    :param x_seq: CUDA input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param w: Materialized initial adaptation state
    :type w: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param v_rest: Resting voltage
    :type v_rest: float
    :param a: Adaptation recovery coefficient
    :type a: float
    :param b: Adaptation sensitivity to membrane voltage
    :type b: float
    :param tau_w: Adaptation time constant
    :type tau_w: float
    :param v_c: Critical voltage
    :type v_c: float
    :param a0: Adaptation increment on reset
    :type a0: float
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param surrogate_function: Surrogate-gradient function used by the CuPy kernel
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_state_seq: Whether to return the full ``v`` and ``w`` sequences
    :type store_state_seq: bool
    :return: ``(spike_seq, v_next, w_next, v_seq_or_none, w_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    """
    from .. import cuda_kernel

    spike_seq, v_seq, w_seq = cuda_kernel.multistep_izhikevich_ptt(
        x_seq.flatten(1),
        v.flatten(0),
        w.flatten(0),
        tau,
        v_threshold,
        v_reset,
        v_rest,
        a,
        b,
        tau_w,
        v_c,
        a0,
        detach_reset,
        surrogate_function,
    )
    spike_seq = spike_seq.reshape(x_seq.shape)
    v_seq = v_seq.reshape(x_seq.shape)
    w_seq = w_seq.reshape(x_seq.shape)
    return (
        spike_seq,
        v_seq[-1].clone(),
        w_seq[-1].clone(),
        v_seq if store_state_seq else None,
        w_seq if store_state_seq else None,
    )


def if_multi_step_triton(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <if_multi_step_triton-cn>` | :ref:`English <if_multi_step_triton-en>`

    ----

    .. _if_multi_step_triton-cn:

    * **中文**

    使用 Triton kernel 执行 IF 多步状态转移，不进行通用 backend 分发。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _if_multi_step_triton-en:

    * **English**

    Run an IF multi-step state transition with the Triton kernel without generic
    backend dispatch.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..triton_kernel import multistep_if

    spike_seq, voltage = multistep_if(
        x_seq,
        v,
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
        store_v_seq,
    )
    if store_v_seq:
        return spike_seq, voltage[-1], voltage
    return spike_seq, voltage, None


def lif_multi_step_triton(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <lif_multi_step_triton-cn>` | :ref:`English <lif_multi_step_triton-en>`

    ----

    .. _lif_multi_step_triton-cn:

    * **中文**

    使用 Triton kernel 执行 LIF 多步状态转移，不进行通用 backend 分发。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _lif_multi_step_triton-en:

    * **English**

    Run a LIF multi-step state transition with the Triton kernel without generic
    backend dispatch.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..triton_kernel import multistep_lif

    spike_seq, voltage = multistep_lif(
        x_seq,
        v,
        decay_input,
        tau,
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
        store_v_seq,
    )
    if store_v_seq:
        return spike_seq, voltage[-1], voltage
    return spike_seq, voltage, None


def plif_multi_step_triton(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <plif_multi_step_triton-cn>` | :ref:`English <plif_multi_step_triton-en>`

    ----

    .. _plif_multi_step_triton-cn:

    * **中文**

    使用 Triton kernel 执行 PLIF 多步状态转移；``w`` 与 torch/Inductor 接口语义一致。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位
    :type v: torch.Tensor
    :param w: PLIF 可学习参数
    :type w: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _plif_multi_step_triton-en:

    * **English**

    Run a PLIF multi-step state transition with the Triton kernel. ``w`` has the
    same semantics as in the torch and Inductor interfaces.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage
    :type v: torch.Tensor
    :param w: Learnable PLIF parameter
    :type w: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    from ..triton_kernel import multistep_plif

    spike_seq, v_seq = multistep_plif(
        x_seq,
        v,
        w.sigmoid().to(x_seq),
        decay_input,
        v_threshold,
        v_reset,
        detach_reset,
        surrogate_function,
    )
    return spike_seq, v_seq[-1].clone(), v_seq if store_v_seq else None


def sliding_psn_step(
    x: torch.Tensor,
    queue: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    r"""
    **API Language** - :ref:`中文 <sliding_psn_step-cn>` | :ref:`English <sliding_psn_step-en>`

    ----

    .. _sliding_psn_step-cn:

    * **中文**

    执行 ``SlidingPSN`` 的单步显式 queue 状态转移。函数接收当前输入 ``x``、
    旧 queue、权重、偏置和替代函数，将 ``x.flatten()`` 追加到 queue 末尾；若
    queue 长度超过 ``weight.numel()``，只弹出最旧的一个元素，以保持既有 module
    对异常外部 queue 状态的行为。随后函数用最近 queue 与尾部权重计算膜电位，
    返回 ``(spike, queue_next)``。

    函数不读取或写入 ``MemoryModule`` memory，不负责 ``step_mode``、
    ``training/eval`` 或 backend dispatch，也不原地修改传入 queue。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param queue: 旧 queue state，元素是已经 flatten 的输入 tensor，按旧到新排列
    :type queue: Tuple[torch.Tensor, ...]
    :param weight: ``SlidingPSN`` 权重，shape 为 ``[k]``
    :type weight: torch.Tensor
    :param bias: ``SlidingPSN`` 偏置，标量 tensor
    :type bias: torch.Tensor
    :param surrogate_function: 作用于 ``h + bias`` 的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, queue_next)``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]

    ----

    .. _sliding_psn_step-en:

    * **English**

    Run one explicit queue-state transition for ``SlidingPSN``. The function
    receives the current input ``x``, previous queue, weight, bias, and surrogate
    function. It appends ``x.flatten()`` to the queue; if the queue length
    exceeds ``weight.numel()``, it pops only the oldest item to preserve the
    existing module behavior for externally corrupted overlong queues. It then
    computes the membrane potential from the recent queue and tail weights, and
    returns ``(spike, queue_next)``.

    The function does not read or write ``MemoryModule`` memory, does not manage
    ``step_mode``, ``training/eval``, or backend dispatch, and does not mutate
    the input queue in place.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param queue: Previous queue state with flattened input tensors ordered from
        oldest to newest
    :type queue: Tuple[torch.Tensor, ...]
    :param weight: ``SlidingPSN`` weight shaped ``[k]``
    :type weight: torch.Tensor
    :param bias: ``SlidingPSN`` scalar bias tensor
    :type bias: torch.Tensor
    :param surrogate_function: Surrogate function applied to ``h + bias``
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, queue_next)``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
    """
    k = weight.numel()
    queue_next = (*queue, x.flatten())
    if len(queue_next) > k:
        queue_next = queue_next[1:]

    psn_weight = weight[k - len(queue_next) : k].unsqueeze(-1)
    x_seq = torch.stack(queue_next)
    h = torch.sum(psn_weight * x_seq, 0)
    spike = surrogate_function(h + bias)
    return spike.view(x.shape), queue_next


def gated_lif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    time_steps: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    tau: torch.Tensor,
    v_threshold: torch.Tensor,
    linear_decay: torch.Tensor,
    v_subreset: torch.Tensor,
    conduct: torch.Tensor,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <gated_lif_multi_step-cn>` | :ref:`English <gated_lif_multi_step-en>`

    ----

    .. _gated_lif_multi_step-cn:

    * **中文**

    执行 ``GatedLIFNode`` 的 Torch 多步显式状态转移。函数接收输入序列、
    已物化的初始膜电位 ``v``、时间步数、门控参数和替代函数，返回
    ``(spike_seq, u_next, v_next)``。``u`` 在现有 GLIF 动力学中每个时间步都会由
    charge 公式覆盖，因此函数不接收旧 ``u``；它只返回最后一个时间步写回 module
    memory 的 ``u_next``。

    函数不读取或写入 ``MemoryModule`` memory，不负责 ``training/eval``、
    ``step_mode`` 或 backend dispatch。调用者必须传入当前 module 已选定执行路径的
    参数和替代函数。

    :param x_seq: 输入序列，现有 ``GatedLIFNode`` 约定 shape 为 ``[T, N, C, H, W]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位，shape 可广播到 ``x_seq[0]``
    :type v: torch.Tensor
    :param time_steps: 执行时间步数，保持既有 ``GatedLIFNode`` 的 ``self.T`` 语义
    :type time_steps: int
    :param alpha: 门控参数 ``alpha``，原始 parameter tensor
    :type alpha: torch.Tensor
    :param beta: 门控参数 ``beta``，原始 parameter tensor
    :type beta: torch.Tensor
    :param gamma: 门控参数 ``gamma``，原始 parameter tensor
    :type gamma: torch.Tensor
    :param tau: 膜电位衰减参数，原始 parameter tensor
    :type tau: torch.Tensor
    :param v_threshold: 阈值参数，原始 parameter tensor
    :type v_threshold: torch.Tensor
    :param linear_decay: 线性衰减参数，原始 parameter tensor
    :type linear_decay: torch.Tensor
    :param v_subreset: soft-reset 参数，原始 parameter tensor
    :type v_subreset: torch.Tensor
    :param conduct: 电导参数，shape 为 ``[T]`` 或 ``[T, C]``
    :type conduct: torch.Tensor
    :param surrogate_function: 作用于 ``u - sigmoid(v_threshold)`` 的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike_seq, u_next, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _gated_lif_multi_step-en:

    * **English**

    Run the explicit Torch multi-step state transition for ``GatedLIFNode``. The
    function receives an input sequence, materialized initial membrane voltage
    ``v``, time-step count, gate parameters, and surrogate function, and returns
    ``(spike_seq, u_next, v_next)``. Existing GLIF dynamics overwrite ``u`` in
    the charge formula at every time step, so the function does not receive the
    old ``u``; it only returns the final ``u_next`` that the module writes back
    to memory.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``training/eval``, ``step_mode``, or backend dispatch. The caller must
    pass the parameters and surrogate function for the execution path already
    selected by the owning module.

    :param x_seq: Input sequence. Existing ``GatedLIFNode`` expects shape
        ``[T, N, C, H, W]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane voltage, broadcastable to
        ``x_seq[0]``
    :type v: torch.Tensor
    :param time_steps: Number of time steps to execute, preserving the existing
        ``GatedLIFNode`` ``self.T`` semantics
    :type time_steps: int
    :param alpha: Raw ``alpha`` gate parameter tensor
    :type alpha: torch.Tensor
    :param beta: Raw ``beta`` gate parameter tensor
    :type beta: torch.Tensor
    :param gamma: Raw ``gamma`` gate parameter tensor
    :type gamma: torch.Tensor
    :param tau: Raw membrane-decay parameter tensor
    :type tau: torch.Tensor
    :param v_threshold: Raw threshold parameter tensor
    :type v_threshold: torch.Tensor
    :param linear_decay: Raw linear-decay parameter tensor
    :type linear_decay: torch.Tensor
    :param v_subreset: Raw soft-reset parameter tensor
    :type v_subreset: torch.Tensor
    :param conduct: Conductance parameter shaped ``[T]`` or ``[T, C]``
    :type conduct: torch.Tensor
    :param surrogate_function: Surrogate function applied to
        ``u - sigmoid(v_threshold)``
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike_seq, u_next, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    alpha = alpha.view(1, -1, 1, 1).sigmoid()
    beta = beta.view(1, -1, 1, 1).sigmoid()
    gamma = gamma.view(1, -1, 1, 1).sigmoid()
    tau = tau.view(1, -1, 1, 1).sigmoid()
    v_threshold = v_threshold.view(1, -1, 1, 1).sigmoid()
    linear_decay = linear_decay.view(1, -1, 1, 1).sigmoid()
    v_subreset = v_subreset.view(1, -1, 1, 1).sigmoid()

    spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
    spike_seq = []
    u = v
    for t in range(time_steps):
        conduct_t = conduct[t].view(1, -1, 1, 1).sigmoid()
        input_current = x_seq[t] * (1 - beta * (1 - conduct_t))
        u = ((1 - alpha * (1 - tau)) * v - (1 - alpha) * linear_decay) + input_current
        u = (
            u
            - (1 - alpha * (1 - tau)) * v * gamma * spike
            - (1 - gamma) * v_subreset * spike
        )
        spike = surrogate_function(u - v_threshold)
        v = u
        spike_seq.append(spike)
    return torch.stack(spike_seq), u, v


def _stbif_step(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = x / q_threshold
    q_next = q + normalized.detach()
    acc_q_next = torch.round(acc_q)
    spike_position = (q_next - 1 >= 0) & (acc_q_next < pos_max)
    neg_spike_position = (q_next < 0) & (acc_q_next > neg_min)
    cur_output_next = spike_position.to(x.dtype) - neg_spike_position.to(x.dtype)
    acc_q_next = acc_q_next + cur_output_next
    q_next = torch.where(spike_position, q_next - 1, q_next)
    q_next = torch.where(neg_spike_position, q_next + 1, q_next)
    return cur_output_next * q_threshold, q_next, acc_q_next, cur_output_next


def stbif_step(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    r"""
    **API Language** - :ref:`中文 <stbif_step-cn>` | :ref:`English <stbif_step-en>`

    ----

    .. _stbif_step-cn:

    * **中文**

    执行 ``STBIFNeuron`` 的单步显式状态转移。函数接收当前输入 ``x``、已物化的
    量化残差 ``q`` 和累计释放量 ``acc_q``，以及已转换到当前 device/dtype 的
    ``q_threshold``、``pos_max``、``neg_min``，返回
    ``(out, q_next, acc_q_next, cur_output_next, is_work)``。

    该函数保持 SpikeZIP STBIF 的推理语义：输入先除以 ``q_threshold``，累加到
    ``q`` 时使用 ``detach``；``acc_q`` 在判断边界前执行 ``round``；输出
    ``cur_output_next * q_threshold``。函数不读取或写入 ``MemoryModule`` memory，
    不负责 ``training/eval``、``step_mode`` 或 backend dispatch，也不原地修改传入
    state。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param q: 当前量化残差 state，shape 与 ``x`` 相同
    :type q: torch.Tensor
    :param acc_q: 当前累计释放量 state，shape 与 ``x`` 相同
    :type acc_q: torch.Tensor
    :param q_threshold: 当前执行 dtype/device 上的量化 scale
    :type q_threshold: torch.Tensor
    :param pos_max: 正向累计量化上界
    :type pos_max: torch.Tensor
    :param neg_min: 负向累计量化下界
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output_next, is_work)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]

    ----

    .. _stbif_step-en:

    * **English**

    Run one explicit state transition for ``STBIFNeuron``. The function receives
    current input ``x``, materialized quantized residual ``q`` and accumulated
    released quantity ``acc_q``, plus ``q_threshold``, ``pos_max``, and
    ``neg_min`` already converted to the current device/dtype. It returns
    ``(out, q_next, acc_q_next, cur_output_next, is_work)``.

    This function preserves SpikeZIP STBIF inference semantics: the input is
    divided by ``q_threshold``; the addition into ``q`` uses ``detach``;
    ``acc_q`` is rounded before bound checks; and the output is
    ``cur_output_next * q_threshold``. The function does not read or write
    ``MemoryModule`` memory, does not manage ``training/eval``, ``step_mode``, or
    backend dispatch, and does not mutate input states in place.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param q: Current quantized residual state with the same shape as ``x``
    :type q: torch.Tensor
    :param acc_q: Current accumulated released-quantity state with the same
        shape as ``x``
    :type acc_q: torch.Tensor
    :param q_threshold: Quantization scale on the current execution dtype/device
    :type q_threshold: torch.Tensor
    :param pos_max: Positive accumulated quantization bound
    :type pos_max: torch.Tensor
    :param neg_min: Negative accumulated quantization bound
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output_next, is_work)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]
    """
    out, q_next, acc_q_next, cur_output_next = _stbif_step(
        x, q, acc_q, q_threshold, pos_max, neg_min
    )
    is_work = bool((x != 0).any() | (out != 0).any())
    return out, q_next, acc_q_next, cur_output_next, is_work


def stbif_multi_step_torch(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    r"""
    **API Language** - :ref:`中文 <stbif_multi_step_torch-cn>` | :ref:`English <stbif_multi_step_torch-en>`

    ----

    .. _stbif_multi_step_torch-cn:

    * **中文**

    执行 ``STBIFNeuron`` 的 Torch 多步显式状态转移。函数接收输入序列、已物化
    ``q``/``acc_q`` state 和边界参数，返回
    ``(out_seq, q_next, acc_q_next, cur_output_next, is_work)``。该函数只描述
    Torch 路径；Triton backend 的选择和调用仍由 module 管理。

    :param x_seq: 输入序列，shape 为 ``[T, ...]``
    :type x_seq: torch.Tensor
    :param q: 初始量化残差 state，shape 与 ``x_seq[0]`` 相同
    :type q: torch.Tensor
    :param acc_q: 初始累计释放量 state，shape 与 ``x_seq[0]`` 相同
    :type acc_q: torch.Tensor
    :param q_threshold: 当前执行 dtype/device 上的量化 scale
    :type q_threshold: torch.Tensor
    :param pos_max: 正向累计量化上界
    :type pos_max: torch.Tensor
    :param neg_min: 负向累计量化下界
    :type neg_min: torch.Tensor
    :return: ``(out_seq, q_next, acc_q_next, cur_output_next, is_work)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]

    ----

    .. _stbif_multi_step_torch-en:

    * **English**

    Run the explicit Torch multi-step state transition for ``STBIFNeuron``. The
    function receives an input sequence, materialized ``q``/``acc_q`` states, and
    bound parameters, and returns
    ``(out_seq, q_next, acc_q_next, cur_output_next, is_work)``. This function
    describes only the Torch path; selecting and calling the Triton backend
    remains the module's responsibility.

    :param x_seq: Input sequence shaped ``[T, ...]``
    :type x_seq: torch.Tensor
    :param q: Initial quantized residual state with the same shape as
        ``x_seq[0]``
    :type q: torch.Tensor
    :param acc_q: Initial accumulated released-quantity state with the same shape
        as ``x_seq[0]``
    :type acc_q: torch.Tensor
    :param q_threshold: Quantization scale on the current execution dtype/device
    :type q_threshold: torch.Tensor
    :param pos_max: Positive accumulated quantization bound
    :type pos_max: torch.Tensor
    :param neg_min: Negative accumulated quantization bound
    :type neg_min: torch.Tensor
    :return: ``(out_seq, q_next, acc_q_next, cur_output_next, is_work)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]
    """
    q_next = q
    acc_q_next = acc_q
    cur_output_next = torch.zeros_like(x_seq[0])
    out_seq = torch.empty_like(x_seq)
    for t in range(x_seq.shape[0]):
        out_seq[t], q_next, acc_q_next, cur_output_next = _stbif_step(
            x_seq[t], q_next, acc_q_next, q_threshold, pos_max, neg_min
        )
    is_work = bool((x_seq != 0).any() | (out_seq != 0).any())
    return out_seq, q_next, acc_q_next, cur_output_next, is_work
