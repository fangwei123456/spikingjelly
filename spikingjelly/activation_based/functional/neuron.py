r"""显式的神经元状态转移函数 / Explicit neuron state-transition functions.

单步函数显式接收当前输入和状态，不读取或写入 ``MemoryModule`` 的 memory。
其共同契约是：

.. math::

   (Y_t, V_t) = \Phi(X_t, V_{t-1}; \theta)

其中 :math:`V_t` 可以是一个或多个状态张量；具体神经元自行定义
:math:`\Phi`，不要求拆分为固定的充电、发放和复位步骤。状态生命周期由拥有它的
neuron module 管理。

多步函数接收时间优先的 ``[T, ...]`` 序列，返回输出序列和最终状态。支持
``store_v_seq=True`` 的后端还会返回可选的状态轨迹 ``v_seq``；它是调试或监控用的
派生结果，不是需要传给下一次调用的持久化状态。后端选择由神经元 module 负责，
这里的 CuPy/Triton 函数只执行已经选定的路径。

Single-step functions explicitly receive the current input and state. They do not read
or write ``MemoryModule`` memory. Their common contract is:

.. math::

   (Y_t, V_t) = \Phi(X_t, V_{t-1}; \theta)

``V_t`` may contain one or more state tensors. Each neuron defines ``Phi`` as needed;
the transition is not required to split into fixed charging, firing, and reset steps.
The owning neuron module manages state lifetime.

Multi-step functions consume time-major ``[T, ...]`` sequences and return the output
sequence and final state. Backends supporting ``store_v_seq=True`` additionally return
the optional state trace ``v_seq``. This trace is derived monitoring data, not persistent
state for the next call. Backend dispatch belongs to the neuron module; CuPy/Triton
functions here execute an already selected path.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


__all__ = [
    "lava_cuba_lif_step",
    "if_step",
    "qif_step",
    "eif_step",
    "lif_charge",
    "lif_step",
    "liaf_step",
    "ilif_step",
    "plif_step",
    "izhikevich_step",
    "klif_step",
    "cuba_lif_step",
    "if_step_cupy",
    "lif_step_cupy",
    "if_multi_step_cupy",
    "lif_multi_step_cupy",
    "plif_multi_step_cupy",
    "qif_multi_step_cupy",
    "eif_multi_step_cupy",
    "izhikevich_multi_step_cupy",
    "if_multi_step_triton",
    "ilif_multi_step_triton",
    "lif_multi_step_triton",
    "plif_multi_step_triton",
    "stbif_single_step_triton",
    "sliding_psn_step",
    "masked_psn_step",
    "gated_lif_step",
    "stbif_step",
    "activation_aware_if_step",
    "activation_aware_if_multi_step_triton",
    "voltage_reset",
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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
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
    voltage_next = voltage_reset(
        voltage_charged,
        spike,
        v_threshold,
        v_reset,
        detach_reset,
    )
    return spike, current_next, voltage_next


def voltage_reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <voltage_reset-cn>` | :ref:`English <voltage_reset-en>`

    ----

    .. _voltage_reset-cn:

    * **中文**

    根据脉冲对膜电位执行硬重置或软重置。

    :param v: 重置前的膜电位
    :type v: torch.Tensor
    :param spike: 当前时间步的脉冲
    :type spike: torch.Tensor
    :param v_threshold: soft reset 时从膜电位减去的阈值
    :type v_threshold: float
    :param v_reset: hard reset 的目标电位；``None`` 表示使用 soft reset
    :type v_reset: Optional[float]
    :param detach_reset: 是否在重置分支中分离脉冲梯度
    :type detach_reset: bool
    :return: 重置后的膜电位
    :rtype: torch.Tensor

    ----

    .. _voltage_reset-en:

    * **English**

    Apply a hard or soft reset to membrane voltage according to the spike.

    :param v: Membrane voltage before reset
    :type v: torch.Tensor
    :param spike: Spike at the current time step
    :type spike: torch.Tensor
    :param v_threshold: Threshold subtracted by soft reset
    :type v_threshold: float
    :param v_reset: Target voltage for hard reset; ``None`` selects soft reset
    :type v_reset: Optional[float]
    :param detach_reset: Whether to detach the spike in the reset branch
    :type detach_reset: bool
    :return: Membrane voltage after reset
    :rtype: torch.Tensor
    """
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

    .. seealso::

       独立多步形式 / Independent multi-step forms:
       :func:`if_multi_step_cupy`, :func:`if_multi_step_triton`.
    """
    v_charged = v + x
    spike = surrogate_function(v_charged - v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


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

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`qif_multi_step_cupy`.
    """
    v_charged = v + (x + a0 * (v - v_rest) * (v - v_c)) / tau
    spike = surrogate_function(v_charged - v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


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

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`eif_multi_step_cupy`.
    """
    v_charged = (
        v + (x + v_rest - v + delta_t * torch.exp((v - theta_rh) / delta_t)) / tau
    )
    spike = surrogate_function(v_charged - v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


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

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`activation_aware_if_multi_step_triton`.
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

    .. seealso::

       单步形式 / Single-step form: :func:`activation_aware_if_step`.
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


def lif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_reset: Optional[float],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <lif_charge-cn>` | :ref:`English <lif_charge-en>`

    ----

    .. _lif_charge-cn:

    * **中文**

    执行 LIF 神经元的充电方程，不进行放电或重置。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_reset: 重置电位；``None`` 在充电方程中按 ``0.0`` 处理
    :type v_reset: Optional[float]
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _lif_charge-en:

    * **English**

    Apply the LIF charging equation without firing or resetting.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param tau: Membrane-voltage time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_reset: Reset voltage; ``None`` is treated as ``0.0`` by the charging equation
    :type v_reset: Optional[float]
    :return: Charged membrane voltage
    :rtype: torch.Tensor
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        return v + (x - (v - v_reset_value)) / tau
    return v - (v - v_reset_value) / tau + x


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

    .. seealso::

       独立多步形式 / Independent multi-step forms:
       :func:`lif_multi_step_cupy`, :func:`lif_multi_step_triton`.
    """
    v_charged = lif_charge(x, v, tau, decay_input, v_reset)
    spike = surrogate_function(v_charged - v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def liaf_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    act: Callable[[torch.Tensor], torch.Tensor],
    threshold_related: bool,
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <liaf_step-cn>` | :ref:`English <liaf_step-en>`

    ----

    .. _liaf_step-cn:

    * **中文**

    执行一次 LIAF 状态转移，返回模拟输出和重置后的膜电位。函数不读取或修改
    module memory。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位，shape、dtype 和 device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param act: 生成模拟输出的激活函数
    :type act: Callable[[torch.Tensor], torch.Tensor]
    :param threshold_related: 为 ``True`` 时将 ``act`` 作用于充电电位减阈值，
        否则直接作用于充电电位
    :type threshold_related: bool
    :param surrogate_function: 计算 reset 所需脉冲的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否在 reset 分支中分离脉冲的计算图
    :type detach_reset: bool
    :return: ``(analog_output, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _liaf_step-en:

    * **English**

    Run one LIAF state transition and return the analog output and reset membrane
    voltage. The function does not read or mutate module memory.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with shape, dtype, and device
        compatible with ``x``
    :type v: torch.Tensor
    :param tau: Membrane-voltage time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` selects soft reset
    :type v_reset: Optional[float]
    :param act: Activation function that produces the analog output
    :type act: Callable[[torch.Tensor], torch.Tensor]
    :param threshold_related: Apply ``act`` to charged voltage minus the threshold
        when ``True``; otherwise apply it directly to charged voltage
    :type threshold_related: bool
    :param surrogate_function: Surrogate function used to obtain the spike for reset
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach the spike in the reset branch
    :type detach_reset: bool
    :return: ``(analog_output, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    v_charged = lif_charge(x, v, tau, decay_input, v_reset)
    output = act(v_charged - v_threshold if threshold_related else v_charged)
    spike = surrogate_function(v_charged - v_threshold)
    v_next = voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)
    return output, v_next


def ilif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <ilif_step-cn>` | :ref:`English <ilif_step-en>`

    ----

    .. _ilif_step-cn:

    * **中文**

    执行一次 I-LIF 状态转移，返回多级脉冲计数和 soft reset 后的膜电位。
    函数不读取或修改 module memory。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 已物化的当前膜电位，shape、dtype 和 device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param surrogate_function: 作用于归一化充电电位的多级脉冲计数函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否在 reset 分支中分离脉冲计数的计算图
    :type detach_reset: bool
    :return: ``(spike_count, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _ilif_step-en:

    * **English**

    Run one I-LIF state transition and return the multi-level spike count and
    membrane voltage after soft reset. The function does not read or mutate
    module memory.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with shape, dtype, and device
        compatible with ``x``
    :type v: torch.Tensor
    :param tau: Membrane-voltage time constant
    :type tau: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param surrogate_function: Multi-level spike-count function applied to the
        normalized charged voltage
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach the spike count in the reset branch
    :type detach_reset: bool
    :return: ``(spike_count, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    v_charged = lif_charge(x, v, tau, False, None)
    spike = surrogate_function(v_charged / v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, None, detach_reset)


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

    .. seealso::

       独立多步形式 / Independent multi-step forms:
       :func:`plif_multi_step_cupy`, :func:`plif_multi_step_triton`.
    """
    reciprocal_tau = w.sigmoid()
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        v_charged = v + (x - (v - v_reset_value)) * reciprocal_tau
    else:
        v_charged = v - (v - v_reset_value) * reciprocal_tau + x
    spike = surrogate_function(v_charged - v_threshold)
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


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

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`izhikevich_multi_step_cupy`.
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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        v_charged = v + (x - (v - v_reset_value)) / tau
    else:
        v_charged = v - (v - v_reset_value) / tau + x
    v_charged = torch.relu(k * v_charged)
    spike = surrogate_function(v_charged - v_threshold)
    if scale_reset:
        return spike, voltage_reset(
            v_charged / k,
            spike,
            v_threshold / k,
            v_reset,
            detach_reset,
        )
    return spike, voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)


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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    current_next = current * current_decay + x
    v_charged = v * voltage_decay + current_next
    spike = surrogate_function(v_charged - v_threshold)
    v_next = voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)
    return spike, current_next, v_next


def if_step_cupy(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <if_step_cupy-cn>` | :ref:`English <if_step_cupy-en>`

    ----

    .. _if_step_cupy-cn:

    * **中文**

    使用 CuPy kernel 执行 IF 单步状态转移。输入必须是 CUDA 上的 ``float32``
    或 ``float16`` 张量，替代梯度函数必须提供可调用的 ``cuda_codes`` 属性。
    函数自行选择并缓存 kernel，并在调用底层 kernel 时展平和恢复 shape。

    :param x: 当前 CUDA 输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 已物化且与 ``x`` shape 相同的膜电位
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next)``，两者 shape 均与 ``x`` 相同
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises TypeError: ``surrogate_function.cuda_codes`` 不可调用
    :raises NotImplementedError: 输入 dtype 不受支持

    ----

    .. _if_step_cupy-en:

    * **English**

    Run one IF state transition with CuPy kernels. Inputs must be CUDA
    ``float32`` or ``float16`` tensors, and the surrogate function must expose
    callable ``cuda_codes``. The function selects and caches kernels and handles
    flattening and shape restoration around the low-level kernel.

    :param x: Current CUDA input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with the same shape as ``x``
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :return: ``(spike, v_next)``, both shaped like ``x``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises TypeError: ``surrogate_function.cuda_codes`` is not callable
    :raises NotImplementedError: The input dtype is unsupported

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`if_multi_step_cupy`.
    """
    from ..cuda_kernel.neuron_kernel.single_step.integrate_and_fire import if_step

    spike, v_next = if_step(
        x.flatten(),
        v.flatten(),
        v_threshold,
        v_reset,
        surrogate_function,
        detach_reset,
    )
    return spike.reshape_as(x), v_next.reshape_as(v)


def lif_step_cupy(
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
    **API Language** - :ref:`中文 <lif_step_cupy-cn>` | :ref:`English <lif_step_cupy-en>`

    ----

    .. _lif_step_cupy-cn:

    * **中文**

    使用 CuPy kernel 执行 LIF 单步状态转移。输入必须是 CUDA 上的 ``float32``
    或 ``float16`` 张量，替代梯度函数必须提供可调用的 ``cuda_codes`` 属性。
    函数自行选择并缓存 kernel，并在调用底层 kernel 时展平和恢复 shape。

    :param x: 当前 CUDA 输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param v: 已物化且与 ``x`` shape 相同的膜电位
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
    :return: ``(spike, v_next)``，两者 shape 均与 ``x`` 相同
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises TypeError: ``surrogate_function.cuda_codes`` 不可调用
    :raises NotImplementedError: 输入 dtype 不受支持

    ----

    .. _lif_step_cupy-en:

    * **English**

    Run one LIF state transition with CuPy kernels. Inputs must be CUDA
    ``float32`` or ``float16`` tensors, and the surrogate function must expose
    callable ``cuda_codes``. The function selects and caches kernels and handles
    flattening and shape restoration around the low-level kernel.

    :param x: Current CUDA input tensor shaped ``[N, *]``
    :type x: torch.Tensor
    :param v: Materialized membrane voltage with the same shape as ``x``
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
    :return: ``(spike, v_next)``, both shaped like ``x``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises TypeError: ``surrogate_function.cuda_codes`` is not callable
    :raises NotImplementedError: The input dtype is unsupported

    .. seealso::

       独立多步形式 / Independent multi-step form:
       :func:`lif_multi_step_cupy`.
    """
    from ..cuda_kernel.neuron_kernel.single_step.lif import lif_step

    spike, v_next = lif_step(
        x.flatten(),
        v.flatten(),
        v_threshold,
        v_reset,
        1.0 / tau,
        decay_input,
        surrogate_function,
        detach_reset,
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

    使用 CuPy kernel 执行 IF 多步状态转移，不进行通用 backend 分发。输入必须是
    CUDA 上的 ``float32`` 或 ``float16`` 张量，并且替代梯度函数必须提供可调用的
    ``cuda_codes`` 属性。运行时需要安装 CuPy。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 与 ``x_seq`` device、dtype 相同的初始膜电位，shape 为
        ``x_seq.shape[1:]``
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
    :raises TypeError: ``surrogate_function.cuda_codes`` 不可调用
    :raises NotImplementedError: 输入 dtype 不受支持

    ----

    .. _if_multi_step_cupy-en:

    * **English**

    Run an IF multi-step state transition with the CuPy kernel without generic
    backend dispatch. Inputs must be CUDA ``float32`` or ``float16`` tensors,
    and the surrogate function must expose callable ``cuda_codes``. CuPy is
    required at runtime.

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
    :raises TypeError: ``surrogate_function.cuda_codes`` is not callable
    :raises NotImplementedError: The input dtype is unsupported

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`if_step`.
       CuPy 单步形式 / CuPy single-step form: :func:`if_step_cupy`.
    """
    from ..cuda_kernel.neuron_kernel.multi_step.integrate_and_fire import (
        if_multi_step,
    )

    spike_seq, v_seq = if_multi_step(
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

    使用 CuPy kernel 执行 LIF 多步状态转移，不进行通用 backend 分发。输入必须是
    CUDA 上的 ``float32`` 或 ``float16`` 张量，并且替代梯度函数必须提供可调用的
    ``cuda_codes`` 属性。运行时需要安装 CuPy。

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
    :raises TypeError: ``surrogate_function.cuda_codes`` 不可调用
    :raises NotImplementedError: 输入 dtype 不受支持

    ----

    .. _lif_multi_step_cupy-en:

    * **English**

    Run a LIF multi-step state transition with the CuPy kernel without generic
    backend dispatch. Inputs must be CUDA ``float32`` or ``float16`` tensors,
    and the surrogate function must expose callable ``cuda_codes``. CuPy is
    required at runtime.

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
    :raises TypeError: ``surrogate_function.cuda_codes`` is not callable
    :raises NotImplementedError: The input dtype is unsupported

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`lif_step`.
       CuPy 单步形式 / CuPy single-step form: :func:`lif_step_cupy`.
    """
    from ..cuda_kernel.neuron_kernel.multi_step.lif import lif_multi_step

    spike_seq, v_seq = lif_multi_step(
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

    使用 CuPy kernel 执行 PLIF 多步状态转移。``w`` 与 torch 接口语义一致。
    输入必须是 CUDA 上的 ``float32`` 或 ``float16`` 张量，并且替代梯度函数必须
    提供可调用的 ``cuda_codes`` 属性。运行时需要安装 CuPy。

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
    :raises TypeError: ``surrogate_function.cuda_codes`` 不可调用
    :raises NotImplementedError: 输入 dtype 不受支持

    ----

    .. _plif_multi_step_cupy-en:

    * **English**

    Run a PLIF multi-step state transition with the CuPy kernel. ``w`` has the
    same semantics as in the torch interface. Inputs must be CUDA
    ``float32`` or ``float16`` tensors, and the surrogate function must expose
    callable ``cuda_codes``. CuPy is required at runtime.

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
    :raises TypeError: ``surrogate_function.cuda_codes`` is not callable
    :raises NotImplementedError: The input dtype is unsupported

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`plif_step`.
    """
    from ..cuda_kernel.neuron_kernel.multi_step.plif import plif_multi_step

    spike_seq, v_seq = plif_multi_step(
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

    .. seealso::

       单步形式 / Single-step form: :func:`qif_step`.
    """
    from ..cuda_kernel.neuron_kernel.multi_step.qif import qif_multi_step

    spike_seq, v_seq = qif_multi_step(
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

    .. seealso::

       单步形式 / Single-step form: :func:`eif_step`.
    """
    from ..cuda_kernel.neuron_kernel.multi_step.eif import eif_multi_step

    spike_seq, v_seq = eif_multi_step(
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

    :param x_seq: ``float32`` CUDA 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 与 ``x_seq`` 位于同一 CUDA device 的 ``float32`` 初始膜电位，
        shape 为 ``x_seq.shape[1:]``
    :type v: torch.Tensor
    :param w: 与 ``x_seq`` 位于同一 CUDA device 的 ``float32`` 初始适应变量，
        shape 与 ``v`` 相同
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
    :param b: 脉冲触发的适应变量增量
    :type b: float
    :param tau_w: 适应变量时间常数
    :type tau_w: float
    :param v_c: 临界电位
    :type v_c: float
    :param a0: 膜电位动力学的二次项系数
    :type a0: float
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param surrogate_function: CuPy kernel 使用的替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_state_seq: 是否返回完整 ``v``、``w`` 序列
    :type store_state_seq: bool
    :return: ``(spike_seq, v_next, w_next, v_seq_or_none, w_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    :raises NotImplementedError: 当输入 dtype 不是 ``float32`` 时抛出
    :raises ValueError: 当输入 device 或 shape 不一致时抛出

    ----

    .. _izhikevich_multi_step_cupy-en:

    * **English**

    Run an Izhikevich multi-step state transition with the already selected
    CuPy kernel, explicitly receiving and returning membrane voltage ``v`` and
    adaptation state ``w``.

    :param x_seq: ``float32`` CUDA input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: ``float32`` initial membrane voltage on the same CUDA device as
        ``x_seq``, shaped as ``x_seq.shape[1:]``
    :type v: torch.Tensor
    :param w: ``float32`` initial adaptation state on the same CUDA device as
        ``x_seq`` and with the same shape as ``v``
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
    :param b: Spike-triggered adaptation increment
    :type b: float
    :param tau_w: Adaptation time constant
    :type tau_w: float
    :param v_c: Critical voltage
    :type v_c: float
    :param a0: Quadratic coefficient in the membrane-voltage dynamics
    :type a0: float
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param surrogate_function: Surrogate-gradient function used by the CuPy kernel
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param store_state_seq: Whether to return the full ``v`` and ``w`` sequences
    :type store_state_seq: bool
    :return: ``(spike_seq, v_next, w_next, v_seq_or_none, w_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    :raises NotImplementedError: If the input dtype is not ``float32``
    :raises ValueError: If the input devices or shapes are inconsistent

    .. seealso::

       单步形式 / Single-step form: :func:`izhikevich_step`.
    """
    for name, tensor in (("x_seq", x_seq), ("v", v), ("w", w)):
        if tensor.dtype != torch.float32:
            raise NotImplementedError(
                "izhikevich_multi_step_cupy requires float32 inputs, but "
                f"{name} has dtype {tensor.dtype}."
            )
    if not x_seq.is_cuda:
        raise ValueError("x_seq, v, and w must be CUDA tensors.")
    if v.device != x_seq.device or w.device != x_seq.device:
        raise ValueError("x_seq, v, and w must be on the same CUDA device.")
    if x_seq.ndim < 2:
        raise ValueError("x_seq must have at least 2 dimensions shaped [T, N, *].")
    if v.shape != x_seq.shape[1:]:
        raise ValueError(
            "v must have shape x_seq.shape[1:] "
            f"(={tuple(x_seq.shape[1:])}), got {tuple(v.shape)}."
        )
    if w.shape != v.shape:
        raise ValueError(
            f"w must have the same shape as v (={tuple(v.shape)}), "
            f"got {tuple(w.shape)}."
        )

    from ..cuda_kernel.neuron_kernel.multi_step.izhikevich import (
        izhikevich_multi_step,
    )

    spike_seq, v_seq, w_seq = izhikevich_multi_step(
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

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`if_step`.
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
        return spike_seq, voltage[-1].clone(), voltage
    return spike_seq, voltage, None


def _if_multi_step_triton_mp(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool,
    store_v_seq: bool,
    precision: tuple[torch.dtype, str, str],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    from ..triton_kernel.neuron_kernel.integrate_and_fire import _multistep_if_mp

    spike_seq, voltage, _ = _multistep_if_mp(
        x_seq,
        v,
        v_threshold=v_threshold,
        v_reset=v_reset,
        storage_dtype=precision[0],
        compute_dtype=precision[1],
        backward_compute_dtype=precision[2],
        spike_dtype=x_seq.dtype,
        save_intermediates=store_v_seq,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )
    return spike_seq, voltage[-1].clone(), voltage if store_v_seq else None


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

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`lif_step`.
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
        return spike_seq, voltage[-1].clone(), voltage
    return spike_seq, voltage, None


def _lif_multi_step_triton_mp(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool,
    store_v_seq: bool,
    precision: tuple[torch.dtype, str, str],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    from ..triton_kernel.neuron_kernel.lif import _multistep_lif_mp

    spike_seq, voltage, _ = _multistep_lif_mp(
        x_seq,
        v,
        decay_input=decay_input,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        storage_dtype=precision[0],
        compute_dtype=precision[1],
        backward_compute_dtype=precision[2],
        spike_dtype=x_seq.dtype,
        save_intermediates=store_v_seq,
        store_v_seq=store_v_seq,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )
    v = voltage[-1].clone() if store_v_seq else voltage.clone()
    return spike_seq, v, voltage if store_v_seq else None


def ilif_multi_step_triton(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <ilif_multi_step_triton-cn>` | :ref:`English <ilif_multi_step_triton-en>`

    ----

    .. _ilif_multi_step_triton-cn:

    * **中文**

    使用 Triton kernel 执行 I-LIF 多步状态转移，不进行 backend 分发。
    ``v`` 是已物化的初始膜电位；``store_v_seq`` 为 ``True`` 时返回完整的
    reset 后膜电位序列。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``。
    :type x_seq: torch.Tensor
    :param v: 初始膜电位，shape 为 ``x_seq.shape[1:]``。
    :type v: torch.Tensor
    :param tau: 膜电位时间常数。
    :type tau: float
    :param v_threshold: 发放阈值。
    :type v_threshold: float
    :param surrogate_function: 多级整数发放函数。
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的发放计数。
    :type detach_reset: bool
    :param store_v_seq: 是否返回膜电位序列。
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``。
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ImportError: Triton kernel 不可用。

    ----

    .. _ilif_multi_step_triton-en:

    * **English**

    Run the I-LIF multi-step state transition with the Triton kernel without
    backend dispatch. ``v`` is the materialized initial membrane voltage;
    ``store_v_seq=True`` also returns the post-reset voltage sequence.

    :param x_seq: Input sequence shaped ``[T, N, *]``.
    :type x_seq: torch.Tensor
    :param v: Initial membrane voltage shaped ``x_seq.shape[1:]``.
    :type v: torch.Tensor
    :param tau: Membrane-voltage time constant.
    :type tau: float
    :param v_threshold: Firing threshold.
    :type v_threshold: float
    :param surrogate_function: Multi-level integer firing function.
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach the firing count in reset.
    :type detach_reset: bool
    :param store_v_seq: Whether to return the membrane-voltage sequence.
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``.
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ImportError: If the Triton kernel is unavailable.
    """
    from ..triton_kernel.neuron_kernel import ilif

    if ilif is None:
        raise ImportError("ilif_multi_step_triton requires the Triton backend.")
    spike_seq, voltage = ilif._multistep_ilif(
        x_seq,
        v,
        1.0 - 1.0 / tau,
        v_threshold,
        surrogate_function.max_spike_count,
        surrogate_function.grad_min,
        surrogate_function.grad_max,
        detach_reset,
        store_v_seq,
    )
    if store_v_seq:
        return spike_seq, voltage[-1].clone(), voltage
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

    使用 Triton kernel 执行 PLIF 多步状态转移。``w`` 与 torch 接口语义一致。

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
    same semantics as in the torch interface.

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

    .. seealso::

       单步状态方程 / Single-step state equation: :func:`plif_step`.
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


def _plif_multi_step_triton_mp(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool,
    store_v_seq: bool,
    precision: tuple[torch.dtype, str, str],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    from ..triton_kernel.neuron_kernel.plif import _multistep_plif_mp

    spike_seq, voltage, _ = _multistep_plif_mp(
        x_seq,
        v,
        w.sigmoid().to(x_seq),
        decay_input=decay_input,
        v_threshold=v_threshold,
        v_reset=v_reset,
        storage_dtype=precision[0],
        compute_dtype=precision[1],
        backward_compute_dtype=precision[2],
        spike_dtype=x_seq.dtype,
        save_intermediates=store_v_seq,
        detach_reset=detach_reset,
        surrogate_function=surrogate_function,
    )
    return spike_seq, voltage[-1].clone(), voltage if store_v_seq else None


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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
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


def masked_psn_step(
    x: torch.Tensor,
    time_step: int,
    queue: tuple[torch.Tensor, ...],
    masked_weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, int, tuple[torch.Tensor, ...]]:
    r"""
    **API Language** - :ref:`中文 <masked_psn_step-cn>` | :ref:`English <masked_psn_step-en>`

    ----

    .. _masked_psn_step-cn:

    * **中文**

    使用显式时间索引和输入队列执行一次 MaskedPSN 状态转移。调用者负责提供当前
    ``lambda`` 对应的完整 masked weight，并保证时间索引位于其范围内。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param time_step: 当前时间索引
    :type time_step: int
    :param queue: 旧输入队列，元素是 flatten 后的张量，按旧到新排列
    :type queue: Tuple[torch.Tensor, ...]
    :param masked_weight: shape 为 ``[T, T]`` 的 masked weight
    :type masked_weight: torch.Tensor
    :param bias: shape 为 ``[T, 1]`` 的偏置
    :type bias: torch.Tensor
    :param k: 队列保留的最大时间步数
    :type k: int
    :param surrogate_function: 作用于膜电位的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, time_step_next, queue_next)``
    :rtype: Tuple[torch.Tensor, int, Tuple[torch.Tensor, ...]]

    ----

    .. _masked_psn_step-en:

    * **English**

    Run one MaskedPSN state transition with an explicit time index and input
    queue. The caller provides the complete masked weight for the current
    ``lambda`` and ensures that the time index is in range.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param time_step: Current time index
    :type time_step: int
    :param queue: Previous input queue containing flattened tensors ordered from
        oldest to newest
    :type queue: Tuple[torch.Tensor, ...]
    :param masked_weight: Masked weight shaped ``[T, T]``
    :type masked_weight: torch.Tensor
    :param bias: Bias shaped ``[T, 1]``
    :type bias: torch.Tensor
    :param k: Maximum number of time steps retained in the queue
    :type k: int
    :param surrogate_function: Surrogate function applied to membrane voltage
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, time_step_next, queue_next)``
    :rtype: Tuple[torch.Tensor, int, Tuple[torch.Tensor, ...]]

    .. note::

       MaskedPSN 已有独立的矩阵化多步实现，本函数只描述其单步递推。
       MaskedPSN has an independent matrix-based multi-step implementation; this
       function describes only its single-step recurrence.
    """
    queue_next = (*queue, x.flatten())
    if len(queue_next) > k:
        queue_next = queue_next[1:]

    weight = masked_weight[
        time_step,
        time_step + 1 - len(queue_next) : time_step + 1,
    ]
    h = torch.sum(weight.unsqueeze(-1) * torch.stack(queue_next), 0)
    spike = surrogate_function(h + bias[time_step])
    return spike.view(x.shape), time_step + 1, queue_next


def gated_lif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    spike: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    tau: torch.Tensor,
    v_threshold: torch.Tensor,
    linear_decay: torch.Tensor,
    v_subreset: torch.Tensor,
    conduct: torch.Tensor,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <gated_lif_step-cn>` | :ref:`English <gated_lif_step-en>`

    ----

    .. _gated_lif_step-cn:

    * **中文**

    执行 ``GatedLIFNode`` 的单步显式状态转移。函数接收当前输入、膜电位和上一
    时间步脉冲，以及已完成 sigmoid 和 shape 变换的参数，返回当前脉冲和下一膜
    电位。``u`` 与 ``v`` 在该步结束时相同，因此只返回一份 next state。

    函数不读取或写入 ``MemoryModule`` memory，不负责 ``training/eval``、
    ``step_mode`` 或 backend dispatch。调用者必须传入当前 module 已选定执行路径的
    参数和替代函数。

    本函数没有独立的多步形式；多步执行由调用者循环调用本函数。

    :param x: 当前输入，现有 ``GatedLIFNode`` 约定 shape 为 ``[N, C, H, W]``
    :type x: torch.Tensor
    :param v: 当前膜电位，shape 可广播到 ``x``
    :type v: torch.Tensor
    :param spike: 上一时间步脉冲，shape 与 ``x`` 相同
    :type spike: torch.Tensor
    :param alpha: sigmoid 后的门控参数 ``alpha``
    :type alpha: torch.Tensor
    :param beta: sigmoid 后的门控参数 ``beta``
    :type beta: torch.Tensor
    :param gamma: sigmoid 后的门控参数 ``gamma``
    :type gamma: torch.Tensor
    :param tau: sigmoid 后的膜电位衰减参数
    :type tau: torch.Tensor
    :param v_threshold: sigmoid 后的阈值参数
    :type v_threshold: torch.Tensor
    :param linear_decay: sigmoid 后的线性衰减参数
    :type linear_decay: torch.Tensor
    :param v_subreset: sigmoid 后的 soft-reset 参数
    :type v_subreset: torch.Tensor
    :param conduct: 当前时间步 sigmoid 后的电导参数
    :type conduct: torch.Tensor
    :param surrogate_function: 作用于 ``u - v_threshold`` 的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike_next, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _gated_lif_step-en:

    * **English**

    Run one explicit state transition for ``GatedLIFNode``. The function receives
    the current input, membrane voltage, previous spike, and parameters whose
    sigmoid and shape transforms have already been applied. It returns the
    current spike and next membrane voltage. ``u`` and ``v`` are equal at the end
    of the step, so the next state is returned once.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``training/eval``, ``step_mode``, or backend dispatch. The caller must
    pass the parameters and surrogate function for the execution path already
    selected by the owning module.

    This function has no independent multi-step form; callers implement
    multi-step execution by looping over this function.

    :param x: Current input. Existing ``GatedLIFNode`` expects shape
        ``[N, C, H, W]``
    :type x: torch.Tensor
    :param v: Current membrane voltage, broadcastable to ``x``
    :type v: torch.Tensor
    :param spike: Previous time-step spike with the same shape as ``x``
    :type spike: torch.Tensor
    :param alpha: Sigmoid-transformed ``alpha`` gate parameter
    :type alpha: torch.Tensor
    :param beta: Sigmoid-transformed ``beta`` gate parameter
    :type beta: torch.Tensor
    :param gamma: Sigmoid-transformed ``gamma`` gate parameter
    :type gamma: torch.Tensor
    :param tau: Sigmoid-transformed membrane-decay parameter
    :type tau: torch.Tensor
    :param v_threshold: Sigmoid-transformed threshold parameter
    :type v_threshold: torch.Tensor
    :param linear_decay: Sigmoid-transformed linear-decay parameter
    :type linear_decay: torch.Tensor
    :param v_subreset: Sigmoid-transformed soft-reset parameter
    :type v_subreset: torch.Tensor
    :param conduct: Sigmoid-transformed conductance for the current time step
    :type conduct: torch.Tensor
    :param surrogate_function: Surrogate function applied to
        ``u - v_threshold``
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike_next, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    input_current = x * (1 - beta * (1 - conduct))
    v_next = ((1 - alpha * (1 - tau)) * v - (1 - alpha) * linear_decay) + input_current
    v_next = (
        v_next
        - (1 - alpha * (1 - tau)) * v * gamma * spike
        - (1 - gamma) * v_subreset * spike
    )
    spike_next = surrogate_function(v_next - v_threshold)
    return spike_next, v_next


def stbif_step(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stbif_step-cn>` | :ref:`English <stbif_step-en>`

    ----

    .. _stbif_step-cn:

    * **中文**

    执行 ``STBIFNeuron`` 的单步显式状态转移。函数接收当前输入 ``x``、已物化的
    量化残差 ``q`` 和累计释放量 ``acc_q``，以及量化尺度与边界 tensor，返回
    ``(out, q_next, acc_q_next, cur_output_next)``。量化尺度和边界会在函数入口
    转换到 ``x`` 的 device/dtype。

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
    :param q_threshold: 可广播到 ``x`` 的量化 scale tensor
    :type q_threshold: torch.Tensor
    :param pos_max: 可广播到 ``x`` 的正向累计量化上界 tensor
    :type pos_max: torch.Tensor
    :param neg_min: 可广播到 ``x`` 的负向累计量化下界 tensor
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _stbif_step-en:

    * **English**

    Run one explicit state transition for ``STBIFNeuron``. The function receives
    current input ``x``, materialized quantized residual ``q`` and accumulated
    released quantity ``acc_q``, plus ``q_threshold``, ``pos_max``, and
    ``neg_min`` tensors. It returns
    ``(out, q_next, acc_q_next, cur_output_next)``. The scale and bounds are
    converted to the device and dtype of ``x`` at function entry.

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
    :param q_threshold: Quantization-scale tensor broadcastable to ``x``
    :type q_threshold: torch.Tensor
    :param pos_max: Positive accumulated-quantization-bound tensor broadcastable
        to ``x``
    :type pos_max: torch.Tensor
    :param neg_min: Negative accumulated-quantization-bound tensor broadcastable
        to ``x``
    :type neg_min: torch.Tensor
    :return: ``(out, q_next, acc_q_next, cur_output_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    .. note::

       本函数没有独立 Torch 多步形式；Torch 多步执行由调用者逐步循环。
       This function has no independent Torch multi-step form; callers iterate it
       for Torch sequence execution.
    """
    q_threshold = q_threshold.to(x)
    pos_max = pos_max.to(x)
    neg_min = neg_min.to(x)
    normalized = x / q_threshold
    q_next = q + normalized.detach()
    acc_q_next = torch.round(acc_q)
    spike_position = (q_next - 1 >= 0) & (acc_q_next < pos_max)
    neg_spike_position = (q_next < 0) & (acc_q_next > neg_min)
    cur_output_next = spike_position.to(x.dtype) - neg_spike_position.to(x.dtype)
    acc_q_next = acc_q_next + cur_output_next
    q_next = torch.where(spike_position, q_next - 1, q_next)
    q_next = torch.where(neg_spike_position, q_next + 1, q_next)
    out = cur_output_next * q_threshold
    return out, q_next, acc_q_next, cur_output_next


def stbif_single_step_triton(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stbif_single_step_triton-cn>` | :ref:`English <stbif_single_step_triton-en>`

    ----

    .. _stbif_single_step_triton-cn:

    * **中文**

    使用专用 Triton kernel 执行 SpikeZIP STBIF 单步状态转移。``x``、``q``
    和 ``acc_q`` 必须是相同 shape、dtype 和 device 的 CUDA FP32、FP16 或
    BF16 张量；该离散推理接口不支持 autograd。

    :param x: CUDA 单步输入
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
    :raises ValueError: 输入 shape、dtype、device 或标量参数不满足约束
    :raises NotImplementedError: dtype 不受 Triton 后端支持

    ----

    .. _stbif_single_step_triton-en:

    * **English**

    Run one SpikeZIP STBIF state transition with the dedicated Triton kernel.
    ``x``, ``q``, and ``acc_q`` must be CUDA FP32, FP16, or BF16 tensors with
    matching shape, dtype, and device. This discrete inference interface does
    not support autograd.

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
    :raises ValueError: If the input shape, dtype, device, or scalar
        parameter is invalid
    :raises NotImplementedError: If the dtype is not supported by the Triton
        backend
    """
    from ..triton_kernel.neuron_kernel import stbif

    return stbif.single_step_stbif(
        x,
        q,
        acc_q,
        q_threshold,
        pos_max,
        neg_min,
    )
