r"""显式的神经元状态转移函数 / Explicit neuron state-transition functions.

单步函数显式接收当前输入和状态，不读取或写入 ``MemoryModule`` 的 memory。
其共同契约是：

.. math::

   (Y_t, V_t) = \Phi(X_t, V_{t-1}; \theta)

其中 :math:`V_t` 可以是一个或多个状态张量；具体神经元自行定义
:math:`\Phi`，不要求拆分为固定的充电、发放和复位步骤。状态生命周期由拥有它的
neuron module 管理。

多步函数接收时间优先的 ``[T, ...]`` 序列，返回输出序列和最终状态。支持
``store_v_seq=True`` 的神经元还会返回可选的状态轨迹 ``v_seq``；它是调试或监控用的
派生结果，不是需要传给下一次调用的持久化状态。算子根据输入张量的设备自动选择实现。

Single-step functions explicitly receive the current input and state. They do not read
or write ``MemoryModule`` memory. Their common contract is:

.. math::

   (Y_t, V_t) = \Phi(X_t, V_{t-1}; \theta)

``V_t`` may contain one or more state tensors. Each neuron defines ``Phi`` as needed;
the transition is not required to split into fixed charging, firing, and reset steps.
The owning neuron module manages state lifetime.

Multi-step functions consume time-major ``[T, ...]`` sequences and return the output
sequence and final state. Neurons that support ``store_v_seq=True`` additionally return
the optional state trace ``v_seq``. This trace is derived monitoring data, not persistent
state for the next call. Operators select implementations from the input tensor device.
"""

from __future__ import annotations

import math
import importlib
from typing import Callable, Optional

import torch

from ..._ops.surrogate import _DTYPES, _surrogate_spec

__all__ = [
    "lif_multi_step",
    "if_multi_step",
    "plif_multi_step",
    "qif_multi_step",
    "eif_multi_step",
    "izhikevich_multi_step",
    "ilif_multi_step",
    "activation_aware_if_multi_step",
    "stbif_multi_step",
    "neuron_implementation",
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
    "raf_step",
    "klif_step",
    "cuba_lif_step",
    "clif_step",
    "sliding_psn_step",
    "masked_psn_step",
    "gated_lif_step",
    "stbif_step",
    "activation_aware_if_step",
    "voltage_reset",
]


SurrogateFunction = Callable[[torch.Tensor], torch.Tensor]


def neuron_implementation(neuron_type: str, device: torch.device) -> dict[str, object]:
    r"""
    **API Language** - :ref:`中文 <neuron_implementation-cn>` | :ref:`English <neuron_implementation-en>`

    ----

    .. _neuron_implementation-cn:

    * **中文**

    查询神经元在指定设备上的当前实现。CPU 返回 Torch 参考实现；CUDA 查询会
    初始化并缓存该设备对应的算子选择，不执行神经元计算，也不改变模块状态。

    :param neuron_type: ``if``、``lif``、``plif``、``qif``、``eif``、``izhikevich``、``ilif``、``activation_aware_if`` 或 ``stbif``。
    :type neuron_type: str
    :param device: CPU 或 NVIDIA CUDA 设备；省略 CUDA 索引时使用当前设备。
    :type device: torch.device
    :return: 包含实际 ``implementation`` 名称和各不可用候选 ``unavailable`` 原因的字典副本。
    :rtype: dict[str, object]
    :raises ValueError: 神经元名称或设备类型无效。
    :raises RuntimeError: CUDA 设备没有可用实现。

    ----

    .. _neuron_implementation-en:

    * **English**

    Query the selected implementation for a neuron on a device. CPU returns the
    Torch reference. A CUDA query initializes and caches that device's operator
    selection; it does not run neuron computation or mutate module state.

    :param neuron_type: One of ``if``, ``lif``, ``plif``, ``qif``, ``eif``, ``izhikevich``, ``ilif``, ``activation_aware_if``, or ``stbif``.
    :type neuron_type: str
    :param device: CPU or NVIDIA CUDA device; an omitted CUDA index uses the current device.
    :type device: torch.device
    :return: A copy of ``implementation`` and the ``unavailable`` reasons for rejected candidates.
    :rtype: dict[str, object]
    :raises ValueError: Invalid neuron name or device type.
    :raises RuntimeError: No CUDA implementation is available.
    """
    packages = {
        "if": "if_",
        "lif": "lif",
        "plif": "plif",
        "qif": "qif",
        "eif": "eif",
        "izhikevich": "izhikevich",
        "ilif": "ilif",
        "activation_aware_if": "activation_aware_if",
        "stbif": "stbif",
    }
    if neuron_type not in packages:
        raise ValueError(f"Unknown neuron type {neuron_type!r}.")
    if device.type == "cpu":
        return {"implementation": "torch-reference", "unavailable": {}}
    if device.type != "cuda":
        raise ValueError(f"Unsupported neuron device {device}.")
    package = importlib.import_module(f"..._ops.{packages[neuron_type]}", __package__)
    return package._selection.diagnostics(device)


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
    from ..._ops.reset import voltage

    return voltage(v, spike, v_threshold, v_reset, detach_reset)


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

       :func:`if_multi_step`.
    """
    scalar = x.ndim == 0
    spikes, voltage, _ = if_multi_step(
        x.reshape(1, 1) if scalar else x.unsqueeze(0),
        v.reshape(1) if scalar else v,
        v_threshold,
        v_reset,
        surrogate_function,
        detach_reset,
    )
    return (
        (spikes[0, 0], voltage.reshape_as(v).clone())
        if scalar
        else (spikes[0], voltage)
    )


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
    spikes, voltage, _ = qif_multi_step(
        x.unsqueeze(0),
        v,
        tau,
        v_threshold,
        v_reset,
        v_rest,
        v_c,
        a0,
        detach_reset,
        surrogate_function,
    )
    return spikes[0], voltage


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
    spikes, voltage, _ = eif_multi_step(
        x.unsqueeze(0),
        v,
        tau,
        v_threshold,
        v_reset,
        v_rest,
        theta_rh,
        delta_t,
        detach_reset,
        surrogate_function,
    )
    return spikes[0], voltage


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
    ``step_mode`` 或 channel-wise 参数广播。module 必须先完成
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
    manage ``training/eval``, ``step_mode``, or channel-wise
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
    from ..._ops.activation_aware_if.reference import step

    return step(x, v, v_threshold, v_offset, v_reset, surrogate_function, detach_reset)


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
    from ..._ops.lif.reference import charge

    return charge(x, v, tau, decay_input, v_reset)


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

    执行一个 LIF 单步状态转移，返回 ``(spike, v_next)``。该函数不读写模块
    状态，surrogate 梯度是否记录由 autograd 状态决定。

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

    Run one LIF single-step state transition and return ``(spike, v_next)``.
    The function does not read or write module state; autograd context determines
    whether surrogate gradients are recorded.

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

       :func:`lif_multi_step`.
    """
    scalar = x.ndim == 0
    spikes, voltage, _ = lif_multi_step(
        x.reshape(1, 1) if scalar else x.unsqueeze(0),
        v.reshape(1) if scalar else v,
        tau,
        decay_input,
        v_threshold,
        v_reset,
        surrogate_function,
        detach_reset,
    )
    return (
        (spikes[0, 0], voltage.reshape_as(v).clone())
        if scalar
        else (spikes[0], voltage)
    )


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

       :func:`plif_multi_step`.
    """
    scalar = x.ndim == 0
    spikes, voltage, _ = plif_multi_step(
        x.reshape(1, 1) if scalar else x.unsqueeze(0),
        v.reshape(1) if scalar else v,
        w,
        decay_input,
        v_threshold,
        v_reset,
        surrogate_function,
        detach_reset,
    )
    return (
        (spikes[0, 0], voltage.reshape_as(v).clone())
        if scalar
        else (spikes[0], voltage)
    )


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
    spikes, voltage, recovery, _, _ = izhikevich_multi_step(
        x.unsqueeze(0),
        v,
        w,
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
    return spikes[0], voltage, recovery


def raf_step(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    b: float,
    omega: float,
    dt: float,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <raf_step-cn>` | :ref:`English <raf_step-en>`

    ----

    .. _raf_step-cn:

    * **中文**

    执行 resonate-and-fire (RAF) 神经元的一次显式状态更新。

    RAF 神经元（Izhikevich, *Resonate-and-fire neurons*, Neural Networks 14
    (2001) 883-894）是一个二维线性阈下系统，状态可等价看作复数 :math:`z = u + iv`，
    按固定的衰减率 :math:`b < 0` 和固有角频率 :math:`\omega` 旋转衰减：

    .. math::

        z[t] = z[t-1] \cdot \exp((b + i\omega)\,dt) + x[t]

    real 分量 :math:`u` 接收输入电流；imaginary 分量 :math:`v` 是放电分量，
    对应阈下阻尼振荡，在阈值附近对输入频率有选择性响应。展开为两个实数状态：

    .. math::

        u[t] &= \alpha (u[t-1]\cos\theta - v[t-1]\sin\theta) + x[t] \\
        v[t] &= \alpha (u[t-1]\sin\theta + v[t-1]\cos\theta)

    其中 :math:`\alpha = \exp(b\,dt)`，:math:`\theta = \omega\,dt`。放电后只重置
    :math:`v`（放电分量），:math:`u` 不受影响，这正是产生放电后反弹
    （post-inhibitory rebound）的原因。

    :param x: 当前输入张量，shape 为 ``[N, *]``
    :type x: torch.Tensor
    :param u: 当前实部状态（接收输入的分量），shape、dtype 和 device 与 ``x`` 兼容
    :type u: torch.Tensor
    :param v: 当前虚部状态（放电分量），shape、dtype 和 device 与 ``x`` 兼容
    :type v: torch.Tensor
    :param b: 衰减率，须为负数
    :type b: float
    :param omega: 固有角频率，须为正数
    :type omega: float
    :param dt: 积分步长，须为正数
    :type dt: float
    :param v_threshold: 放电阈值
    :type v_threshold: float
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否在重置分支中分离脉冲梯度
    :type detach_reset: bool
    :return: ``(spike, u_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _raf_step-en:

    * **English**

    Run one explicit state update of a resonate-and-fire (RAF) neuron.

    The RAF neuron (Izhikevich, *Resonate-and-fire neurons*, Neural Networks
    14 (2001) 883-894) is a 2-D linear subthreshold system, equivalently a
    complex state :math:`z = u + iv` that rotates and decays at a fixed rate
    :math:`b < 0` and intrinsic angular frequency :math:`\omega`:

    .. math::

        z[t] = z[t-1] \cdot \exp((b + i\omega)\,dt) + x[t]

    The real component :math:`u` receives the input current; the imaginary
    component :math:`v` is the firing component, giving damped subthreshold
    oscillations and a preferential response to input near :math:`\omega`.
    Expanded into two real states:

    .. math::

        u[t] &= \alpha (u[t-1]\cos\theta - v[t-1]\sin\theta) + x[t] \\
        v[t] &= \alpha (u[t-1]\sin\theta + v[t-1]\cos\theta)

    where :math:`\alpha = \exp(b\,dt)` and :math:`\theta = \omega\,dt`. Only
    :math:`v` (the firing component) is reset after a spike; :math:`u` is left
    untouched, which is what produces post-inhibitory rebound.

    :param x: Current input tensor, shape ``[N, *]``
    :type x: torch.Tensor
    :param u: Current real-part state (receives the input), shape, dtype and
        device compatible with ``x``
    :type u: torch.Tensor
    :param v: Current imaginary-part state (the firing component), shape, dtype
        and device compatible with ``x``
    :type v: torch.Tensor
    :param b: Decay rate, must be negative
    :type b: float
    :param omega: Intrinsic angular frequency, must be positive
    :type omega: float
    :param dt: Integration step size, must be positive
    :type dt: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: Whether to detach spike in the voltage-reset branch
    :type detach_reset: bool
    :return: ``(spike, u_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    alpha = math.exp(b * dt)
    cos_theta = math.cos(omega * dt)
    sin_theta = math.sin(omega * dt)
    u_charged = alpha * (u * cos_theta - v * sin_theta) + x
    v_charged = alpha * (u * sin_theta + v * cos_theta)
    spike = surrogate_function(v_charged - v_threshold)
    v_next = voltage_reset(v_charged, spike, v_threshold, v_reset, detach_reset)
    return spike, u_charged, v_next


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


def clif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    tau: float,
    v_threshold: float,
    spike_function: SurrogateFunction,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <clif_step-cn>` | :ref:`English <clif_step-en>`

    ----

    .. _clif_step-cn:

    * **中文**

    执行一次 ComplementaryLIF 状态转移，返回脉冲、下一膜电位和下一互补电位。
    输入状态须已物化；函数不读取 module memory，也不原地修改输入状态。
    ``spike_function`` 由调用者按训练或推理模式选择。

    :param x: 当前输入，形状为 ``[N, *]``。
    :type x: torch.Tensor
    :param v: 当前膜电位，与 ``x`` 形状、dtype 和 device 相同。
    :type v: torch.Tensor
    :param m: 当前互补电位，与 ``x`` 形状、dtype 和 device 相同。
    :type m: torch.Tensor
    :param tau: 膜电位时间常数，大于 ``1``。
    :type tau: float
    :param v_threshold: 放电阈值。
    :type v_threshold: float
    :param spike_function: 已选定路径的放电函数；可携带替代梯度。
    :type spike_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, v_next, m_next)``，各张量形状与 ``x`` 相同。
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _clif_step-en:

    * **English**

    Run one ComplementaryLIF state transition and return spikes, next membrane
    voltage, and next complementary voltage. States must be materialized. The
    function does not read module memory or mutate input states. The caller
    selects ``spike_function`` for training or inference.

    :param x: Current input shaped ``[N, *]``.
    :type x: torch.Tensor
    :param v: Current membrane voltage with the shape, dtype, and device of ``x``.
    :type v: torch.Tensor
    :param m: Current complementary voltage with the shape, dtype, and device of ``x``.
    :type m: torch.Tensor
    :param tau: Membrane time constant greater than ``1``.
    :type tau: float
    :param v_threshold: Firing threshold.
    :type v_threshold: float
    :param spike_function: Firing function for the selected path, possibly with
        a surrogate gradient.
    :type spike_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, v_next, m_next)`` with tensors shaped like ``x``.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    charged = lif_charge(x, v, tau, False, None)
    m_next = m * torch.sigmoid(charged / tau)
    spike = spike_function(charged - v_threshold)
    m_next = m_next + spike
    v_next = charged - spike * (v_threshold + torch.sigmoid(m_next))
    return spike, v_next, m_next


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
    ``training/eval``，也不原地修改传入 queue。

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
    ``step_mode``, ``training/eval``, and does not mutate
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
    ``step_mode``。调用者必须传入当前 module 已广播的
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
    manage ``training/eval`` or ``step_mode``. The caller passes the parameters
    and surrogate function prepared by the owning module.

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

    执行 ``STBIFNode`` 的单步显式状态转移。函数接收当前输入 ``x``、已物化的
    量化残差 ``q`` 和累计释放量 ``acc_q``，以及量化尺度与边界 tensor，返回
    ``(out, q_next, acc_q_next, cur_output_next)``。量化尺度和边界会在函数入口
    转换到 ``x`` 的 device/dtype。

    该函数保持 SpikeZIP STBIF 的推理语义：输入先除以 ``q_threshold``，累加到
    ``q`` 时使用 ``detach``；``acc_q`` 在判断边界前执行 ``round``；输出
    ``cur_output_next * q_threshold``。函数不读取或写入 ``MemoryModule`` memory，
    不负责 ``training/eval`` 或 ``step_mode``，也不原地修改传入
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

    Run one explicit state transition for ``STBIFNode``. The function receives
    current input ``x``, materialized quantized residual ``q`` and accumulated
    released quantity ``acc_q``, plus ``q_threshold``, ``pos_max``, and
    ``neg_min`` tensors. It returns
    ``(out, q_next, acc_q_next, cur_output_next)``. The scale and bounds are
    converted to the device and dtype of ``x`` at function entry.

    This function preserves SpikeZIP STBIF inference semantics: the input is
    divided by ``q_threshold``; the addition into ``q`` uses ``detach``;
    ``acc_q`` is rounded before bound checks; and the output is
    ``cur_output_next * q_threshold``. The function does not read or write
    ``MemoryModule`` memory, does not manage ``training/eval`` or ``step_mode``, and does not mutate input
    states in place.

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
    out_seq, q_next, acc_q_next, cur_output_next = stbif_multi_step(
        x.unsqueeze(0), q, acc_q, q_threshold, pos_max, neg_min
    )
    return out_seq[0], q_next, acc_q_next, cur_output_next


def lif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float = 2.0,
    decay_input: bool = True,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    surrogate_function: Optional[SurrogateFunction] = None,
    detach_reset: bool = False,
    store_v_seq: bool = False,
    *,
    neuron_storage: Optional[torch.dtype | str] = None,
    neuron_fwd: str = "fp32",
    neuron_bwd: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <lif_multi_step-cn>` | :ref:`English <lif_multi_step-en>`

    ----

    .. _lif_multi_step-cn:

    * **中文**

    对 ``[T, ...]`` 输入执行 LIF 序列状态转移，使用显式初态并返回脉冲、最终电位及可选电位序列。设备类型决定 CPU 或 CUDA 执行，CUDA 实现按设备和执行配置自动选择。兼容 Python surrogate 的参考实现由 Torch 自动微分。

    :param x_seq: 时间优先的输入序列，形状 ``[T, ...]``。
    :type x_seq: torch.Tensor
    :param v: 形状为 ``x_seq.shape[1:]`` 的初始膜电位；dtype 和 device 决定状态的存储类型。
    :type v: torch.Tensor
    :param tau: LIF 时间常数，必须大于 1。
    :type tau: float
    :param decay_input: 是否对输入施加时间衰减。
    :type decay_input: bool
    :param v_threshold: 放电阈值。
    :type v_threshold: float
    :param v_reset: 硬重置电位；``None`` 表示软重置。
    :type v_reset: Optional[float]
    :param surrogate_function: surrogate 对象或 Python callable；``None`` 表示默认 Sigmoid surrogate。
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: 是否截断 reset 分支中的 spike 梯度。
    :type detach_reset: bool
    :param store_v_seq: 是否返回完整电位序列；默认 ``False``。
    :type store_v_seq: bool
    :param neuron_storage: 可选神经元状态存储 dtype。显式精度配置只使用支持该组合的实现。
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: 显式配置的前向计算 dtype。
    :type neuron_fwd: str
    :param neuron_bwd: 显式配置的反向计算 dtype。
    :type neuron_bwd: str
    :return: ``(spike_seq, v_final, v_seq_or_none)``；可选轨迹与输入时间维一致。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _lif_multi_step-en:

    * **English**

    Run the LIF sequence transition on a time-major ``[T, ...]`` input with an explicit initial state. Return spikes, final voltage, and an optional voltage trace. CPU/CUDA execution follows the tensor device; CUDA implementations are selected automatically by device and execution profile. The Torch reference supports Python surrogate callables and autograd.

    :param x_seq: Time-major input sequence shaped ``[T, ...]``.
    :type x_seq: torch.Tensor
    :param v: Initial voltage shaped like ``x_seq.shape[1:]``; its dtype and device determine state storage.
    :type v: torch.Tensor
    :param tau: LIF time constant, greater than 1.
    :type tau: float
    :param decay_input: Whether to decay the input term.
    :type decay_input: bool
    :param v_threshold: Firing threshold.
    :type v_threshold: float
    :param v_reset: Hard-reset voltage; ``None`` selects soft reset.
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate object or Python callable; ``None`` selects the default Sigmoid surrogate.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: Whether to detach spikes through reset.
    :type detach_reset: bool
    :param store_v_seq: Whether to return the full voltage trace; default ``False``.
    :type store_v_seq: bool
    :param neuron_storage: Optional neuron-state storage dtype. Explicit precision settings require a matching implementation.
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: Explicit forward computation dtype.
    :type neuron_fwd: str
    :param neuron_bwd: Explicit backward computation dtype.
    :type neuron_bwd: str
    :return: ``(spike_seq, v_final, v_seq_or_none)``; the optional trace includes the time dimension.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    spec = _surrogate_spec(surrogate_function)
    if neuron_storage is not None:
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        if spec is None:
            raise TypeError(
                "Explicit neuron precision requires a supported built-in surrogate."
            )
        from ..._ops.lif import _selection, triton_precision
        from ..._ops.selection import _require_provider

        _require_provider(
            _selection, x_seq.device, "triton", "the requested neuron precision"
        )

        spikes, voltage, _ = triton_precision._multistep_lif_mp(
            x_seq,
            v,
            decay_input=decay_input,
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            storage_dtype=neuron_storage,
            compute_dtype=neuron_fwd,
            backward_compute_dtype=neuron_bwd,
            store_v_seq=store_v_seq,
            detach_reset=detach_reset,
            surrogate_function=surrogate_function,
        )
        return (
            spikes,
            voltage[-1].clone() if store_v_seq else voltage,
            voltage if store_v_seq else None,
        )

    if spec is None or x_seq.dtype not in _DTYPES or v.dtype != torch.float32:
        from ..._ops.lif.reference import multi_step
        from ..._ops.lif import _selection
        from ..._ops.selection import _require_automatic_torch

        _require_automatic_torch(
            _selection, x_seq.device, "Torch reference implementation required"
        )
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()

        return multi_step(
            x_seq,
            v,
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        )

    from ..._ops.lif import _forward

    surrogate_id, alpha = spec
    spikes, voltage, _ = _forward(
        x_seq,
        v,
        tau,
        decay_input,
        v_threshold,
        v_reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    return (
        (spikes, voltage[-1].clone(), voltage)
        if store_v_seq
        else (spikes, voltage, None)
    )


def if_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    surrogate_function: Optional[SurrogateFunction] = None,
    detach_reset: bool = False,
    store_v_seq: bool = False,
    *,
    neuron_storage: Optional[torch.dtype | str] = None,
    neuron_fwd: str = "fp32",
    neuron_bwd: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <if_multi_step-cn>` | :ref:`English <if_multi_step-en>`

    ----

    .. _if_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA 浮点输入 [T, ...]，T >= 1；神经元维度非空。
        FP32/FP16/BF16 使用注册算子，其余支持的 dtype 使用 Torch 参考实现。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param surrogate_function: 固定参数替代梯度；None 使用 Sigmoid(4)。支持七种已实现的二值替代梯度。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :param neuron_storage: 可选神经元状态存储 dtype；仅 CUDA Triton 支持显式精度配置。
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: 前向计算 dtype；默认 fp32。
    :type neuron_fwd: str
    :param neuron_bwd: 反向计算 dtype；默认 fp32。
    :type neuron_bwd: str
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数无效，或状态形状、dtype、device 不匹配。
    :raises RuntimeError: 输入、状态或参数需要梯度，或设备没有可用实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _if_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param surrogate_function: Fixed-parameter surrogate; None selects Sigmoid(4). Accepts the seven implemented binary surrogates. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: Default False; True detaches the reset spike.  Default: ``False``.
    :type detach_reset: bool
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :param neuron_storage: Optional neuron-state storage dtype. Explicit
        precision settings require CUDA Triton.
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: Forward computation dtype; default fp32.
    :type neuron_fwd: str
    :param neuron_bwd: Backward computation dtype; default fp32.
    :type neuron_bwd: str
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters, or mismatched state shape, dtype, or device.
    :raises RuntimeError: Inputs, states, or parameters require gradients, or no implementation is available.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    spec = _surrogate_spec(surrogate_function)
    if neuron_storage is not None:
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        if spec is None:
            raise TypeError(
                "Explicit neuron precision requires a supported built-in surrogate."
            )
        from ..._ops.if_ import _selection, triton_precision
        from ..._ops.selection import _require_provider

        _require_provider(
            _selection, x_seq.device, "triton", "the requested neuron precision"
        )
        spikes, voltage, _ = triton_precision._multistep_if_mp(
            x_seq,
            v,
            v_threshold=v_threshold,
            v_reset=v_reset,
            storage_dtype=neuron_storage,
            compute_dtype=neuron_fwd,
            backward_compute_dtype=neuron_bwd,
            detach_reset=detach_reset,
            surrogate_function=surrogate_function,
        )
        return spikes, voltage[-1].clone(), voltage if store_v_seq else None
    if spec is None or x_seq.dtype not in _DTYPES or v.dtype != torch.float32:
        from ..._ops.if_.reference import multi_step
        from ..._ops.if_ import _selection
        from ..._ops.selection import _require_automatic_torch

        _require_automatic_torch(
            _selection, x_seq.device, "Torch reference implementation required"
        )
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()

        return multi_step(
            x_seq,
            v,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        )
    from ..._ops.if_ import _forward

    surrogate_id, alpha = spec
    spikes, voltage, _ = _forward(
        x_seq, v, v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_id
    )
    return (
        (spikes, voltage[-1].clone(), voltage)
        if store_v_seq
        else (spikes, voltage, None)
    )


def plif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool = True,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    surrogate_function: Optional[SurrogateFunction] = None,
    detach_reset: bool = False,
    store_v_seq: bool = False,
    *,
    neuron_storage: Optional[torch.dtype | str] = None,
    neuron_fwd: str = "fp32",
    neuron_bwd: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <plif_multi_step-cn>` | :ref:`English <plif_multi_step-en>`

    ----

    .. _plif_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param w: 同设备 FP32/FP16/BF16 可微单元素参数，q=sigmoid(w)，在 FP32 计算及归约后转换梯度。
    :type w: torch.Tensor
    :param decay_input: 是否也衰减输入；默认 True。 默认 ``True``.
    :type decay_input: bool
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param surrogate_function: 固定参数替代梯度；None 使用 Sigmoid(4)。支持七种已实现的二值替代梯度。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :param neuron_storage: 可选神经元状态存储 dtype；仅 CUDA Triton 支持显式精度配置。
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: 前向计算 dtype；默认 fp32。
    :type neuron_fwd: str
    :param neuron_bwd: 反向计算 dtype；默认 fp32。
    :type neuron_bwd: str
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _plif_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param w: Differentiable FP32/FP16/BF16 single-element parameter on the input device, q=sigmoid(w); computation/reduction are FP32 before casting gradients.
    :type w: torch.Tensor
    :param decay_input: Also decay the input; default True. Default: ``True``.
    :type decay_input: bool
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param surrogate_function: Fixed-parameter surrogate; None selects Sigmoid(4). Accepts the seven implemented binary surrogates. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: Default False; True detaches the reset spike.  Default: ``False``.
    :type detach_reset: bool
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :param neuron_storage: Optional neuron-state storage dtype. Explicit
        precision settings require CUDA Triton.
    :type neuron_storage: Optional[torch.dtype | str]
    :param neuron_fwd: Forward computation dtype; default fp32.
    :type neuron_fwd: str
    :param neuron_bwd: Backward computation dtype; default fp32.
    :type neuron_bwd: str
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    spec = _surrogate_spec(surrogate_function)
    if neuron_storage is not None:
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        if spec is None:
            raise TypeError(
                "Explicit neuron precision requires a supported built-in surrogate."
            )
        from ..._ops.plif import _selection, triton_precision
        from ..._ops.selection import _require_provider

        _require_provider(
            _selection, x_seq.device, "triton", "the requested neuron precision"
        )
        spikes, voltage, _ = triton_precision._multistep_plif_mp(
            x_seq,
            v,
            torch.sigmoid(w),
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            storage_dtype=neuron_storage,
            compute_dtype=neuron_fwd,
            backward_compute_dtype=neuron_bwd,
            detach_reset=detach_reset,
            surrogate_function=surrogate_function,
        )
        return spikes, voltage[-1].clone(), voltage if store_v_seq else None
    if (
        spec is None
        or x_seq.dtype not in _DTYPES
        or v.dtype != torch.float32
        or w.ndim != 0
        or w.dtype not in _DTYPES
    ):
        from ..._ops.plif.reference import multi_step
        from ..._ops.plif import _selection
        from ..._ops.selection import _require_automatic_torch

        _require_automatic_torch(
            _selection, x_seq.device, "Torch reference implementation required"
        )
        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()

        return multi_step(
            x_seq,
            v,
            w,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        )
    from ..._ops.plif import _forward

    surrogate_id, alpha = spec
    spikes, voltage, _ = _forward(
        x_seq,
        v,
        w,
        decay_input,
        v_threshold,
        v_reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    return (
        (spikes, voltage[-1].clone(), voltage)
        if store_v_seq
        else (spikes, voltage, None)
    )


def qif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    v_rest: float = 0.0,
    v_c: float = 0.8,
    a0: float = 1.0,
    detach_reset: bool = False,
    surrogate_function: Optional[SurrogateFunction] = None,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <qif_multi_step-cn>` | :ref:`English <qif_multi_step-en>`

    ----

    .. _qif_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param tau: 有限膜电位时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
    :type tau: float
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: 有限静息膜电位。 默认 ``0.0``.
    :type v_rest: float
    :param v_c: 有限临界膜电位。 默认 ``0.8``.
    :type v_c: float
    :param a0: 有限二次项系数。 默认 ``1.0``.
    :type a0: float
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param surrogate_function: 固定参数替代梯度；None 使用 Sigmoid(4)。支持七种已实现的二值替代梯度。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _qif_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param tau: Finite voltage time constant in time steps, greater than one. Default: ``2.0``.
    :type tau: float
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: Finite resting voltage. Default: ``0.0``.
    :type v_rest: float
    :param v_c: Finite critical voltage. Default: ``0.8``.
    :type v_c: float
    :param a0: Finite quadratic coefficient. Default: ``1.0``.
    :type a0: float
    :param detach_reset: Default False; True detaches the reset spike.  Default: ``False``.
    :type detach_reset: bool
    :param surrogate_function: Fixed-parameter surrogate; None selects Sigmoid(4). Accepts the seven implemented binary surrogates. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    spec = _surrogate_spec(surrogate_function)
    registered_profile = (
        x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and v.dtype == torch.float32
    )
    if spec is None or not registered_profile:
        if x_seq.is_cuda:
            from ..._ops.qif import _selection
            from ..._ops.selection import _require_automatic_torch

            _require_automatic_torch(
                _selection, x_seq.device, "Torch reference implementation required"
            )
        from ..._ops.qif.reference import multi_step

        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        s, voltage, _, _ = multi_step(
            x_seq,
            v,
            tau,
            a0,
            v_rest,
            v_c,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        )
    else:
        from ..._ops.qif import _forward

        surrogate_id, alpha = spec
        s, voltage, _, _ = _forward(
            x_seq,
            v,
            tau,
            v_rest,
            v_c,
            a0,
            v_threshold,
            v_reset,
            detach_reset,
            alpha,
            store_v_seq,
            surrogate_id,
        )
    return (
        s,
        voltage[-1].clone() if store_v_seq else voltage,
        voltage if store_v_seq else None,
    )


def eif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    v_rest: float = 0.0,
    theta_rh: float = 1.0,
    delta_t: float = 1.0,
    detach_reset: bool = False,
    surrogate_function: Optional[SurrogateFunction] = None,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <eif_multi_step-cn>` | :ref:`English <eif_multi_step-en>`

    ----

    .. _eif_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param tau: 有限膜电位时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
    :type tau: float
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: 有限静息膜电位。 默认 ``0.0``.
    :type v_rest: float
    :param theta_rh: 有限流变阈值。 默认 ``1.0``.
    :type theta_rh: float
    :param delta_t: 有限正指数项电位宽度。 默认 ``1.0``.
    :type delta_t: float
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param surrogate_function: 固定参数替代梯度；None 使用 Sigmoid(4)。支持七种已实现的二值替代梯度。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _eif_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param tau: Finite voltage time constant in time steps, greater than one. Default: ``2.0``.
    :type tau: float
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: Finite resting voltage. Default: ``0.0``.
    :type v_rest: float
    :param theta_rh: Finite rheobase threshold. Default: ``1.0``.
    :type theta_rh: float
    :param delta_t: Finite positive voltage width of the exponential term. Default: ``1.0``.
    :type delta_t: float
    :param detach_reset: Default False; True detaches the reset spike.  Default: ``False``.
    :type detach_reset: bool
    :param surrogate_function: Fixed-parameter surrogate; None selects Sigmoid(4). Accepts the seven implemented binary surrogates. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    spec = _surrogate_spec(surrogate_function)
    registered_profile = (
        x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and v.dtype == torch.float32
    )
    if spec is None or not registered_profile:
        if x_seq.is_cuda:
            from ..._ops.eif import _selection
            from ..._ops.selection import _require_automatic_torch

            _require_automatic_torch(
                _selection, x_seq.device, "Torch reference implementation required"
            )
        from ..._ops.eif.reference import multi_step

        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        s, voltage, _, _ = multi_step(
            x_seq,
            v,
            tau,
            v_rest,
            theta_rh,
            delta_t,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        )
    else:
        from ..._ops.eif import _forward

        surrogate_id, alpha = spec
        s, voltage, _, _ = _forward(
            x_seq,
            v,
            tau,
            v_rest,
            theta_rh,
            delta_t,
            v_threshold,
            v_reset,
            detach_reset,
            alpha,
            store_v_seq,
            surrogate_id,
        )
    return (
        s,
        voltage[-1].clone() if store_v_seq else voltage,
        voltage if store_v_seq else None,
    )


def izhikevich_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    v_rest: float = 0.0,
    a: float = 0.1,
    b: float = 0.2,
    tau_w: float = 2.0,
    v_c: float = 0.8,
    a0: float = 1.0,
    detach_reset: bool = False,
    surrogate_function: Optional[SurrogateFunction] = None,
    store_state_seq: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    r"""
    **API Language** - :ref:`中文 <izhikevich_multi_step-cn>` | :ref:`English <izhikevich_multi_step-en>`

    ----

    .. _izhikevich_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param w: 与 v 同形状、同设备的 FP32 恢复初态，可微。
    :type w: torch.Tensor
    :param tau: 有限膜电位时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
    :type tau: float
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: 有限静息膜电位。 默认 ``0.0``.
    :type v_rest: float
    :param a: 有限恢复变量耦合系数。 默认 ``0.1``.
    :type a: float
    :param b: 有限发放后的恢复变量增量。 默认 ``0.2``.
    :type b: float
    :param tau_w: 有限正恢复变量时间常数，以时间步为单位。 默认 ``2.0``.
    :type tau_w: float
    :param v_c: 有限临界膜电位。 默认 ``0.8``.
    :type v_c: float
    :param a0: 有限二次项系数。 默认 ``1.0``.
    :type a0: float
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。Izhikevich 恢复脉冲及硬重置的 spike*v_reset 项始终可微。 默认 ``False``.
    :type detach_reset: bool
    :param surrogate_function: 固定参数替代梯度；None 使用 Sigmoid(4)。支持七种已实现的二值替代梯度。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_state_seq: 默认 False；True 返回完整 FP32 膜电位及恢复状态轨迹。 默认 ``False``.
    :type store_state_seq: bool
    :return: (spike_seq, v_final, w_final, v_seq_or_none, w_seq_or_none)，均与输入同设备；轨迹受 store_state_seq 控制。
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _izhikevich_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param w: Differentiable FP32 initial recovery state with the shape and device of v.
    :type w: torch.Tensor
    :param tau: Finite voltage time constant in time steps, greater than one. Default: ``2.0``.
    :type tau: float
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param v_rest: Finite resting voltage. Default: ``0.0``.
    :type v_rest: float
    :param a: Finite recovery coupling coefficient. Default: ``0.1``.
    :type a: float
    :param b: Finite post-spike recovery increment. Default: ``0.2``.
    :type b: float
    :param tau_w: Finite positive recovery time constant in time steps. Default: ``2.0``.
    :type tau_w: float
    :param v_c: Finite critical voltage. Default: ``0.8``.
    :type v_c: float
    :param a0: Finite quadratic coefficient. Default: ``1.0``.
    :type a0: float
    :param detach_reset: Default False; True detaches the reset spike. Izhikevich recovery spikes and the hard-reset spike*v_reset term remain differentiable. Default: ``False``.
    :type detach_reset: bool
    :param surrogate_function: Fixed-parameter surrogate; None selects Sigmoid(4). Accepts the seven implemented binary surrogates. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param store_state_seq: Default False; True returns complete FP32 voltage and recovery traces. Default: ``False``.
    :type store_state_seq: bool
    :return: (spike_seq, v_final, w_final, v_seq_or_none, w_seq_or_none), on the input device; store_state_seq controls traces.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    spec = _surrogate_spec(surrogate_function)
    registered_profile = (
        x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and v.dtype == torch.float32
        and w.dtype == torch.float32
    )
    if spec is None or not registered_profile:
        if x_seq.is_cuda:
            from ..._ops.izhikevich import _selection
            from ..._ops.selection import _require_automatic_torch

            _require_automatic_torch(
                _selection, x_seq.device, "Torch reference implementation required"
            )
        from ..._ops.izhikevich.reference import multi_step

        if surrogate_function is None:
            from ..surrogate import Sigmoid

            surrogate_function = Sigmoid()
        s, voltage, recovery, _, _ = multi_step(
            x_seq,
            v,
            w,
            tau,
            v_rest,
            v_c,
            a0,
            a,
            b,
            tau_w,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_state_seq,
        )
    else:
        from ..._ops.izhikevich import _forward

        surrogate_id, alpha = spec
        s, voltage, recovery, _, _ = _forward(
            x_seq,
            v,
            w,
            tau,
            v_rest,
            v_c,
            a0,
            a,
            b,
            tau_w,
            v_threshold,
            v_reset,
            detach_reset,
            alpha,
            store_state_seq,
            surrogate_id,
        )
    return (
        s,
        voltage[-1].clone() if store_state_seq else voltage,
        recovery[-1].clone() if store_state_seq else recovery,
        voltage if store_state_seq else None,
        recovery if store_state_seq else None,
    )


def ilif_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    surrogate_function: Optional[SurrogateFunction] = None,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <ilif_multi_step-cn>` | :ref:`English <ilif_multi_step-en>`

    ----

    .. _ilif_multi_step-cn:

    * **中文**

    执行显式状态序列转移，不修改输入或模块 memory。输入设备自动分发到支持的注册实现；不适用的 surrogate 或状态 dtype 使用 Torch 参考公式。具体输出状态与梯度遵循神经元动力学定义。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param tau: 有限膜电位时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
    :type tau: float
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。 默认 ``1.0``.
    :type v_threshold: float
    :param surrogate_function: 仅接受 MultiLevelSpikeCount(spiking=True)；None 使用 MultiLevelSpikeCount(4)，窗口 [0, 4]。 默认 ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: 默认 False；True 分离重置脉冲梯度。 默认 ``False``.
    :type detach_reset: bool
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。
    :raises TypeError: 替代梯度类型或参数不受支持。

    ----

    .. _ilif_multi_step-en:

    * **English**

    Run an explicit-state sequence transition without mutating inputs or module memory. The input device selects a supported registered implementation; unsupported surrogate or state-dtype profiles use the Torch reference formulas. Output states and gradients follow the neuron dynamics.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param tau: Finite voltage time constant in time steps, greater than one. Default: ``2.0``.
    :type tau: float
    :param v_threshold: Finite firing threshold; positive for I-LIF. Default: ``1.0``.
    :type v_threshold: float
    :param surrogate_function: Accepts only MultiLevelSpikeCount(spiking=True); None selects MultiLevelSpikeCount(4), with window [0, 4]. Default: ``None``.
    :type surrogate_function: Optional[SurrogateFunction]
    :param detach_reset: Default False; True detaches the reset spike.  Default: ``False``.
    :type detach_reset: bool
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    :raises TypeError: Unsupported surrogate type or parameters.
    """
    from ..._ops.ilif import _forward
    from ..surrogate import MultiLevelSpikeCount

    function = (
        MultiLevelSpikeCount(4) if surrogate_function is None else surrogate_function
    )
    if type(function) is not MultiLevelSpikeCount or not function.spiking:
        raise TypeError("Registered I-LIF requires MultiLevelSpikeCount(spiking=True)")
    if (
        x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and v.dtype == torch.float32
    ):
        from ..._ops.ilif import _forward

        s, voltage, _ = _forward(
            x_seq,
            v,
            tau,
            float(function.max_spike_count),
            float(function.grad_min),
            float(function.grad_max),
            v_threshold,
            detach_reset,
            store_v_seq,
        )
    else:
        if x_seq.is_cuda:
            from ..._ops.ilif import _selection
            from ..._ops.selection import _require_automatic_torch

            _require_automatic_torch(
                _selection, x_seq.device, "Torch reference implementation required"
            )
        from ..._ops.ilif.reference import multi_step

        s, voltage, _ = multi_step(
            x_seq,
            v,
            tau,
            float(function.max_spike_count),
            float(function.grad_min),
            float(function.grad_max),
            v_threshold,
            detach_reset,
            store_v_seq,
        )
    return (
        s,
        voltage[-1].clone() if store_v_seq else voltage,
        voltage if store_v_seq else None,
    )


def activation_aware_if_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_offset: torch.Tensor,
    channel_size: int,
    inner_size: int,
    v_reset: Optional[float] = 0.0,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <activation_aware_if_multi_step-cn>` | :ref:`English <activation_aware_if_multi_step-en>`

    ----

    .. _activation_aware_if_multi_step-cn:

    * **中文**

    执行显式状态序列推理，不修改输入或模块 memory。算子按输入设备自动分发；该路径不支持输入、状态或参数梯度。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param v: 同设备 FP32 初始膜电位，形状为一个输入时间步；支持非连续存储。
    :type v: torch.Tensor
    :param v_threshold: 有限发放阈值；I-LIF 必须为正。ActivationAwareIF 使用同设备 FP32 单元素或 channel_size 元素张量。
    :type v_threshold: torch.Tensor
    :param v_offset: 同设备 FP32 单元素或 channel_size 元素发放电位偏移张量。
    :type v_offset: torch.Tensor
    :param channel_size: 正通道数；与 inner_size 的乘积必须整除状态元素数。
    :type channel_size: int
    :param inner_size: 每个通道内部的正元素数；通道索引为扁平索引 // inner_size % channel_size。
    :type inner_size: int
    :param v_reset: 有限硬重置电位；None 表示软重置。 默认 ``0.0``.
    :type v_reset: Optional[float]
    :param store_v_seq: 默认 False；True 返回完整 FP32 膜电位轨迹。 默认 ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none)；脉冲与输入同形状，最终状态与初态同形状；轨迹受 store_v_seq 控制，均与输入同设备。
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。

    ----

    .. _activation_aware_if_multi_step-en:

    * **English**

    Run an explicit-state inference sequence without mutating inputs or module memory. The operator dispatches by input device. Inputs, states, and parameters must not require gradients.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1`` and nonempty neuron dimensions. Default accelerated profiles support FP32, FP16, or BF16.
    :type x_seq: torch.Tensor
    :param v: FP32 initial voltage on the input device, shaped like one input step; noncontiguous storage is supported.
    :type v: torch.Tensor
    :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF takes an FP32 scalar or channel_size-element tensor on the input device.
    :type v_threshold: torch.Tensor
    :param v_offset: FP32 scalar or channel_size-element firing offset on the input device.
    :type v_offset: torch.Tensor
    :param channel_size: Positive channel count; channels times inner_size must divide the state element count.
    :type channel_size: int
    :param inner_size: Positive elements per channel; channel index is flat_index // inner_size % channel_size.
    :type inner_size: int
    :param v_reset: Finite hard-reset voltage; None selects soft reset. Default: ``0.0``.
    :type v_reset: Optional[float]
    :param store_v_seq: Default False; True returns the complete FP32 voltage trace. Default: ``False``.
    :type store_v_seq: bool
    :return: (spike_seq, v_final, v_seq_or_none); spikes match input shape, final state matches the initial state, and store_v_seq controls traces; all on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    """
    from ..._ops.activation_aware_if import _forward

    s, voltage = _forward(
        x_seq, v, v_threshold, v_offset, channel_size, inner_size, v_reset, store_v_seq
    )
    return (
        s,
        voltage[-1].clone() if store_v_seq else voltage,
        voltage if store_v_seq else None,
    )


def stbif_multi_step(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stbif_multi_step-cn>` | :ref:`English <stbif_multi_step-en>`

    ----

    .. _stbif_multi_step-cn:

    * **中文**

    执行显式状态序列推理，不修改输入或模块 memory。算子按输入设备自动分发；该路径不支持输入、状态或参数梯度。

    :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 [T, ...]，T >= 1；神经元维度非空。
    :type x_seq: torch.Tensor
    :param q: 与一个输入步同形状、同设备的状态；注册算子使用 FP32，参考实现与输入 dtype 相同。
    :type q: torch.Tensor
    :param acc_q: 与 q 同形状、同设备且 dtype 相同的已释放数量状态。
    :type acc_q: torch.Tensor
    :param q_threshold: 单元素浮点量化尺度张量；注册算子要求与状态同设备的 FP32。
    :type q_threshold: torch.Tensor
    :param pos_max: 单元素浮点正释放上界张量；注册算子要求与状态同设备的 FP32。
    :type pos_max: torch.Tensor
    :param neg_min: 单元素浮点负释放下界张量；注册算子要求与状态同设备的 FP32。
    :type neg_min: torch.Tensor
    :return: (out, q_final, acc_q_final, cur_output)；out 与输入同形状和 dtype。
        注册算子的状态为 FP32，Torch 参考实现的状态 dtype 与输入一致；所有输出均与输入同 device。
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: 标量参数或实现配置无效。
    :raises RuntimeError: 张量约束不满足或没有可用设备实现。

    ----

    .. _stbif_multi_step-en:

    * **English**

    Run an explicit-state inference sequence without mutating inputs or module memory. The operator dispatches by input device. Inputs, states, and parameters must not require gradients.

    :param x_seq: Floating-point CPU or NVIDIA CUDA input ``[T, ...]`` with ``T >= 1``
        and nonempty neuron dimensions. FP32, FP16, and BF16 use registered operators;
        other supported dtypes use the Torch reference implementation.
    :type x_seq: torch.Tensor
    :param q: Residual state shaped like one input step and on the same device.
        Registered operators use FP32; the reference implementation matches the input dtype.
    :type q: torch.Tensor
    :param acc_q: Accumulated released quantity with the shape, device, and dtype of q.
    :type acc_q: torch.Tensor
    :param q_threshold: Single-element floating-point quantization scale. Registered operators require FP32 on the input device.
    :type q_threshold: torch.Tensor
    :param pos_max: Single-element floating-point positive release bound. Registered operators require FP32 on the input device.
    :type pos_max: torch.Tensor
    :param neg_min: Single-element floating-point negative release bound. Registered operators require FP32 on the input device.
    :type neg_min: torch.Tensor
    :return: (out, q_final, acc_q_final, cur_output); out matches input shape/dtype.
        Registered operators return FP32 states; the Torch reference matches the input dtype.
        All outputs are on the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Tensor constraints violated or no available device implementation.
    """
    if x_seq.shape[0] == 0:
        raise ValueError("STBIF does not support empty input sequences.")
    if any(
        tensor.requires_grad
        for tensor in (x_seq, q, acc_q, q_threshold, pos_max, neg_min)
    ):
        raise RuntimeError("STBIF inference transitions do not support autograd.")
    registered_profile = (
        x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and q.dtype == torch.float32
        and acc_q.dtype == torch.float32
        and all(
            tensor.dtype == torch.float32 for tensor in (q_threshold, pos_max, neg_min)
        )
    )
    if not registered_profile:
        from ..._ops.stbif import _selection
        from ..._ops.selection import _require_automatic_torch
        from ..._ops.stbif.reference import step
        from ..._ops.stbif.validation import _check_reference

        _check_reference(x_seq, q, acc_q, q_threshold, pos_max, neg_min)
        _require_automatic_torch(
            _selection, x_seq.device, "Torch reference implementation required"
        )
        outputs = []
        for current in x_seq:
            output, q, acc_q, cur_output = step(
                current, q, acc_q, q_threshold, pos_max, neg_min
            )
            outputs.append(output.to(x_seq.dtype))
        return torch.stack(outputs), q, acc_q, cur_output
    if any(parameter.numel() != 1 for parameter in (q_threshold, pos_max, neg_min)):
        raise ValueError("parameters must be scalar tensors")
    if (
        q.shape != x_seq.shape[1:]
        or q.dtype != torch.float32
        or q.device != x_seq.device
        or acc_q.shape != q.shape
        or acc_q.dtype != torch.float32
        or acc_q.device != x_seq.device
    ):
        raise ValueError("state shape, dtype, and device must match")
    from ..._ops.stbif import _forward

    return _forward(x_seq, q, acc_q, q_threshold, pos_max, neg_min)
