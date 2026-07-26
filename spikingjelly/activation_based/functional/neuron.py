from __future__ import annotations

from typing import Any, Callable, Optional

import torch


__all__ = [
    "neuron_fire",
    "hard_reset",
    "soft_reset",
    "if_charge",
    "lif_charge",
    "plif_charge",
    "qif_charge",
    "eif_charge",
    "adaptive_current_update",
    "adaptive_reset",
    "izhikevich_charge",
    "klif_charge",
    "klif_reset",
    "cuba_lif_charge",
    "lava_cuba_lif_charge",
    "lava_cuba_lif_single_step",
    "lava_cuba_lif_multi_step",
    "liaf_output",
    "mpbn_fire",
    "online_lif_charge",
    "ottt_trace_update",
    "if_single_step",
    "if_multi_step",
    "lif_single_step",
    "lif_multi_step",
    "lif_single_step_with_pre_spike_mean",
    "lif_multi_step_with_pre_spike_mean",
    "plif_single_step",
    "plif_multi_step",
    "if_multi_step_inductor",
    "lif_multi_step_inductor",
    "plif_multi_step_inductor",
    "if_single_step_cupy",
    "lif_single_step_cupy",
    "if_multi_step_cupy",
    "lif_multi_step_cupy",
    "plif_multi_step_cupy",
    "qif_multi_step_cupy",
    "eif_multi_step_cupy",
    "izhikevich_multi_step_cupy",
    "if_multi_step_triton",
    "lif_multi_step_triton",
    "plif_multi_step_triton",
    "masked_psn_advance_queue",
    "masked_psn_single_step_from_queue",
    "sliding_psn_single_step",
    "gated_lif_multi_step",
    "stbif_single_step",
    "stbif_multi_step_torch",
    "activation_aware_if_single_step",
    "activation_aware_if_multi_step",
    "activation_aware_if_multi_step_triton",
]


SurrogateFunction = Callable[[torch.Tensor], torch.Tensor]


def neuron_fire(
    v: torch.Tensor,
    v_threshold: float,
    surrogate_function: SurrogateFunction,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <neuron_fire-cn>` | :ref:`English <neuron_fire-en>`

    ----

    .. _neuron_fire-cn:

    * **中文**

    根据膜电位和阈值计算脉冲。该函数不读取 module 状态，也不判断
    ``training/eval``；调用者必须传入当前执行路径使用的替代函数。

    :param v: 已物化的膜电位张量，shape、dtype 和 device 与当前输入一致
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param surrogate_function: 作用于 ``v - v_threshold`` 的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: 脉冲张量
    :rtype: torch.Tensor

    ----

    .. _neuron_fire-en:

    * **English**

    Compute spikes from membrane voltage and threshold. This function does not read
    module state or inspect ``training/eval``; the caller must pass the surrogate
    function for the selected execution path.

    :param v: Materialized membrane voltage tensor with the same shape, dtype, and
        device as the current input
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param surrogate_function: Surrogate function applied to ``v - v_threshold``
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: Spike tensor
    :rtype: torch.Tensor
    """
    return surrogate_function(v - v_threshold)


def hard_reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    v_reset: float,
    detach_reset: bool = False,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <hard_reset-cn>` | :ref:`English <hard_reset-en>`

    ----

    .. _hard_reset-cn:

    * **中文**

    执行硬重置：发放脉冲的位置置为 ``v_reset``，其余位置保持当前膜电位。
    该函数不原地修改 ``v`` 或 ``spike``。

    :param v: 当前膜电位
    :type v: torch.Tensor
    :param spike: 当前脉冲
    :type spike: torch.Tensor
    :param v_reset: 重置电压
    :type v_reset: float
    :param detach_reset: 是否在重置分支中分离 ``spike`` 的计算图
    :type detach_reset: bool
    :return: 重置后的膜电位
    :rtype: torch.Tensor

    ----

    .. _hard_reset-en:

    * **English**

    Apply hard reset: positions that fired are set to ``v_reset`` and other
    positions keep the current membrane voltage. This function does not mutate
    ``v`` or ``spike`` in place.

    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param spike: Current spike tensor
    :type spike: torch.Tensor
    :param v_reset: Reset voltage
    :type v_reset: float
    :param detach_reset: Whether to detach ``spike`` in the reset branch
    :type detach_reset: bool
    :return: Membrane voltage after reset
    :rtype: torch.Tensor
    """
    spike_d = spike.detach() if detach_reset else spike
    return v_reset * spike_d + (1.0 - spike_d) * v


def soft_reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    v_threshold: float,
    detach_reset: bool = False,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <soft_reset-cn>` | :ref:`English <soft_reset-en>`

    ----

    .. _soft_reset-cn:

    * **中文**

    执行软重置：发放脉冲的位置从当前膜电位中减去 ``v_threshold``。
    该函数不原地修改 ``v`` 或 ``spike``。

    :param v: 当前膜电位
    :type v: torch.Tensor
    :param spike: 当前脉冲
    :type spike: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param detach_reset: 是否在重置分支中分离 ``spike`` 的计算图
    :type detach_reset: bool
    :return: 重置后的膜电位
    :rtype: torch.Tensor

    ----

    .. _soft_reset-en:

    * **English**

    Apply soft reset by subtracting ``v_threshold`` from the current membrane
    voltage where spikes fired. This function does not mutate ``v`` or ``spike``.

    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param spike: Current spike tensor
    :type spike: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param detach_reset: Whether to detach ``spike`` in the reset branch
    :type detach_reset: bool
    :return: Membrane voltage after reset
    :rtype: torch.Tensor
    """
    spike_d = spike.detach() if detach_reset else spike
    return v - spike_d * v_threshold


def if_charge(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <if_charge-cn>` | :ref:`English <if_charge-en>`

    ----

    .. _if_charge-cn:

    * **中文**

    IF 神经元充电：``v + x``。输入 ``v`` 必须是已物化的 tensor state。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _if_charge-en:

    * **English**

    IF neuron charge: ``v + x``. The input ``v`` must be a materialized tensor
    state.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :return: Membrane voltage after charge
    :rtype: torch.Tensor
    """
    return v + x


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

    LIF 神经元充电。``v_reset is None`` 时按现有实现使用 ``0.0`` 作为充电方程中的
    reset 参照值；函数不管理 soft/hard reset 分支。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :param tau: 膜电位时间常数，必须与 module 路径使用的值一致
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _lif_charge-en:

    * **English**

    LIF neuron charge. When ``v_reset is None``, the existing implementation uses
    ``0.0`` as the reset reference in the charge equation; this function does not
    manage the soft/hard reset branch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :param tau: Membrane time constant, matching the module path
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :return: Membrane voltage after charge
    :rtype: torch.Tensor
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        return v + (x - (v - v_reset_value)) / tau
    return v - (v - v_reset_value) / tau + x


def plif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    v_reset: Optional[float],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <plif_charge-cn>` | :ref:`English <plif_charge-en>`

    ----

    .. _plif_charge-cn:

    * **中文**

    PLIF 神经元充电，其中 ``w.sigmoid()`` 为可学习的 ``1 / tau``。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :param w: PLIF 的可学习参数
    :type w: torch.Tensor
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _plif_charge-en:

    * **English**

    PLIF neuron charge where ``w.sigmoid()`` is the learnable ``1 / tau``.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :param w: Learnable PLIF parameter
    :type w: torch.Tensor
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :return: Membrane voltage after charge
    :rtype: torch.Tensor
    """
    reciprocal_tau = w.sigmoid()
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        return v + (x - (v - v_reset_value)) * reciprocal_tau
    return v - (v - v_reset_value) * reciprocal_tau + x


def qif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    a0: float,
    v_rest: float,
    v_c: float,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <qif_charge-cn>` | :ref:`English <qif_charge-en>`

    ----

    .. _qif_charge-cn:

    * **中文**

    QIF 神经元充电公式：
    ``v + (x + a0 * (v - v_rest) * (v - v_c)) / tau``。函数只接收显式
    tensor state，不读取 ``MemoryModule`` memory，不负责 fire/reset、
    ``training/eval`` 或 backend dispatch。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param tau: 时间常数
    :type tau: float
    :param a0: 二次项系数
    :type a0: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_c: 临界电位参数
    :type v_c: float
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _qif_charge-en:

    * **English**

    QIF neuron charge equation:
    ``v + (x + a0 * (v - v_rest) * (v - v_c)) / tau``. The function receives
    explicit tensor state only and does not read ``MemoryModule`` memory or
    manage fire/reset, ``training/eval``, or backend dispatch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param tau: Time constant
    :type tau: float
    :param a0: Quadratic-term coefficient
    :type a0: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_c: Critical-voltage parameter
    :type v_c: float
    :return: Charged membrane voltage
    :rtype: torch.Tensor
    """
    return v + (x + a0 * (v - v_rest) * (v - v_c)) / tau


def eif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_rest: float,
    delta_t: float,
    theta_rh: float,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <eif_charge-cn>` | :ref:`English <eif_charge-en>`

    ----

    .. _eif_charge-cn:

    * **中文**

    EIF 神经元充电公式：
    ``v + (x + v_rest - v + delta_t * exp((v - theta_rh) / delta_t)) / tau``。
    函数只接收显式 tensor state，不读取 ``MemoryModule`` memory，不负责
    fire/reset、``training/eval`` 或 backend dispatch。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param tau: 时间常数
    :type tau: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param delta_t: 指数项尺度
    :type delta_t: float
    :param theta_rh: rheobase 阈值
    :type theta_rh: float
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _eif_charge-en:

    * **English**

    EIF neuron charge equation:
    ``v + (x + v_rest - v + delta_t * exp((v - theta_rh) / delta_t)) / tau``.
    The function receives explicit tensor state only and does not read
    ``MemoryModule`` memory or manage fire/reset, ``training/eval``, or backend
    dispatch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param tau: Time constant
    :type tau: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param delta_t: Exponential-term scale
    :type delta_t: float
    :param theta_rh: Rheobase threshold
    :type theta_rh: float
    :return: Charged membrane voltage
    :rtype: torch.Tensor
    """
    return v + (x + v_rest - v + delta_t * torch.exp((v - theta_rh) / delta_t)) / tau


def adaptive_current_update(
    w: torch.Tensor,
    v: torch.Tensor,
    tau_w: float,
    a: float,
    v_rest: float,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <adaptive_current_update-cn>` | :ref:`English <adaptive_current_update-en>`

    ----

    .. _adaptive_current_update-cn:

    * **中文**

    更新适应性电流：``w + (a * (v - v_rest) - w) / tau_w``。函数不读取
    ``MemoryModule`` memory，不负责 charge、fire/reset、``training/eval`` 或
    backend dispatch。

    :param w: 当前适应性电流
    :type w: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param tau_w: 适应性电流时间常数
    :type tau_w: float
    :param a: 阈下耦合参数
    :type a: float
    :param v_rest: 静息电位
    :type v_rest: float
    :return: 更新后的适应性电流
    :rtype: torch.Tensor

    ----

    .. _adaptive_current_update-en:

    * **English**

    Update adaptation current as ``w + (a * (v - v_rest) - w) / tau_w``. The
    function does not read ``MemoryModule`` memory and does not manage charge,
    fire/reset, ``training/eval``, or backend dispatch.

    :param w: Current adaptation current
    :type w: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param tau_w: Adaptation-current time constant
    :type tau_w: float
    :param a: Subthreshold coupling parameter
    :type a: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :return: Updated adaptation current
    :rtype: torch.Tensor
    """
    return w + (a * (v - v_rest) - w) / tau_w


def adaptive_reset(
    v: torch.Tensor,
    w: torch.Tensor,
    spike: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    b: float,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <adaptive_reset-cn>` | :ref:`English <adaptive_reset-en>`

    ----

    .. _adaptive_reset-cn:

    * **中文**

    执行带适应电流的 reset。``v_reset is None`` 时使用 soft reset，否则使用 hard
    reset；``w`` 总是加上 ``b * spike``。函数不读取或写入 ``MemoryModule`` memory。

    :param v: 当前膜电位
    :type v: torch.Tensor
    :param w: 当前适应性电流
    :type w: torch.Tensor
    :param spike: 当前脉冲
    :type spike: torch.Tensor
    :param v_threshold: soft reset 使用的阈值
    :type v_threshold: float
    :param v_reset: hard reset 使用的重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param b: 脉冲触发的适应电流增量
    :type b: float
    :param detach_reset: 是否在膜电位 reset 分支中 detach spike
    :type detach_reset: bool
    :return: ``(v_next, w_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _adaptive_reset-en:

    * **English**

    Apply reset with adaptation current. ``v_reset is None`` selects soft reset;
    otherwise hard reset is used. ``w`` always receives ``b * spike``. The
    function does not read or write ``MemoryModule`` memory.

    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param w: Current adaptation current
    :type w: torch.Tensor
    :param spike: Current spike tensor
    :type spike: torch.Tensor
    :param v_threshold: Threshold used by soft reset
    :type v_threshold: float
    :param v_reset: Hard-reset voltage; ``None`` selects soft reset
    :type v_reset: Optional[float]
    :param b: Spike-triggered adaptation increment
    :type b: float
    :param detach_reset: Whether to detach spike in the voltage-reset branch
    :type detach_reset: bool
    :return: ``(v_next, w_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    spike_d = spike.detach() if detach_reset else spike
    if v_reset is None:
        v_next = v - spike_d * v_threshold
    else:
        v_next = (1.0 - spike_d) * v + spike * v_reset
    return v_next, w + b * spike


def izhikevich_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    a0: float,
    v_rest: float,
    v_c: float,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <izhikevich_charge-cn>` | :ref:`English <izhikevich_charge-en>`

    ----

    .. _izhikevich_charge-cn:

    * **中文**

    Izhikevich 神经元充电公式：
    ``v + (x + a0 * (v - v_rest) * (v - v_c) - w) / tau``。函数只接收显式
    tensor state，不读取 ``MemoryModule`` memory，不负责适应电流更新、fire/reset、
    ``training/eval`` 或 backend dispatch。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param w: 当前适应性电流
    :type w: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param a0: 二次项系数
    :type a0: float
    :param v_rest: 静息电位
    :type v_rest: float
    :param v_c: 临界电位参数
    :type v_c: float
    :return: 充电后的膜电位
    :rtype: torch.Tensor

    ----

    .. _izhikevich_charge-en:

    * **English**

    Izhikevich neuron charge equation:
    ``v + (x + a0 * (v - v_rest) * (v - v_c) - w) / tau``. The function receives
    explicit tensor state only and does not read ``MemoryModule`` memory or
    manage adaptation-current update, fire/reset, ``training/eval``, or backend
    dispatch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param w: Current adaptation current
    :type w: torch.Tensor
    :param tau: Membrane-voltage time constant
    :type tau: float
    :param a0: Quadratic-term coefficient
    :type a0: float
    :param v_rest: Resting voltage
    :type v_rest: float
    :param v_c: Critical-voltage parameter
    :type v_c: float
    :return: Charged membrane voltage
    :rtype: torch.Tensor
    """
    return v + (x + a0 * (v - v_rest) * (v - v_c) - w) / tau


def klif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_reset: Optional[float],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <klif_charge-cn>` | :ref:`English <klif_charge-en>`

    ----

    .. _klif_charge-cn:

    * **中文**

    KLIF 神经元 charge 和 ``relu(k * h)`` 变换。``v_reset is None`` 时 charge 公式
    中使用 ``0.0``。函数不读取 ``MemoryModule`` memory，不负责 fire/reset、
    ``training/eval`` 或 backend dispatch，且不原地修改输入。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位
    :type v: torch.Tensor
    :param k: KLIF 可学习缩放参数
    :type k: torch.Tensor
    :param tau: 时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_reset: 重置电位；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :return: charge 后的膜电位
    :rtype: torch.Tensor

    ----

    .. _klif_charge-en:

    * **English**

    Run KLIF charge followed by ``relu(k * h)``. When ``v_reset is None``, the
    charge equation uses ``0.0``. The function does not read ``MemoryModule``
    memory and does not manage fire/reset, ``training/eval``, or backend
    dispatch, and it does not mutate inputs in place.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage
    :type v: torch.Tensor
    :param k: Learnable KLIF scale
    :type k: torch.Tensor
    :param tau: Time constant
    :type tau: float
    :param decay_input: Whether the input participates in decay
    :type decay_input: bool
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :return: Charged membrane voltage
    :rtype: torch.Tensor
    """
    v_reset_value = 0.0 if v_reset is None else v_reset
    if decay_input:
        h = v + (x - (v - v_reset_value)) / tau
    else:
        h = v - (v - v_reset_value) / tau + x
    return torch.relu(k * h)


def klif_reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    k: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    scale_reset: bool,
    detach_reset: bool = False,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <klif_reset-cn>` | :ref:`English <klif_reset-en>`

    ----

    .. _klif_reset-cn:

    * **中文**

    执行 KLIF reset。``scale_reset=True`` 时在 reset 后按既有 KLIF 语义除以
    ``k`` 或先对 ``v / k`` 执行 hard reset；否则使用普通 hard/soft reset。
    函数不读取或写入 ``MemoryModule`` memory。

    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :param spike: 当前脉冲张量
    :type spike: torch.Tensor
    :param k: KLIF 缩放参数
    :type k: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: reset 电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param scale_reset: 是否使用 KLIF 的缩放 reset 语义
    :type scale_reset: bool
    :param detach_reset: 是否在 reset 分支中分离 ``spike`` 的计算图
    :type detach_reset: bool
    :return: reset 后的膜电位
    :rtype: torch.Tensor

    ----

    .. _klif_reset-en:

    * **English**

    Apply KLIF reset. With ``scale_reset=True``, it follows existing KLIF
    semantics by dividing the reset result by ``k`` or applying hard reset to
    ``v / k`` first; otherwise it uses regular hard/soft reset. The function
    does not read or write ``MemoryModule`` memory.

    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :param spike: Current spike tensor
    :type spike: torch.Tensor
    :param k: KLIF scaling parameter
    :type k: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` indicates soft reset
    :type v_reset: Optional[float]
    :param scale_reset: Whether to use KLIF scaled-reset semantics
    :type scale_reset: bool
    :param detach_reset: Whether to detach ``spike`` in the reset branch
    :type detach_reset: bool
    :return: Membrane voltage after reset
    :rtype: torch.Tensor
    """
    spike_d = spike.detach() if detach_reset else spike
    if scale_reset:
        if v_reset is None:
            return soft_reset(v, spike_d, v_threshold, False) / k
        return hard_reset(v / k, spike_d, v_reset, False)
    if v_reset is None:
        return soft_reset(v, spike_d, v_threshold, False)
    return hard_reset(v, spike_d, v_reset, False)


def cuba_lif_charge(
    x: torch.Tensor,
    c: torch.Tensor,
    v: torch.Tensor,
    c_decay: float,
    v_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <cuba_lif_charge-cn>` | :ref:`English <cuba_lif_charge-en>`

    ----

    .. _cuba_lif_charge-cn:

    * **中文**

    CUBA-LIF charge：先更新输入电流 ``c_next = c * c_decay + x``，再更新膜电位
    ``v_next = v * v_decay + c_next``。函数不读取 ``MemoryModule`` memory，不负责
    fire/reset。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param c: 当前输入电流张量
    :type c: torch.Tensor
    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :param c_decay: 输入电流衰减系数
    :type c_decay: float
    :param v_decay: 膜电位衰减系数
    :type v_decay: float
    :return: ``(c_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _cuba_lif_charge-en:

    * **English**

    CUBA-LIF charge: first update current ``c_next = c * c_decay + x``, then
    membrane potential ``v_next = v * v_decay + c_next``. The function does not
    read ``MemoryModule`` memory and does not manage fire/reset.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param c: Current input-current tensor
    :type c: torch.Tensor
    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :param c_decay: Input-current decay factor
    :type c_decay: float
    :param v_decay: Membrane-voltage decay factor
    :type v_decay: float
    :return: ``(c_next, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    c_next = c * c_decay + x
    return c_next, v * v_decay + c_next


def lava_cuba_lif_charge(
    x: torch.Tensor,
    current_state: torch.Tensor,
    voltage_state: torch.Tensor,
    current_decay: torch.Tensor,
    voltage_decay: torch.Tensor,
    s_scale: float,
    norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lava_cuba_lif_charge-cn>` | :ref:`English <lava_cuba_lif_charge-en>`

    ----

    .. _lava_cuba_lif_charge-cn:

    * **中文**

    执行 ``lava_exchange.CubaLIFNode`` 的量化 charge 路径：先用
    ``LeakyIntegratorStep`` 更新电流，再可选调用调用方已经选定的 ``norm``，最后用
    同一量化积分器更新膜电位。函数不读取 ``MemoryModule`` memory，不判断
    ``training/eval``，也不负责 fire/reset。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param current_state: 当前电流状态张量
    :type current_state: torch.Tensor
    :param voltage_state: 当前电压状态张量
    :type voltage_state: torch.Tensor
    :param current_decay: 已由 module 持有的电流衰减 tensor
    :type current_decay: torch.Tensor
    :param voltage_decay: 已由 module 持有的电压衰减 tensor
    :type voltage_decay: torch.Tensor
    :param s_scale: Lava 突触缩放因子
    :type s_scale: float
    :param norm: 可选归一化 callable；其参数、buffer 和训练状态由调用方拥有
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _lava_cuba_lif_charge-en:

    * **English**

    Run the quantized charge path of ``lava_exchange.CubaLIFNode``: update the
    current through ``LeakyIntegratorStep``, optionally call the already selected
    ``norm`` callable, then update the membrane voltage through the same
    quantized integrator. The function does not read ``MemoryModule`` memory,
    inspect ``training/eval``, or manage fire/reset.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param current_state: Current current-state tensor
    :type current_state: torch.Tensor
    :param voltage_state: Current voltage-state tensor
    :type voltage_state: torch.Tensor
    :param current_decay: Current-decay tensor owned by the caller module
    :type current_decay: torch.Tensor
    :param voltage_decay: Voltage-decay tensor owned by the caller module
    :type voltage_decay: torch.Tensor
    :param s_scale: Lava synaptic scale
    :type s_scale: float
    :param norm: Optional normalization callable; its parameters, buffers, and
        training state are owned by the caller
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    from ..lava_exchange import LeakyIntegratorStep, step_quantize

    current_next = LeakyIntegratorStep.apply(
        x,
        step_quantize(current_decay),
        current_state.contiguous(),
        s_scale,
    )
    if norm is not None:
        current_next = norm(current_next)
    voltage_next = LeakyIntegratorStep.apply(
        current_next,
        step_quantize(voltage_decay),
        voltage_state.contiguous(),
        s_scale,
    )
    return current_next, voltage_next


def lava_cuba_lif_single_step(
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
    norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lava_cuba_lif_single_step-cn>` | :ref:`English <lava_cuba_lif_single_step-en>`

    ----

    .. _lava_cuba_lif_single_step-cn:

    * **中文**

    执行 ``lava_exchange.CubaLIFNode`` 已选 Torch 路径的一次状态转移，返回脉冲、
    下一电流状态和 reset 后的下一电压状态。函数不物化 state，不读取 module，也不
    判断 ``training/eval``。

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
    :param norm: 可选归一化 callable；其生命周期由调用方管理
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(spike, current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _lava_cuba_lif_single_step-en:

    * **English**

    Run one state transition for the selected Torch path of
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
    :param norm: Optional normalization callable; its lifecycle is managed by the
        caller
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(spike, current_next, voltage_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    current_next, voltage_charged = lava_cuba_lif_charge(
        x,
        current_state,
        voltage_state,
        current_decay,
        voltage_decay,
        s_scale,
        norm,
    )
    spike = surrogate_function(voltage_charged - (v_threshold + v_threshold_eps))
    voltage_next = hard_reset(voltage_charged, spike, v_reset, detach_reset)
    return spike, current_next, voltage_next


def lava_cuba_lif_multi_step(
    x_seq: torch.Tensor,
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
    store_i_seq: bool = False,
    store_v_seq: bool = False,
    norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    r"""
    **API Language** - :ref:`中文 <lava_cuba_lif_multi_step-cn>` | :ref:`English <lava_cuba_lif_multi_step-en>`

    ----

    .. _lava_cuba_lif_multi_step-cn:

    * **中文**

    对 ``lava_exchange.CubaLIFNode`` 已选 Torch 路径执行多步状态转移。函数返回固定
    arity：脉冲序列、最终电流状态、最终电压状态、可选电流序列和可选电压序列。
    ``store_i_seq``/``store_v_seq`` 只控制是否返回辅助轨迹，不改变返回项数量。

    :param x_seq: 输入序列，时间维为第 0 维
    :type x_seq: torch.Tensor
    :param current_state: 初始电流状态张量
    :type current_state: torch.Tensor
    :param voltage_state: 初始电压状态张量
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
    :param store_i_seq: 是否返回每步电流状态序列
    :type store_i_seq: bool
    :param store_v_seq: 是否返回每步 reset 后电压状态序列
    :type store_v_seq: bool
    :param norm: 可选归一化 callable；其生命周期由调用方管理
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(spike_seq, current_next, voltage_next, current_seq, voltage_seq)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]

    ----

    .. _lava_cuba_lif_multi_step-en:

    * **English**

    Run multi-step state transitions for the selected Torch path of
    ``lava_exchange.CubaLIFNode``. The function returns a fixed arity: spike
    sequence, final current state, final voltage state, optional current
    sequence, and optional voltage sequence. ``store_i_seq``/``store_v_seq`` only
    control auxiliary traces and do not change the tuple arity.

    :param x_seq: Input sequence with time as dimension 0
    :type x_seq: torch.Tensor
    :param current_state: Initial current-state tensor
    :type current_state: torch.Tensor
    :param voltage_state: Initial voltage-state tensor
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
    :param store_i_seq: Whether to return the per-step current-state sequence
    :type store_i_seq: bool
    :param store_v_seq: Whether to return the per-step reset voltage-state sequence
    :type store_v_seq: bool
    :param norm: Optional normalization callable; its lifecycle is managed by the
        caller
    :type norm: Optional[Callable[[torch.Tensor], torch.Tensor]]
    :return: ``(spike_seq, current_next, voltage_next, current_seq, voltage_seq)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    """
    current = current_state
    voltage = voltage_state
    spikes = []
    current_trace = [] if store_i_seq else None
    voltage_trace = [] if store_v_seq else None
    for x in x_seq:
        spike, current, voltage = lava_cuba_lif_single_step(
            x,
            current,
            voltage,
            current_decay,
            voltage_decay,
            s_scale,
            v_threshold,
            v_threshold_eps,
            v_reset,
            surrogate_function,
            detach_reset,
            norm,
        )
        spikes.append(spike)
        if current_trace is not None:
            current_trace.append(current)
        if voltage_trace is not None:
            voltage_trace.append(voltage)

    return (
        torch.stack(spikes),
        current,
        voltage,
        torch.stack(current_trace) if current_trace is not None else None,
        torch.stack(voltage_trace) if voltage_trace is not None else None,
    )


def liaf_output(
    v: torch.Tensor,
    v_threshold: float,
    act: Callable[[torch.Tensor], torch.Tensor],
    threshold_related: bool,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <liaf_output-cn>` | :ref:`English <liaf_output-en>`

    ----

    .. _liaf_output-cn:

    * **中文**

    计算 LIAF 的模拟输出。``threshold_related=True`` 时返回
    ``act(v - v_threshold)``，否则返回 ``act(v)``。函数不读取 ``MemoryModule``
    memory，不负责 charge、spike fire/reset、``training/eval`` 或 backend dispatch。

    :param v: 当前膜电位张量
    :type v: torch.Tensor
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param act: 用于生成模拟输出的激活函数
    :type act: Callable[[torch.Tensor], torch.Tensor]
    :param threshold_related: 是否从膜电位中减去阈值后再调用 ``act``
    :type threshold_related: bool
    :return: LIAF 模拟输出
    :rtype: torch.Tensor

    ----

    .. _liaf_output-en:

    * **English**

    Compute LIAF analog output. When ``threshold_related=True``, return
    ``act(v - v_threshold)``; otherwise return ``act(v)``. The function does not
    read ``MemoryModule`` memory and does not manage charge, spike fire/reset,
    ``training/eval``, or backend dispatch.

    :param v: Current membrane voltage tensor
    :type v: torch.Tensor
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param act: Activation function used to produce analog output
    :type act: Callable[[torch.Tensor], torch.Tensor]
    :param threshold_related: Whether to subtract the threshold from ``v`` before
        calling ``act``
    :type threshold_related: bool
    :return: LIAF analog output
    :rtype: torch.Tensor
    """
    return act(v - v_threshold) if threshold_related else act(v)


def mpbn_fire(
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    surrogate_function: SurrogateFunction,
    normalize_residual: bool = False,
    gamma: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    sigma2: torch.Tensor | None = None,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <mpbn_fire-cn>` | :ref:`English <mpbn_fire-en>`

    ----

    .. _mpbn_fire-cn:

    * **中文**

    执行 MPBN 神经元的放电计算，并在 ``normalize_residual=True`` 时对未放电位置
    的残余膜电位执行 BN 反变换。调用方必须传入已经由 module 选择好的有效阈值
    ``v_threshold`` 和 BN 参数；本函数不判断 ``training/eval``、不更新 running
    stats，也不读取 ``MemoryModule`` memory。

    :param v: 当前膜电位，支持 2D ``[N, C]`` 或 4D ``[N, C, H, W]``
    :type v: torch.Tensor
    :param v_threshold: 每通道有效阈值，形状 ``[C]``
    :type v_threshold: torch.Tensor
    :param surrogate_function: 已选定替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param normalize_residual: 是否归一化未放电残余
    :type normalize_residual: bool
    :param gamma: BatchNorm 缩放参数
    :type gamma: torch.Tensor or None
    :param mu: BatchNorm 均值
    :type mu: torch.Tensor or None
    :param beta: BatchNorm 偏置
    :type beta: torch.Tensor or None
    :param sigma2: BatchNorm 方差
    :type sigma2: torch.Tensor or None
    :param eps: BatchNorm epsilon
    :type eps: float or None
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 ``v`` 不是 2D 或 4D 时抛出
    :raises ValueError: 当启用 residual normalization 但缺少 BN 参数时抛出

    ----

    .. _mpbn_fire-en:

    * **English**

    Run MPBN firing and, when ``normalize_residual=True``, apply the BN inverse
    transform to membrane residuals that did not fire. The caller must pass the
    effective ``v_threshold`` and BN parameters already selected by the module.
    This function does not inspect ``training/eval``, update running stats, or
    read ``MemoryModule`` memory.

    :param v: Current 2D ``[N, C]`` or 4D ``[N, C, H, W]`` membrane voltage
    :type v: torch.Tensor
    :param v_threshold: Effective per-channel threshold shaped ``[C]``
    :type v_threshold: torch.Tensor
    :param surrogate_function: Selected surrogate function
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param normalize_residual: Whether to normalize non-firing residuals
    :type normalize_residual: bool
    :param gamma: Batch-normalization scale
    :type gamma: torch.Tensor or None
    :param mu: Batch-normalization mean
    :type mu: torch.Tensor or None
    :param beta: Batch-normalization bias
    :type beta: torch.Tensor or None
    :param sigma2: Batch-normalization variance
    :type sigma2: torch.Tensor or None
    :param eps: Batch-normalization epsilon
    :type eps: float or None
    :return: ``(spike, v_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises NotImplementedError: If ``v`` is neither 2D nor 4D
    :raises ValueError: If residual normalization is enabled without BN parameters
    """
    if v.ndim == 2:
        view_shape = (1, v_threshold.shape[0])
    elif v.ndim == 4:
        view_shape = (1, v_threshold.shape[0], 1, 1)
    else:
        raise NotImplementedError(
            f"Only 2D and 4D tensors are supported, but got {v.ndim}D tensors."
        )

    diff = v - v_threshold.view(view_shape)
    spike = surrogate_function(diff)
    if not normalize_residual:
        return spike, v

    if gamma is None or mu is None or beta is None or sigma2 is None or eps is None:
        raise ValueError("BN parameters are required when normalize_residual=True.")

    mask = diff <= 0
    gamma_expanded = gamma.view(view_shape).expand_as(mask)
    mu_expanded = mu.view(view_shape).expand_as(mask)
    beta_expanded = beta.view(view_shape).expand_as(mask)
    sigma_expanded = torch.sqrt(sigma2 + eps).view(view_shape).expand_as(mask)
    normalized_residual = (v[mask] - mu_expanded[mask]) / sigma_expanded[
        mask
    ] * gamma_expanded[mask] + beta_expanded[mask]
    v_next = v.clone()
    v_next.masked_scatter_(mask, normalized_residual)
    return spike, v_next


def online_lif_charge(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_reset: Optional[float],
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <online_lif_charge-cn>` | :ref:`English <online_lif_charge-en>`

    ----

    .. _online_lif_charge-cn:

    * **中文**

    执行 OTTT/SLTT LIF training 路径使用的 charge：先 detach 上一膜电位 ``v``，
    再按 LIF charge 公式更新。函数不判断 ``training/eval``，也不负责 fire/reset 或
    backend dispatch。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param v: 当前膜电位张量；函数会在内部使用 ``v.detach()``
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 是否对输入项应用衰减
    :type decay_input: bool
    :param v_reset: reset 参照电压；``None`` 表示 soft-reset 路径
    :type v_reset: Optional[float]
    :return: charge 后的膜电位
    :rtype: torch.Tensor

    ----

    .. _online_lif_charge-en:

    * **English**

    Run the charge used by OTTT/SLTT LIF training paths: detach the previous
    membrane potential ``v`` first, then apply the LIF charge equation. The
    function does not inspect ``training/eval`` and does not manage fire/reset
    or backend dispatch.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param v: Current membrane voltage tensor; the function uses ``v.detach()``
        internally
    :type v: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param decay_input: Whether to decay the input term
    :type decay_input: bool
    :param v_reset: Reset reference voltage; ``None`` indicates the soft-reset
        path
    :type v_reset: Optional[float]
    :return: Membrane voltage after charge
    :rtype: torch.Tensor
    """
    return lif_charge(x, v.detach(), tau, decay_input, v_reset)


def ottt_trace_update(
    spike: torch.Tensor,
    trace: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <ottt_trace_update-cn>` | :ref:`English <ottt_trace_update-en>`

    ----

    .. _ottt_trace_update-cn:

    * **中文**

    更新 OTTT trace：``trace * (1 - 1 / tau) + spike``。函数在
    ``torch.no_grad()`` 下执行，保持现有 ``OTTTLIFNode.track_trace`` 语义。

    :param spike: 当前脉冲张量
    :type spike: torch.Tensor
    :param trace: 当前 trace 张量
    :type trace: torch.Tensor
    :param tau: trace 衰减时间常数
    :type tau: float
    :return: 更新后的 trace
    :rtype: torch.Tensor

    ----

    .. _ottt_trace_update-en:

    * **English**

    Update OTTT trace as ``trace * (1 - 1 / tau) + spike`` under
    ``torch.no_grad()``, matching existing ``OTTTLIFNode.track_trace`` semantics.

    :param spike: Current spike tensor
    :type spike: torch.Tensor
    :param trace: Current trace tensor
    :type trace: torch.Tensor
    :param tau: Trace decay time constant
    :type tau: float
    :return: Updated trace
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        return trace * (1.0 - 1.0 / tau) + spike


def _reset(
    v: torch.Tensor,
    spike: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
) -> torch.Tensor:
    if v_reset is None:
        return soft_reset(v, spike, v_threshold, detach_reset)
    return hard_reset(v, spike, v_reset, detach_reset)


def _canonicalize_inductor_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous()


def _inductor_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(tensor.shape),
        tensor.ndim,
        str(tensor.dtype),
        tensor.device.type,
        tensor.device.index,
        tensor.is_contiguous(),
        bool(tensor.requires_grad),
    )


def _inductor_runtime_cache_key(*tensors: torch.Tensor) -> tuple[Any, ...]:
    return tuple(_inductor_tensor_signature(tensor) for tensor in tensors)


def _surrogate_inductor_cache_key(
    surrogate_function: SurrogateFunction,
) -> tuple[Any, ...] | None:
    from spikingjelly.activation_based.neuron import inductor_cache

    return inductor_cache.surrogate_key(surrogate_function)


def _compile_inductor_graph(
    cache_key: tuple[Any, ...] | None, fn: Callable
) -> Callable:
    from spikingjelly.activation_based.neuron import inductor_cache

    return inductor_cache.compile_graph(cache_key, fn)


def _normalise_inductor_multi_step_output(
    out: tuple[torch.Tensor, ...],
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if store_v_seq:
        spike_seq, v_next, v_seq = out
        return spike_seq, v_next, v_seq
    spike_seq, v_next = out
    return spike_seq, v_next, None


def if_single_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <if_single_step-cn>` | :ref:`English <if_single_step-en>`

    ----

    .. _if_single_step-cn:

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

    .. _if_single_step-en:

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
    v_charged = if_charge(x, v)
    spike = neuron_fire(v_charged, v_threshold, surrogate_function)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def if_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <if_multi_step-cn>` | :ref:`English <if_multi_step-en>`

    ----

    .. _if_multi_step-cn:

    * **中文**

    执行 IF 多步状态转移，输入序列第 0 维为时间维。返回固定结构
    ``(spike_seq, v_next, v_seq_or_none)``；``store_v_seq=False`` 时第三项为
    ``None``。返回的 ``v_next`` 和 ``v_seq`` 不 alias 输入 ``v``。

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

    .. _if_multi_step-en:

    * **English**

    Run IF multi-step state transition with the time dimension at axis 0. The
    return structure is fixed as ``(spike_seq, v_next, v_seq_or_none)``; the third
    item is ``None`` when ``store_v_seq=False``. Returned ``v_next`` and ``v_seq``
    do not alias the input ``v``.

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
    spike_seq = []
    v_seq = []
    for t in range(x_seq.shape[0]):
        spike, v = if_single_step(
            x_seq[t], v, v_threshold, v_reset, surrogate_function, detach_reset
        )
        spike_seq.append(spike)
        if store_v_seq:
            v_seq.append(v)
    stacked_spike_seq = torch.stack(spike_seq)
    if store_v_seq:
        return stacked_spike_seq, v, torch.stack(v_seq)
    return stacked_spike_seq, v, None


def activation_aware_if_single_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_offset: torch.Tensor,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <activation_aware_if_single_step-cn>` | :ref:`English <activation_aware_if_single_step-en>`

    ----

    .. _activation_aware_if_single_step-cn:

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

    .. _activation_aware_if_single_step-en:

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


def activation_aware_if_multi_step(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_offset: torch.Tensor,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_v_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <activation_aware_if_multi_step-cn>` | :ref:`English <activation_aware_if_multi_step-en>`

    ----

    .. _activation_aware_if_multi_step-cn:

    * **中文**

    执行一条已确定 Torch 路径上的 activation-aware IF 多步状态转移。函数逐时间步
    调用 :func:`activation_aware_if_single_step`，返回
    ``(spike_seq, v_next, v_seq_or_none)``。``v_threshold`` 和 ``v_offset`` 必须已
    按单个时间步输入 ``x_seq[0]`` 广播完成。

    函数不读取或写入 ``MemoryModule`` memory，不负责 ``training/eval``、
    ``step_mode``、backend dispatch、输入合法性检查或 channel-wise 参数广播。

    :param x_seq: 输入序列，shape 为 ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state，shape 与 ``x_seq[0]`` 相同
    :type v: torch.Tensor
    :param v_threshold: 已广播的发放阈值
    :type v_threshold: torch.Tensor
    :param v_offset: 已广播的膜电位偏移
    :type v_offset: torch.Tensor
    :param v_reset: 硬复位电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_v_seq: 是否返回每个时间步 reset 后的膜电位序列
    :type store_v_seq: bool
    :return: ``(spike_seq, v_next, v_seq_or_none)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _activation_aware_if_multi_step-en:

    * **English**

    Run an activation-aware IF multi-step state transition on an already selected
    Torch path. The function calls
    :func:`activation_aware_if_single_step` over time and returns
    ``(spike_seq, v_next, v_seq_or_none)``. ``v_threshold`` and ``v_offset`` must
    already be broadcast for a single time-step input ``x_seq[0]``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``training/eval``, ``step_mode``, backend dispatch, input validation,
    or channel-wise parameter broadcasting.

    :param x_seq: Input sequence shaped ``[T, N, *]``
    :type x_seq: torch.Tensor
    :param v: Materialized initial membrane-voltage tensor state with the same
        shape as ``x_seq[0]``
    :type v: torch.Tensor
    :param v_threshold: Broadcast threshold
    :type v_threshold: torch.Tensor
    :param v_offset: Broadcast membrane offset
    :type v_offset: torch.Tensor
    :param v_reset: Hard-reset voltage; ``None`` means soft reset
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
    spike_seq = []
    v_seq = []
    for x in x_seq:
        spike, v = activation_aware_if_single_step(
            x,
            v,
            v_threshold,
            v_offset,
            v_reset,
            surrogate_function,
            detach_reset,
        )
        spike_seq.append(spike)
        if store_v_seq:
            v_seq.append(v)
    stacked_spike_seq = torch.stack(spike_seq)
    if store_v_seq:
        return stacked_spike_seq, v, torch.stack(v_seq)
    return stacked_spike_seq, v, None


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

    x_seq = _canonicalize_inductor_tensor(x_seq)
    v = _canonicalize_inductor_tensor(v)
    surrogate_key = _surrogate_inductor_cache_key(surrogate_function)
    graph = _compile_inductor_graph(
        None
        if surrogate_key is None
        else (
            "functional_if",
            store_v_seq,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            _inductor_runtime_cache_key(x_seq, v),
        ),
        inductor_cache._build_if_multi_step_graph(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalise_inductor_multi_step_output(graph(x_seq, v), store_v_seq)


def lif_single_step(
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
    **API Language** - :ref:`中文 <lif_single_step-cn>` | :ref:`English <lif_single_step-en>`

    ----

    .. _lif_single_step-cn:

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

    .. _lif_single_step-en:

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
    v_charged = lif_charge(x, v, tau, decay_input, v_reset)
    spike = neuron_fire(v_charged, v_threshold, surrogate_function)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def lif_multi_step(
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
    **API Language** - :ref:`中文 <lif_multi_step-cn>` | :ref:`English <lif_multi_step-en>`

    ----

    .. _lif_multi_step-cn:

    * **中文**

    执行 LIF 多步状态转移，返回固定结构
    ``(spike_seq, v_next, v_seq_or_none)``。``store_v_seq=False`` 时第三项为
    ``None``。

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

    .. _lif_multi_step-en:

    * **English**

    Run LIF multi-step state transition and return the fixed structure
    ``(spike_seq, v_next, v_seq_or_none)``. The third item is ``None`` when
    ``store_v_seq=False``.

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
    spike_seq = []
    v_seq = []
    for t in range(x_seq.shape[0]):
        spike, v = lif_single_step(
            x_seq[t],
            v,
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
        )
        spike_seq.append(spike)
        if store_v_seq:
            v_seq.append(v)
    stacked_spike_seq = torch.stack(spike_seq)
    if store_v_seq:
        return stacked_spike_seq, v, torch.stack(v_seq)
    return stacked_spike_seq, v, None


def lif_single_step_with_pre_spike_mean(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lif_single_step_with_pre_spike_mean-cn>` | :ref:`English <lif_single_step_with_pre_spike_mean-en>`

    ----

    .. _lif_single_step_with_pre_spike_mean-cn:

    * **中文**

    执行 LIF 单步状态转移，并额外返回 charge 后、fire 前的
    ``(v_charged - v_threshold).mean()``。该函数覆盖
    ``spike_dhs.save_v_LIFNode`` 的观测路径；不读取 module memory，不判断
    ``training/eval``。

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
    :param v_reset: reset 电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :return: ``(spike, v_next, pre_spike_mean)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _lif_single_step_with_pre_spike_mean-en:

    * **English**

    Run one LIF state transition and additionally return
    ``(v_charged - v_threshold).mean()`` between charge and fire. This function
    covers the observation path of ``spike_dhs.save_v_LIFNode``; it does not read
    module memory or inspect ``training/eval``.

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
    :return: ``(spike, v_next, pre_spike_mean)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    v_charged = lif_charge(x, v, tau, decay_input, v_reset)
    pre_spike_mean = (v_charged - v_threshold).mean()
    spike = neuron_fire(v_charged, v_threshold, surrogate_function)
    return (
        spike,
        _reset(v_charged, spike, v_threshold, v_reset, detach_reset),
        pre_spike_mean,
    )


def lif_multi_step_with_pre_spike_mean(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: SurrogateFunction,
    detach_reset: bool = False,
    store_pre_spike_mean_seq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <lif_multi_step_with_pre_spike_mean-cn>` | :ref:`English <lif_multi_step_with_pre_spike_mean-en>`

    ----

    .. _lif_multi_step_with_pre_spike_mean-cn:

    * **中文**

    执行 LIF 多步状态转移，并可选返回每步 charge 后、fire 前的
    ``(v_charged - v_threshold).mean()`` 序列。返回 arity 固定为
    ``(spike_seq, v_next, pre_spike_mean_seq_or_none)``。

    :param x_seq: 输入序列张量，时间维为第 0 维
    :type x_seq: torch.Tensor
    :param v: 已物化的初始膜电位 tensor state
    :type v: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param decay_input: 输入是否参与衰减
    :type decay_input: bool
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: reset 电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param surrogate_function: 当前执行路径使用的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param store_pre_spike_mean_seq: 是否返回每步 pre-spike mean 序列
    :type store_pre_spike_mean_seq: bool
    :return: ``(spike_seq, v_next, pre_spike_mean_seq_or_none)``
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

    ----

    .. _lif_multi_step_with_pre_spike_mean-en:

    * **English**

    Run multi-step LIF transitions and optionally return the per-step
    ``(v_charged - v_threshold).mean()`` sequence between charge and fire. The
    return arity is fixed as
    ``(spike_seq, v_next, pre_spike_mean_seq_or_none)``.

    :param x_seq: Input sequence tensor with time as dimension 0
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
    :param store_pre_spike_mean_seq: Whether to return the per-step pre-spike
        mean sequence
    :type store_pre_spike_mean_seq: bool
    :return: ``(spike_seq, v_next, pre_spike_mean_seq_or_none)``
    :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    """
    spike_seq = []
    pre_spike_mean_seq = [] if store_pre_spike_mean_seq else None
    for x in x_seq:
        spike, v, pre_spike_mean = lif_single_step_with_pre_spike_mean(
            x,
            v,
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
        )
        spike_seq.append(spike)
        if pre_spike_mean_seq is not None:
            pre_spike_mean_seq.append(pre_spike_mean)
    return (
        torch.stack(spike_seq),
        v,
        torch.stack(pre_spike_mean_seq) if pre_spike_mean_seq is not None else None,
    )


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

    x_seq = _canonicalize_inductor_tensor(x_seq)
    v = _canonicalize_inductor_tensor(v)
    surrogate_key = _surrogate_inductor_cache_key(surrogate_function)
    graph = _compile_inductor_graph(
        None
        if surrogate_key is None
        else (
            "functional_lif",
            store_v_seq,
            decay_input,
            tau,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            _inductor_runtime_cache_key(x_seq, v),
        ),
        inductor_cache._build_lif_multi_step_graph(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalise_inductor_multi_step_output(graph(x_seq, v), store_v_seq)


def plif_single_step(
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
    **API Language** - :ref:`中文 <plif_single_step-cn>` | :ref:`English <plif_single_step-en>`

    ----

    .. _plif_single_step-cn:

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

    .. _plif_single_step-en:

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
    v_charged = plif_charge(x, v, w, decay_input, v_reset)
    spike = neuron_fire(v_charged, v_threshold, surrogate_function)
    return spike, _reset(v_charged, spike, v_threshold, v_reset, detach_reset)


def plif_multi_step(
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
    **API Language** - :ref:`中文 <plif_multi_step-cn>` | :ref:`English <plif_multi_step-en>`

    ----

    .. _plif_multi_step-cn:

    * **中文**

    执行 PLIF 多步状态转移，返回固定结构
    ``(spike_seq, v_next, v_seq_or_none)``。``store_v_seq=False`` 时第三项为
    ``None``。

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

    .. _plif_multi_step-en:

    * **English**

    Run PLIF multi-step state transition and return the fixed structure
    ``(spike_seq, v_next, v_seq_or_none)``. The third item is ``None`` when
    ``store_v_seq=False``.

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
    spike_seq = []
    v_seq = []
    for t in range(x_seq.shape[0]):
        spike, v = plif_single_step(
            x_seq[t],
            v,
            w,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
        )
        spike_seq.append(spike)
        if store_v_seq:
            v_seq.append(v)
    stacked_spike_seq = torch.stack(spike_seq)
    if store_v_seq:
        return stacked_spike_seq, v, torch.stack(v_seq)
    return stacked_spike_seq, v, None


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

    x_seq = _canonicalize_inductor_tensor(x_seq)
    v = _canonicalize_inductor_tensor(v)
    reciprocal_tau = _canonicalize_inductor_tensor(w.sigmoid().to(x_seq))
    surrogate_key = _surrogate_inductor_cache_key(surrogate_function)
    graph = _compile_inductor_graph(
        None
        if surrogate_key is None
        else (
            "functional_plif",
            store_v_seq,
            decay_input,
            v_threshold,
            v_reset,
            detach_reset,
            surrogate_key,
            _inductor_runtime_cache_key(x_seq, v, reciprocal_tau),
        ),
        inductor_cache._build_plif_multi_step_graph(
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            store_v_seq,
        ),
    )
    return _normalise_inductor_multi_step_output(
        graph(x_seq, v, reciprocal_tau), store_v_seq
    )


def if_single_step_cupy(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    forward_kernel: Any,
    backward_kernel: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <if_single_step_cupy-cn>` | :ref:`English <if_single_step_cupy-en>`

    ----

    .. _if_single_step_cupy-cn:

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

    .. _if_single_step_cupy-en:

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


def lif_single_step_cupy(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    forward_kernel: Any,
    backward_kernel: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <lif_single_step_cupy-cn>` | :ref:`English <lif_single_step_cupy-en>`

    ----

    .. _lif_single_step_cupy-cn:

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

    .. _lif_single_step_cupy-en:

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


def masked_psn_advance_queue(
    x: torch.Tensor,
    queue: tuple[torch.Tensor, ...],
    k: int,
) -> tuple[torch.Tensor, ...]:
    r"""
    **API Language** - :ref:`中文 <masked_psn_advance_queue-cn>` | :ref:`English <masked_psn_advance_queue-en>`

    ----

    .. _masked_psn_advance_queue-cn:

    * **中文**

    执行 ``MaskedPSN`` 单步路径中的 queue 推进：追加 ``x.flatten()``，当长度超过
    ``k`` 时弹出最旧元素。函数不原地修改传入 queue。该步骤与 overflow 检查分离，
    以允许 module 保持“先更新 queue，再抛 ``OverflowError``”的既有异常副作用。

    :param x: 当前输入张量
    :type x: torch.Tensor
    :param queue: 旧 queue state，元素按旧到新排列
    :type queue: tuple[torch.Tensor, ...]
    :param k: queue 最大长度
    :type k: int
    :return: 推进后的 queue state
    :rtype: tuple[torch.Tensor, ...]

    ----

    .. _masked_psn_advance_queue-en:

    * **English**

    Advance the queue in the ``MaskedPSN`` single-step path: append
    ``x.flatten()`` and pop the oldest item when the length exceeds ``k``. The
    function does not mutate the input queue. This step is separated from the
    overflow check so the module can preserve its existing side effect of
    updating the queue before raising ``OverflowError``.

    :param x: Current input tensor
    :type x: torch.Tensor
    :param queue: Previous queue state ordered from oldest to newest
    :type queue: tuple[torch.Tensor, ...]
    :param k: Maximum queue length
    :type k: int
    :return: Advanced queue state
    :rtype: tuple[torch.Tensor, ...]
    """
    queue_next = (*queue, x.flatten())
    if len(queue_next) > k:
        queue_next = queue_next[1:]
    return queue_next


def masked_psn_single_step_from_queue(
    x_shape: torch.Size,
    queue: tuple[torch.Tensor, ...],
    time_step: int,
    T: int,
    lambda_: torch.Tensor,
    mask0: torch.Tensor,
    mask1: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, int]:
    r"""
    **API Language** - :ref:`中文 <masked_psn_single_step_from_queue-cn>` | :ref:`English <masked_psn_single_step_from_queue-en>`

    ----

    .. _masked_psn_single_step_from_queue-cn:

    * **中文**

    使用已经推进后的 ``MaskedPSN`` queue 和当前 ``time_step`` 计算单步脉冲，并返回
    更新后的时间索引。函数不修改 queue，不读取 module memory，不判断
    ``training/eval``。若 ``time_step + 1 > T``，按既有 module 语义抛
    ``OverflowError``。

    :param x_shape: 当前输入的原始 shape，用于恢复输出 shape
    :type x_shape: torch.Size
    :param queue: 已推进后的 queue state
    :type queue: tuple[torch.Tensor, ...]
    :param time_step: 当前 Python 时间索引
    :type time_step: int
    :param T: 最大时间步数
    :type T: int
    :param lambda_: progressive mask 系数
    :type lambda_: torch.Tensor
    :param mask0: 局部 mask
    :type mask0: torch.Tensor
    :param mask1: 全连接 mask
    :type mask1: torch.Tensor
    :param weight: ``MaskedPSN`` 权重，shape 为 ``[T, T]``
    :type weight: torch.Tensor
    :param bias: ``MaskedPSN`` bias，shape 为 ``[T, 1]``
    :type bias: torch.Tensor
    :param surrogate_function: 作用于膜电位的替代函数
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, time_step_next)``
    :rtype: tuple[torch.Tensor, int]
    :raises OverflowError: 当 ``time_step + 1 > T`` 时抛出

    ----

    .. _masked_psn_single_step_from_queue-en:

    * **English**

    Compute one ``MaskedPSN`` spike from an already advanced queue and the current
    ``time_step``, then return the updated time index. The function does not
    mutate the queue, read module memory, or inspect ``training/eval``. It raises
    ``OverflowError`` when ``time_step + 1 > T``, matching the existing module
    semantics.

    :param x_shape: Original current-input shape used to restore output shape
    :type x_shape: torch.Size
    :param queue: Already advanced queue state
    :type queue: tuple[torch.Tensor, ...]
    :param time_step: Current Python time index
    :type time_step: int
    :param T: Maximum number of time steps
    :type T: int
    :param lambda_: Progressive-mask coefficient
    :type lambda_: torch.Tensor
    :param mask0: Local mask
    :type mask0: torch.Tensor
    :param mask1: Dense mask
    :type mask1: torch.Tensor
    :param weight: ``MaskedPSN`` weight shaped ``[T, T]``
    :type weight: torch.Tensor
    :param bias: ``MaskedPSN`` bias shaped ``[T, 1]``
    :type bias: torch.Tensor
    :param surrogate_function: Surrogate function applied to membrane potential
    :type surrogate_function: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(spike, time_step_next)``
    :rtype: tuple[torch.Tensor, int]
    :raises OverflowError: Raised when ``time_step + 1 > T``
    """
    if time_step + 1 > T:
        raise OverflowError(f"The MaskedPSN(T={T}) has run {time_step + 1} time-steps!")

    if lambda_ >= 1.0:
        masked_weight = weight * mask0
    else:
        masked_weight = (lambda_ * mask0 + (1.0 - lambda_) * mask1) * weight

    weight_step = masked_weight[
        time_step,
        time_step + 1 - len(queue) : time_step + 1,
    ]
    x_seq = torch.stack(queue)
    for _ in range(len(x_shape)):
        weight_step = weight_step.unsqueeze(-1)
    h = torch.sum(weight_step * x_seq, 0)
    spike = surrogate_function(h + bias[time_step])
    return spike.view(x_shape), time_step + 1


def sliding_psn_single_step(
    x: torch.Tensor,
    queue: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    surrogate_function: SurrogateFunction,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    r"""
    **API Language** - :ref:`中文 <sliding_psn_single_step-cn>` | :ref:`English <sliding_psn_single_step-en>`

    ----

    .. _sliding_psn_single_step-cn:

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

    .. _sliding_psn_single_step-en:

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


def stbif_single_step(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    r"""
    **API Language** - :ref:`中文 <stbif_single_step-cn>` | :ref:`English <stbif_single_step-en>`

    ----

    .. _stbif_single_step-cn:

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

    .. _stbif_single_step-en:

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
    normalized = x / q_threshold
    q_next = q + normalized.detach()
    acc_q_next = torch.round(acc_q)
    spike_position = (q_next - 1 >= 0) & (acc_q_next < pos_max)
    neg_spike_position = (q_next < 0) & (acc_q_next > neg_min)
    cur_output_next = spike_position.to(x.dtype) - neg_spike_position.to(x.dtype)
    acc_q_next = acc_q_next + cur_output_next
    q_next = torch.where(spike_position, q_next - 1, q_next)
    q_next = torch.where(neg_spike_position, q_next + 1, q_next)
    is_work = bool((normalized != 0).any() | (cur_output_next != 0).any())
    return cur_output_next * q_threshold, q_next, acc_q_next, cur_output_next, is_work


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
        (
            out_seq[t],
            q_next,
            acc_q_next,
            cur_output_next,
            _,
        ) = stbif_single_step(
            x_seq[t], q_next, acc_q_next, q_threshold, pos_max, neg_min
        )
    is_work = bool((x_seq != 0).any() | (out_seq != 0).any())
    return out_seq, q_next, acc_q_next, cur_output_next, is_work
