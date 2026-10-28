from __future__ import annotations

import math
from collections.abc import Callable

import torch


__all__ = [
    "stdp_linear_step",
    "mstdp_linear_step",
    "mstdpet_linear_step",
    "stdp_conv1d_step",
    "stdp_conv2d_step",
    "mstdpet_reward_step",
]


def _check_time_constants(
    *, exponential: bool = False, **time_constants: float
) -> None:
    for name, tau in time_constants.items():
        if not math.isfinite(tau) or tau <= 0 or (not exponential and tau < 1):
            bound = "> 0" if exponential else ">= 1"
            raise ValueError(f"{name} must be finite and {bound} time steps.")


def _check_inputs(in_spike, out_spike, trace, weight, ndim: int) -> None:
    if in_spike.ndim != ndim or out_spike.ndim != ndim:
        raise ValueError(f"Spikes must both have {ndim} dimensions.")
    if weight.ndim != (2 if ndim <= 2 else ndim):
        raise ValueError("Weight dimensions do not match the spike layout.")
    channel_dim = 0 if ndim == 1 else 1
    if (
        in_spike.shape[channel_dim] != weight.shape[1]
        or out_spike.shape[channel_dim] != weight.shape[0]
        or (ndim > 1 and in_spike.shape[0] != out_spike.shape[0])
    ):
        raise ValueError("Spike shapes do not match the weight or batch dimensions.")
    for name, spike, state in zip(("pre", "post"), (in_spike, out_spike), trace):
        if state.shape != spike.shape:
            raise ValueError(
                f"{name} trace shape differs from spikes; reset before changing batch size."
            )
        if spike.device != weight.device or state.device != weight.device:
            raise ValueError("Spikes, traces and weight must be on the same device.")


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _sum_spatial_contributions(x: torch.Tensor, dim: list[int]) -> torch.Tensor:
    # Keep large potentiation/depression totals finite until they are combined.
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else None
    return x.sum(dim=dim, dtype=dtype)


def stdp_linear_step(
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    *,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_linear_step-cn>` | :ref:`English <functional_stdp_linear_step-en>`

    ----

    .. _functional_stdp_linear_step-cn:

    * **中文**

    时间常数单位为模拟时间步，须为有限值且 >= 1。活动及 trace 的形状必须精确匹配，不允许状态通过广播适配 batch。建议活动、状态及权重使用相同浮点
    dtype，device 必须相同。使用 Euler 衰减 1-1/tau，更新按 batch 和空间位置求和。

    执行全连接权重的单步 STDP 更新。``trace`` 是
    ``(trace_pre, trace_post)``；函数先更新两个 trace，再用更新后的 trace
    计算权重增量，返回 ``(delta_w, trace_next)``。输入状态不会被原地修改。

    .. math::

       tr_{pre}^{t+1} &= tr_{pre}^{t} - tr_{pre}^{t} / \tau_{pre} + s_{pre}^{t} \\
       tr_{post}^{t+1} &= tr_{post}^{t} - tr_{post}^{t} / \tau_{post} + s_{post}^{t}

    :param in_spike: 输入脉冲，形状 ``[N, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, out_features]``
    :type out_spike: torch.Tensor
    :param trace: 当前 ``(trace_pre, trace_post)``，两者分别与 ``in_spike`` 和
        ``out_spike`` 同形状、同 device，且 dtype 可参与对应计算
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: pre-synaptic trace 时间常数
    :type tau_pre: float
    :param tau_post: post-synaptic trace 时间常数
    :type tau_post: float
    :param f_pre: 作用于 pre 分支权重的调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: 作用于 post 分支权重的调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``；``delta_w`` 与
        ``weight`` 同形状
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    :raises ValueError: 时间常数或张量形状/device 不合法。

    ----

    .. _functional_stdp_linear_step-en:

    * **English**

    Time constants are in simulated time steps, finite and >= 1. Activity and trace
    shapes must match exactly; state broadcasting cannot adapt batch size. Matching
    floating dtypes are recommended for activity, state and weight; devices must
    match. Uses Euler decay 1-1/tau and sums updates over batch and spatial positions.

    Run one STDP update for a linear weight. ``trace`` is
    ``(trace_pre, trace_post)``. The function updates both traces first, computes
    the weight increment from the updated traces, and returns
    ``(delta_w, trace_next)``. It does not mutate the input state in place.

    .. math::

       tr_{pre}^{t+1} &= tr_{pre}^{t} - tr_{pre}^{t} / \tau_{pre} + s_{pre}^{t} \\
       tr_{post}^{t+1} &= tr_{post}^{t} - tr_{post}^{t} / \tau_{post} + s_{post}^{t}

    :param in_spike: Input spikes shaped ``[N, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, out_features]``
    :type out_spike: torch.Tensor
    :param trace: Current ``(trace_pre, trace_post)``. The tensors have the same
        shapes and devices as ``in_spike`` and ``out_spike``, respectively, and
        dtypes compatible with the corresponding computations
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: Weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: Time constant of the pre-synaptic trace
    :type tau_pre: float
    :param tau_post: Time constant of the post-synaptic trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``; ``delta_w`` has
        the same shape as ``weight``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    :raises ValueError: Invalid time constants or tensor shapes/devices.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_pre=tau_pre, tau_post=tau_post, exponential=False)
    _check_inputs(in_spike, out_spike, trace, weight, 2)
    trace_pre, trace_post = trace
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w_pre = -f_pre(weight) * (
        trace_post.unsqueeze(2) * in_spike.unsqueeze(1)
    ).sum(0)
    delta_w_post = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ).sum(0)
    return delta_w_pre + delta_w_post, (trace_pre, trace_post)


def mstdp_linear_step(
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    *,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdp_linear_step-cn>` | :ref:`English <functional_mstdp_linear_step-en>`

    ----

    .. _functional_mstdp_linear_step-cn:

    * **中文**

    时间常数单位为模拟时间步，须为有限值且 > 0。活动及 trace 的形状必须精确匹配，不允许状态通过广播适配 batch。建议活动、状态及权重使用相同浮点
    dtype，device 必须相同。使用指数衰减 exp(-1/tau)，保留原有离散定义。

    执行全连接权重的单步 mSTDP eligibility 计算。``trace`` 是
    ``(trace_pre, trace_post)``。返回的 eligibility 保留 batch 维，供调用者
    进一步施加 reward；本函数不接收或处理 reward。

    :param in_spike: 输入脉冲，形状 ``[N, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, out_features]``
    :type out_spike: torch.Tensor
    :param trace: 当前 ``(trace_pre, trace_post)``，两者分别与 ``in_spike`` 和
        ``out_spike`` 同形状、同 device
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: pre-synaptic trace 时间常数
    :type tau_pre: float
    :param tau_post: post-synaptic trace 时间常数
    :type tau_post: float
    :param f_pre: 作用于 pre 分支权重的调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: 作用于 post 分支权重的调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, (trace_pre_next, trace_post_next))``；
        ``eligibility`` 形状为 ``[N, out_features, in_features]``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    :raises ValueError: 时间常数或张量形状/device 不合法。

    ----

    .. _functional_mstdp_linear_step-en:

    * **English**

    Time constants are in simulated time steps, finite and > 0. Activity and trace
    shapes must match exactly; state broadcasting cannot adapt batch size. Matching
    floating dtypes are recommended for activity, state and weight; devices must
    match. Uses exponential decay exp(-1/tau), preserving the existing discretization.

    Compute one mSTDP eligibility step for a linear weight. ``trace`` is
    ``(trace_pre, trace_post)``. The returned eligibility retains its batch
    dimension for subsequent reward modulation; this function neither receives
    nor applies a reward.

    :param in_spike: Input spikes shaped ``[N, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, out_features]``
    :type out_spike: torch.Tensor
    :param trace: Current ``(trace_pre, trace_post)`` with the same shapes and
        devices as ``in_spike`` and ``out_spike``, respectively
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: Weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: Time constant of the pre-synaptic trace
    :type tau_pre: float
    :param tau_post: Time constant of the post-synaptic trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, (trace_pre_next, trace_post_next))``;
        ``eligibility`` is shaped ``[N, out_features, in_features]``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    :raises ValueError: Invalid time constants or tensor shapes/devices.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_pre=tau_pre, tau_post=tau_post, exponential=True)
    _check_inputs(in_spike, out_spike, trace, weight, 2)
    trace_pre, trace_post = trace
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ) - f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1))
    return eligibility, (trace_pre, trace_post)


def mstdpet_linear_step(
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    *,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdpet_linear_step-cn>` | :ref:`English <functional_mstdpet_linear_step-en>`

    ----

    .. _functional_mstdpet_linear_step-cn:

    * **中文**

    时间常数单位为模拟时间步，须为有限值且 > 0。活动及 trace 的形状必须精确匹配，不允许状态通过广播适配 batch。建议活动、状态及权重使用相同浮点
    dtype，device 必须相同。使用指数衰减 exp(-1/tau)，保留原有离散定义。

    执行无 batch 维全连接脉冲的单步 mSTDP-ET eligibility 计算。``trace`` 是
    ``(trace_pre, trace_post)``。本函数只更新神经元 trace 并计算 eligibility；
    eligibility trace 的衰减与 reward 调制由 :func:`mstdpet_reward_step` 完成。

    :param in_spike: 输入脉冲，形状 ``[in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[out_features]``
    :type out_spike: torch.Tensor
    :param trace: 当前 ``(trace_pre, trace_post)``，两者分别与 ``in_spike`` 和
        ``out_spike`` 同形状、同 device
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: pre-synaptic trace 时间常数
    :type tau_pre: float
    :param tau_post: post-synaptic trace 时间常数
    :type tau_post: float
    :param f_pre: 作用于 pre 分支权重的调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: 作用于 post 分支权重的调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, (trace_pre_next, trace_post_next))``；
        ``eligibility`` 与 ``weight`` 同形状
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    :raises ValueError: 时间常数或张量形状/device 不合法。

    ----

    .. _functional_mstdpet_linear_step-en:

    * **English**

    Time constants are in simulated time steps, finite and > 0. Activity and trace
    shapes must match exactly; state broadcasting cannot adapt batch size. Matching
    floating dtypes are recommended for activity, state and weight; devices must
    match. Uses exponential decay exp(-1/tau), preserving the existing discretization.

    Compute one mSTDP-ET eligibility step for unbatched linear spikes. ``trace``
    is ``(trace_pre, trace_post)``. This function only updates the neuronal
    traces and computes eligibility. :func:`mstdpet_reward_step` handles
    eligibility-trace decay and reward modulation.

    :param in_spike: Input spikes shaped ``[in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[out_features]``
    :type out_spike: torch.Tensor
    :param trace: Current ``(trace_pre, trace_post)`` with the same shapes and
        devices as ``in_spike`` and ``out_spike``, respectively
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: Weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param tau_pre: Time constant of the pre-synaptic trace
    :type tau_pre: float
    :param tau_post: Time constant of the post-synaptic trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, (trace_pre_next, trace_post_next))``;
        ``eligibility`` has the same shape as ``weight``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    :raises ValueError: Invalid time constants or tensor shapes/devices.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_pre=tau_pre, tau_post=tau_post, exponential=True)
    _check_inputs(in_spike, out_spike, trace, weight, 1)
    trace_pre, trace_post = trace
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * torch.outer(out_spike, trace_pre) - f_pre(
        weight
    ) * torch.outer(trace_post, in_spike)
    return eligibility, (trace_pre, trace_post)


def stdp_conv2d_step(
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_conv2d_step-cn>` | :ref:`English <functional_stdp_conv2d_step-en>`

    ----

    .. _functional_stdp_conv2d_step-cn:

    * **中文**

    时间常数单位为模拟时间步，须为有限值且 >= 1。活动及 trace 的形状必须精确匹配，不允许状态通过广播适配 batch。建议活动、状态及权重使用相同浮点
    dtype，device 必须相同。使用 Euler 衰减 1-1/tau，更新按 batch 和空间位置求和。

    执行二维卷积权重的单步 STDP 更新。``trace`` 是
    ``(trace_pre, trace_post)``。``in_spike`` 必须已经按突触层的 padding
    规则展开；函数因此只表达 dilation 为 1、groups 为 1 的卷积 STDP 方程，
    不读取 ``Conv2d`` module 或解释 padding mode。

    FP16/BF16 贡献使用 FP32 求和，pre/post 更新合并后再转换为权重 dtype。
    trace 更新及其 dtype 提升规则不变；最终更新超出权重 dtype 范围时仍可能溢出。

    :param in_spike: 已 padding 的输入脉冲，形状 ``[N, C_in, H_pad, W_pad]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace: 当前 ``(trace_pre, trace_post)``，两者分别与 ``in_spike`` 和
        ``out_spike`` 同形状、同 device
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: 权重，形状 ``[C_out, C_in, K_h, K_w]``
    :type weight: torch.Tensor
    :param stride: 二维卷积步长 ``(stride_h, stride_w)``
    :type stride: Tuple[int, int]
    :param tau_pre: pre-synaptic trace 时间常数
    :type tau_pre: float
    :param tau_post: post-synaptic trace 时间常数
    :type tau_post: float
    :param f_pre: 作用于 pre 分支权重的调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: 作用于 post 分支权重的调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``；``delta_w`` 与
        ``weight`` 同形状
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    :raises ValueError: 时间常数或张量形状/device 不合法。

    ----

    .. _functional_stdp_conv2d_step-en:

    * **English**

    Time constants are in simulated time steps, finite and >= 1. Activity and trace
    shapes must match exactly; state broadcasting cannot adapt batch size. Matching
    floating dtypes are recommended for activity, state and weight; devices must
    match. Uses Euler decay 1-1/tau and sums updates over batch and spatial positions.

    Run one STDP update for a 2D convolution weight. ``trace`` is
    ``(trace_pre, trace_post)``. ``in_spike`` must already include the synaptic
    layer's padding. The function consequently expresses only the convolutional
    STDP equation for dilation 1 and one group; it neither reads a ``Conv2d``
    module nor interprets a padding mode.

    Spatial reductions of FP16/BF16 contributions use FP32; the pre/post updates
    are combined before conversion to the weight dtype. Trace-update arithmetic
    and dtype promotion are unchanged. A final update outside the weight dtype's
    range can still overflow.

    :param in_spike: Padded input spikes shaped ``[N, C_in, H_pad, W_pad]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace: Current ``(trace_pre, trace_post)`` with the same shapes and
        devices as ``in_spike`` and ``out_spike``, respectively
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: Weight shaped ``[C_out, C_in, K_h, K_w]``
    :type weight: torch.Tensor
    :param stride: Convolution stride ``(stride_h, stride_w)``
    :type stride: Tuple[int, int]
    :param tau_pre: Time constant of the pre-synaptic trace
    :type tau_pre: float
    :param tau_post: Time constant of the post-synaptic trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``; ``delta_w`` has
        the same shape as ``weight``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    :raises ValueError: Invalid time constants or tensor shapes/devices.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_pre=tau_pre, tau_post=tau_post, exponential=False)
    _check_inputs(in_spike, out_spike, trace, weight, 4)
    if any(value <= 0 for value in stride):
        raise ValueError("Stride must be positive.")
    expected = tuple(
        (size - kernel) // step + 1
        for size, kernel, step in zip(in_spike.shape[2:], weight.shape[2:], stride)
    )
    if out_spike.shape[2:] != expected:
        raise ValueError("Output spatial shape does not match the kernel and stride.")
    trace_pre, trace_post = trace
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(weight)
    stride_h, stride_w = stride
    for h in range(weight.shape[2]):
        for w in range(weight.shape[3]):
            h_end = in_spike.shape[2] - weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - weight.shape[3] + 1 + w
            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]
            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]
            weight_hw = weight[:, :, h, w]
            delta_w_pre = -f_pre(weight_hw) * _sum_spatial_contributions(
                (trace_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute(
                    [1, 2, 0, 3, 4]
                ),
                [2, 3, 4],
            )
            delta_w_post = f_post(weight_hw) * _sum_spatial_contributions(
                (tr_pre.unsqueeze(1) * out_spike.unsqueeze(2)).permute([1, 2, 0, 3, 4]),
                [2, 3, 4],
            )
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post
    return delta_w, (trace_pre, trace_post)


def stdp_conv1d_step(
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    *,
    stride: tuple[int],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_conv1d_step-cn>` | :ref:`English <functional_stdp_conv1d_step-en>`

    ----

    .. _functional_stdp_conv1d_step-cn:

    * **中文**

    时间常数单位为模拟时间步，须为有限值且 >= 1。活动及 trace 的形状必须精确匹配，不允许状态通过广播适配 batch。建议活动、状态及权重使用相同浮点
    dtype，device 必须相同。使用 Euler 衰减 1-1/tau，更新按 batch 和空间位置求和。

    执行一维卷积权重的单步 STDP 更新。``trace`` 是
    ``(trace_pre, trace_post)``。``in_spike`` 必须已经按突触层的 padding
    规则展开；函数因此只表达 dilation 为 1、groups 为 1 的卷积 STDP 方程，
    不读取 ``Conv1d`` module 或解释 padding mode。

    FP16/BF16 贡献使用 FP32 求和，pre/post 更新合并后再转换为权重 dtype。
    trace 更新及其 dtype 提升规则不变；最终更新超出权重 dtype 范围时仍可能溢出。

    :param in_spike: 已 padding 的输入脉冲，形状 ``[N, C_in, L_pad]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace: 当前 ``(trace_pre, trace_post)``，两者分别与 ``in_spike`` 和
        ``out_spike`` 同形状、同 device
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: 权重，形状 ``[C_out, C_in, K]``
    :type weight: torch.Tensor
    :param stride: 一维卷积步长 ``(stride,)``
    :type stride: Tuple[int]
    :param tau_pre: pre-synaptic trace 时间常数
    :type tau_pre: float
    :param tau_post: post-synaptic trace 时间常数
    :type tau_post: float
    :param f_pre: 作用于 pre 分支权重的调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: 作用于 post 分支权重的调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``；``delta_w`` 与
        ``weight`` 同形状
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    :raises ValueError: 时间常数或张量形状/device 不合法。

    ----

    .. _functional_stdp_conv1d_step-en:

    * **English**

    Time constants are in simulated time steps, finite and >= 1. Activity and trace
    shapes must match exactly; state broadcasting cannot adapt batch size. Matching
    floating dtypes are recommended for activity, state and weight; devices must
    match. Uses Euler decay 1-1/tau and sums updates over batch and spatial positions.

    Run one STDP update for a 1D convolution weight. ``trace`` is
    ``(trace_pre, trace_post)``. ``in_spike`` must already include the synaptic
    layer's padding. The function consequently expresses only the convolutional
    STDP equation for dilation 1 and one group; it neither reads a ``Conv1d``
    module nor interprets a padding mode.

    Spatial reductions of FP16/BF16 contributions use FP32; the pre/post updates
    are combined before conversion to the weight dtype. Trace-update arithmetic
    and dtype promotion are unchanged. A final update outside the weight dtype's
    range can still overflow.

    :param in_spike: Padded input spikes shaped ``[N, C_in, L_pad]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace: Current ``(trace_pre, trace_post)`` with the same shapes and
        devices as ``in_spike`` and ``out_spike``, respectively
    :type trace: Tuple[torch.Tensor, torch.Tensor]
    :param weight: Weight shaped ``[C_out, C_in, K]``
    :type weight: torch.Tensor
    :param stride: Convolution stride ``(stride,)``
    :type stride: Tuple[int]
    :param tau_pre: Time constant of the pre-synaptic trace
    :type tau_pre: float
    :param tau_post: Time constant of the post-synaptic trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, (trace_pre_next, trace_post_next))``; ``delta_w`` has
        the same shape as ``weight``
    :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    :raises ValueError: Invalid time constants or tensor shapes/devices.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_pre=tau_pre, tau_post=tau_post, exponential=False)
    _check_inputs(in_spike, out_spike, trace, weight, 3)
    if any(value <= 0 for value in stride):
        raise ValueError("Stride must be positive.")
    expected = tuple(
        (size - kernel) // step + 1
        for size, kernel, step in zip(in_spike.shape[2:], weight.shape[2:], stride)
    )
    if out_spike.shape[2:] != expected:
        raise ValueError("Output spatial shape does not match the kernel and stride.")
    trace_pre, trace_post = trace
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(weight)
    stride_l = stride[0]
    for l in range(weight.shape[2]):
        l_end = in_spike.shape[2] - weight.shape[2] + 1 + l
        pre_spike = in_spike[:, :, l:l_end:stride_l]
        tr_pre = trace_pre[:, :, l:l_end:stride_l]
        weight_l = weight[:, :, l]
        delta_w_pre = -f_pre(weight_l) * _sum_spatial_contributions(
            (trace_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute([1, 2, 0, 3]),
            [2, 3],
        )
        delta_w_post = f_post(weight_l) * _sum_spatial_contributions(
            (tr_pre.unsqueeze(1) * out_spike.unsqueeze(2)).permute([1, 2, 0, 3]),
            [2, 3],
        )
        delta_w[:, :, l] += delta_w_pre + delta_w_post
    return delta_w, (trace_pre, trace_post)


def mstdpet_reward_step(
    reward: torch.Tensor | float,
    eligibility: torch.Tensor,
    trace_e: torch.Tensor,
    *,
    tau_trace: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdpet_reward_step-cn>` | :ref:`English <functional_mstdpet_reward_step-en>`

    ----

    .. _functional_mstdpet_reward_step-cn:

    * **中文**

    reward 和 eligibility 按 PyTorch 规则与 trace_e 广播，dtype/device 遵循 PyTorch 运算规则。
    tau_trace 单位为时间步，须为有限正数。本函数只计算传入状态，
    不负责 Learner 的活动时间对齐或标量 reward 限制。

    更新 mSTDP-ET eligibility trace，并用 reward 调制更新后的 trace。

    .. math::

       tr_e^{t+1} &= tr_e^t \exp(-1 / \tau_{trace}) + e^t / \tau_{trace} \\
       \Delta W^t &= r^t tr_e^{t+1}

    :param reward: 标量或可与 ``trace_e`` 广播的 reward
    :type reward: torch.Tensor or float
    :param eligibility: 当前 eligibility，形状与 ``trace_e`` 相同或可广播
    :type eligibility: torch.Tensor
    :param trace_e: 当前 eligibility trace
    :type trace_e: torch.Tensor
    :param tau_trace: eligibility trace 时间常数
    :type tau_trace: float
    :return: ``(delta_w, trace_e_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    :raises ValueError: tau_trace 非有限或不为正。
    :raises RuntimeError: 张量广播或 dtype/device 不符合 PyTorch 运算要求。

    ----

    .. _functional_mstdpet_reward_step-en:

    * **English**

    Reward and eligibility broadcast with trace_e under PyTorch dtype/device rules.
    tau_trace is in time steps and must be finite and positive. This function computes
    supplied states; activity alignment and scalar-only rewards belong to the Learner.

    Update the mSTDP-ET eligibility trace and modulate the updated trace with the
    reward.

    .. math::

       tr_e^{t+1} &= tr_e^t \exp(-1 / \tau_{trace}) + e^t / \tau_{trace} \\
       \Delta W^t &= r^t tr_e^{t+1}

    :param reward: Scalar reward or a tensor broadcastable with ``trace_e``
    :type reward: torch.Tensor or float
    :param eligibility: Current eligibility with the same shape as ``trace_e`` or
        a broadcast-compatible shape
    :type eligibility: torch.Tensor
    :param trace_e: Current eligibility trace
    :type trace_e: torch.Tensor
    :param tau_trace: Time constant of the eligibility trace
    :type tau_trace: float
    :return: ``(delta_w, trace_e_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: tau_trace is nonfinite or not positive.
    :raises RuntimeError: Tensor broadcasting or dtype/device violates PyTorch rules.

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    _check_time_constants(tau_trace=tau_trace, exponential=True)
    trace_e = trace_e * math.exp(-1 / tau_trace) + eligibility / tau_trace
    return reward * trace_e, trace_e
