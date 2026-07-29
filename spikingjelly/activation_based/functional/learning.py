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


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


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

    ----

    .. _functional_stdp_linear_step-en:

    * **English**

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
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

    ----

    .. _functional_mstdp_linear_step-en:

    * **English**

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
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

    ----

    .. _functional_mstdpet_linear_step-en:

    * **English**

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
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

    执行二维卷积权重的单步 STDP 更新。``trace`` 是
    ``(trace_pre, trace_post)``。``in_spike`` 必须已经按突触层的 padding
    规则展开；函数因此只表达 dilation 为 1、groups 为 1 的卷积 STDP 方程，
    不读取 ``Conv2d`` module 或解释 padding mode。

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

    ----

    .. _functional_stdp_conv2d_step-en:

    * **English**

    Run one STDP update for a 2D convolution weight. ``trace`` is
    ``(trace_pre, trace_post)``. ``in_spike`` must already include the synaptic
    layer's padding. The function consequently expresses only the convolutional
    STDP equation for dilation 1 and one group; it neither reads a ``Conv2d``
    module nor interprets a padding mode.

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
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
            delta_w_pre = -(
                f_pre(weight_hw)
                * (trace_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                .permute([1, 2, 0, 3, 4])
                .sum(dim=[2, 3, 4])
            )
            delta_w_post = f_post(weight_hw) * (
                tr_pre.unsqueeze(1) * out_spike.unsqueeze(2)
            ).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4])
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

    执行一维卷积权重的单步 STDP 更新。``trace`` 是
    ``(trace_pre, trace_post)``。``in_spike`` 必须已经按突触层的 padding
    规则展开；函数因此只表达 dilation 为 1、groups 为 1 的卷积 STDP 方程，
    不读取 ``Conv1d`` module 或解释 padding mode。

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

    ----

    .. _functional_stdp_conv1d_step-en:

    * **English**

    Run one STDP update for a 1D convolution weight. ``trace`` is
    ``(trace_pre, trace_post)``. ``in_spike`` must already include the synaptic
    layer's padding. The function consequently expresses only the convolutional
    STDP equation for dilation 1 and one group; it neither reads a ``Conv1d``
    module nor interprets a padding mode.

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
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
        delta_w_pre = -(
            f_pre(weight_l)
            * (trace_post.unsqueeze(2) * pre_spike.unsqueeze(1))
            .permute([1, 2, 0, 3])
            .sum(dim=[2, 3])
        )
        delta_w_post = f_post(weight_l) * (
            tr_pre.unsqueeze(1) * out_spike.unsqueeze(2)
        ).permute([1, 2, 0, 3]).sum(dim=[2, 3])
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

    ----

    .. _functional_mstdpet_reward_step-en:

    * **English**

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

    .. note::

       本函数没有独立多步形式；多步执行由调用者逐步循环。
       This function has no independent multi-step form; callers iterate it.
    """
    trace_e = trace_e * math.exp(-1 / tau_trace) + eligibility / tau_trace
    return reward * trace_e, trace_e
