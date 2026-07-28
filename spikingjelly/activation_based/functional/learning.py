from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


__all__ = [
    "stdp_linear_single_step",
    "mstdp_linear_single_step",
    "mstdpet_linear_single_step",
    "stdp_conv1d_single_step",
    "stdp_conv2d_single_step",
    "mstdpet_reward_delta",
]


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def stdp_linear_single_step(
    weight: torch.Tensor,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_linear_single_step-cn>` | :ref:`English <functional_stdp_linear_single_step-en>`

    ----

    .. _functional_stdp_linear_single_step-cn:

    * **中文**

    对 linear 权重执行单步 STDP tensor 更新，返回 ``delta_w`` 和更新后的
    ``trace_pre``、``trace_post``。函数接收 raw ``weight``，不读取 module、
    monitor、``MemoryModule`` memory、``step_mode`` 或 ``training/eval``。

    :param weight: linear 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: 输入脉冲，形状 ``[batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: 已物化的 pre trace tensor
    :type trace_pre: torch.Tensor
    :param trace_post: 已物化的 post trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: pre trace 时间常数
    :type tau_pre: float
    :param tau_post: post trace 时间常数
    :type tau_post: float
    :param f_pre: pre 分支权重调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: post 分支权重调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _functional_stdp_linear_single_step-en:

    * **English**

    Run one tensor-only STDP update for a linear weight and return ``delta_w``
    followed by the updated ``trace_pre`` and ``trace_post``. The function
    receives raw ``weight`` and does not read modules, monitors, ``MemoryModule``
    memory, ``step_mode`` or ``training/eval``.

    :param weight: Linear weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: Input spikes shaped ``[batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Materialized pre-synaptic trace tensor
    :type trace_pre: torch.Tensor
    :param trace_post: Materialized post-synaptic trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w_pre = -f_pre(weight) * (
        trace_post.unsqueeze(2) * in_spike.unsqueeze(1)
    ).sum(0)
    delta_w_post = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ).sum(0)
    return delta_w_pre + delta_w_post, trace_pre, trace_post


def mstdp_linear_single_step(
    weight: torch.Tensor,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdp_linear_single_step-cn>` | :ref:`English <functional_mstdp_linear_single_step-en>`

    ----

    .. _functional_mstdp_linear_single_step-cn:

    * **中文**

    对 linear 权重执行单步 mSTDP eligibility tensor 计算，返回每样本 eligibility
    和更新后的 ``trace_pre``、``trace_post``。函数不读取 module、
    monitor、``MemoryModule`` memory、``step_mode`` 或 ``training/eval``。

    :param weight: linear 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: 输入脉冲，形状 ``[batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre trace
    :type trace_pre: torch.Tensor
    :param trace_post: post trace
    :type trace_post: torch.Tensor
    :param tau_pre: pre trace 时间常数
    :type tau_pre: float
    :param tau_post: post trace 时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _functional_mstdp_linear_single_step-en:

    * **English**

    Run one tensor-only mSTDP eligibility computation for a linear weight and
    return per-sample eligibility followed by the updated traces. The function
    does not read modules, monitors, ``MemoryModule`` memory, ``step_mode`` or
    ``training/eval``.

    :param weight: Linear weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: Input spikes shaped ``[batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Materialized pre-synaptic trace tensor
    :type trace_pre: torch.Tensor
    :param trace_post: Materialized post-synaptic trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ) - f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1))
    return eligibility, trace_pre, trace_post


def mstdpet_linear_single_step(
    weight: torch.Tensor,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdpet_linear_single_step-cn>` | :ref:`English <functional_mstdpet_linear_single_step-en>`

    ----

    .. _functional_mstdpet_linear_single_step-cn:

    * **中文**

    对 linear 权重执行单步 mSTDP-ET eligibility tensor 计算。该函数只计算
    eligibility；``trace_e`` 衰减和 reward 调制由
    :func:`mstdpet_reward_delta` 处理。

    :param weight: linear 权重，形状 ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: 输入脉冲，形状 ``[in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre trace
    :type trace_pre: torch.Tensor
    :param trace_post: post trace
    :type trace_post: torch.Tensor
    :param tau_pre: pre trace 时间常数
    :type tau_pre: float
    :param tau_post: post trace 时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _functional_mstdpet_linear_single_step-en:

    * **English**

    Run one tensor-only mSTDP-ET eligibility computation for a linear weight.
    This function only computes eligibility; ``trace_e`` decay and reward
    modulation are handled by :func:`mstdpet_reward_delta`.

    :param weight: Linear weight shaped ``[out_features, in_features]``
    :type weight: torch.Tensor
    :param in_spike: Input spikes shaped ``[in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Materialized pre-synaptic trace tensor
    :type trace_pre: torch.Tensor
    :param trace_post: Materialized post-synaptic trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(eligibility, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * torch.outer(out_spike, trace_pre) - f_pre(
        weight
    ) * torch.outer(trace_post, in_spike)
    return eligibility, trace_pre, trace_post


def stdp_conv2d_single_step(
    weight: torch.Tensor,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    stride: tuple[int, int],
    padding: tuple[int, int],
    padding_mode: str = "zeros",
    reversed_padding_repeated_twice: tuple[int, ...] | list[int] | None = None,
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_conv2d_single_step-cn>` | :ref:`English <functional_stdp_conv2d_single_step-en>`

    ----

    .. _functional_stdp_conv2d_single_step-cn:

    * **中文**

    对 Conv2d raw 参数执行单步 STDP tensor 更新。当前保持既有支持边界：只支持
    ``dilation == (1, 1)`` 且 ``groups == 1``。函数不读取 module、
    ``MemoryModule`` memory、``step_mode`` 或 ``training/eval``。

    :param weight: Conv2d 权重
    :type weight: torch.Tensor
    :param in_spike: 输入脉冲，形状 ``[N, C_in, H, W]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: 已物化的 pre trace tensor；存在 padding 时，其空间维必须与
        padding 后的 ``in_spike`` 一致
    :type trace_pre: torch.Tensor
    :param trace_post: 已物化的 post trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: pre trace 时间常数
    :type tau_pre: float
    :param tau_post: post trace 时间常数
    :type tau_post: float
    :param stride: 二维卷积步长
    :type stride: Tuple[int, int]
    :param padding: 二维卷积 padding
    :type padding: Tuple[int, int]
    :param padding_mode: padding 模式
    :type padding_mode: str
    :param reversed_padding_repeated_twice: 非零 padding 模式使用的展开 padding
    :type reversed_padding_repeated_twice: Tuple[int, ...] or List[int] or None
    :param dilation: 二维卷积 dilation
    :type dilation: Tuple[int, int]
    :param groups: 二维卷积分组数
    :type groups: int
    :param f_pre: pre 分支权重调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: post 分支权重调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 dilation 或 groups 不受支持时抛出

    ----

    .. _functional_stdp_conv2d_single_step-en:

    * **English**

    Run one tensor-only STDP update for raw Conv2d parameters. It preserves the
    existing support boundary: only ``dilation == (1, 1)`` and ``groups == 1``
    are supported. The function does not read modules, ``MemoryModule`` memory,
    ``step_mode`` or ``training/eval``.

    :param weight: Conv2d weight tensor
    :type weight: torch.Tensor
    :param in_spike: Input spikes shaped ``[N, C_in, H, W]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: Materialized pre-synaptic trace tensor. With non-zero
        padding, its spatial dimensions must match the padded ``in_spike``
    :type trace_pre: torch.Tensor
    :param trace_post: Materialized post-synaptic trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param stride: Two-dimensional convolution stride
    :type stride: Tuple[int, int]
    :param padding: Two-dimensional convolution padding
    :type padding: Tuple[int, int]
    :param padding_mode: Padding mode
    :type padding_mode: str
    :param reversed_padding_repeated_twice: Expanded padding used by non-zero modes
    :type reversed_padding_repeated_twice: Tuple[int, ...] or List[int] or None
    :param dilation: Two-dimensional convolution dilation
    :type dilation: Tuple[int, int]
    :param groups: Number of convolution groups
    :type groups: int
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: If dilation or groups is unsupported
    """
    if dilation != (1, 1):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv2d has not been implemented!"
        )
    if groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv2d has not been implemented!"
        )

    stride_h, stride_w = stride
    if padding != (0, 0):
        p_h, p_w = padding
        if padding_mode != "zeros":
            in_spike = F.pad(
                in_spike, reversed_padding_repeated_twice, mode=padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(p_w, p_w, p_h, p_h))

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(weight)
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
    return delta_w, trace_pre, trace_post


def stdp_conv1d_single_step(
    weight: torch.Tensor,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    tau_pre: float,
    tau_post: float,
    stride: tuple[int],
    padding: tuple[int],
    padding_mode: str = "zeros",
    reversed_padding_repeated_twice: tuple[int, ...] | list[int] | None = None,
    dilation: tuple[int] = (1,),
    groups: int = 1,
    f_pre: Callable[[torch.Tensor], torch.Tensor] = _identity,
    f_post: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_stdp_conv1d_single_step-cn>` | :ref:`English <functional_stdp_conv1d_single_step-en>`

    ----

    .. _functional_stdp_conv1d_single_step-cn:

    * **中文**

    对 Conv1d raw 参数执行单步 STDP tensor 更新。当前保持既有支持边界：只支持
    ``dilation == (1,)`` 且 ``groups == 1``。函数不读取 module 或
    ``MemoryModule`` memory，不管理 ``step_mode`` 或 ``training/eval``。

    :param weight: Conv1d 权重
    :type weight: torch.Tensor
    :param in_spike: 输入脉冲，形状 ``[N, C_in, L]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，形状 ``[N, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: 已物化的 pre trace tensor；存在 padding 时，其长度必须与
        padding 后的 ``in_spike`` 一致
    :type trace_pre: torch.Tensor
    :param trace_post: 已物化的 post trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: pre trace 时间常数
    :type tau_pre: float
    :param tau_post: post trace 时间常数
    :type tau_post: float
    :param stride: 一维卷积步长
    :type stride: Tuple[int]
    :param padding: 一维卷积 padding
    :type padding: Tuple[int]
    :param padding_mode: padding 模式
    :type padding_mode: str
    :param reversed_padding_repeated_twice: 非零 padding 模式使用的展开 padding
    :type reversed_padding_repeated_twice: Tuple[int, ...] or List[int] or None
    :param dilation: 一维卷积 dilation
    :type dilation: Tuple[int]
    :param groups: 一维卷积分组数
    :type groups: int
    :param f_pre: pre 分支权重调制函数
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: post 分支权重调制函数
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 dilation 或 groups 不受支持时抛出

    ----

    .. _functional_stdp_conv1d_single_step-en:

    * **English**

    Run one tensor-only STDP update for raw Conv1d parameters. It preserves the
    existing support boundary: only ``dilation == (1,)`` and ``groups == 1`` are
    supported. The function does not read modules or ``MemoryModule`` memory and
    does not manage ``step_mode`` or ``training/eval``.

    :param weight: Conv1d weight tensor
    :type weight: torch.Tensor
    :param in_spike: Input spikes shaped ``[N, C_in, L]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes shaped ``[N, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: Materialized pre-synaptic trace tensor. With non-zero
        padding, its length must match the padded ``in_spike``
    :type trace_pre: torch.Tensor
    :param trace_post: Materialized post-synaptic trace tensor
    :type trace_post: torch.Tensor
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param stride: One-dimensional convolution stride
    :type stride: Tuple[int]
    :param padding: One-dimensional convolution padding
    :type padding: Tuple[int]
    :param padding_mode: Padding mode
    :type padding_mode: str
    :param reversed_padding_repeated_twice: Expanded padding used by non-zero modes
    :type reversed_padding_repeated_twice: Tuple[int, ...] or List[int] or None
    :param dilation: One-dimensional convolution dilation
    :type dilation: Tuple[int]
    :param groups: Number of convolution groups
    :type groups: int
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable[[torch.Tensor], torch.Tensor]
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable[[torch.Tensor], torch.Tensor]
    :return: ``(delta_w, trace_pre_next, trace_post_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: If dilation or groups is unsupported
    """
    if dilation != (1,):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv1d has not been implemented!"
        )
    if groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv1d has not been implemented!"
        )

    stride_l = stride[0]
    if padding != (0,):
        p_l = padding[0]
        if padding_mode != "zeros":
            in_spike = F.pad(
                in_spike, reversed_padding_repeated_twice, mode=padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(p_l, p_l))

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(weight)
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
    return delta_w, trace_pre, trace_post


def mstdpet_reward_delta(
    reward: torch.Tensor | float,
    eligibility: torch.Tensor,
    trace_e: torch.Tensor,
    tau_trace: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <functional_mstdpet_reward_delta-cn>` | :ref:`English <functional_mstdpet_reward_delta-en>`

    ----

    .. _functional_mstdpet_reward_delta-cn:

    * **中文**

    执行 mSTDP-ET reward 调制前的 eligibility trace 衰减：
    ``trace_e = trace_e * exp(-1 / tau_trace) + eligibility / tau_trace``，
    并返回 ``reward * trace_e``。

    :param reward: reward 调制信号
    :type reward: torch.Tensor or float
    :param eligibility: 已物化的 eligibility tensor
    :type eligibility: torch.Tensor
    :param trace_e: 已物化的 eligibility trace tensor
    :type trace_e: torch.Tensor
    :param tau_trace: eligibility trace 时间常数
    :type tau_trace: float
    :return: ``(delta_w, trace_e_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    ----

    .. _functional_mstdpet_reward_delta-en:

    * **English**

    Apply the mSTDP-ET eligibility-trace decay before reward modulation:
    ``trace_e = trace_e * exp(-1 / tau_trace) + eligibility / tau_trace``, then
    return ``reward * trace_e``.

    :param reward: Reward modulation signal
    :type reward: torch.Tensor or float
    :param eligibility: Materialized eligibility tensor
    :type eligibility: torch.Tensor
    :param trace_e: Materialized eligibility-trace tensor
    :type trace_e: torch.Tensor
    :param tau_trace: Time constant of the eligibility trace
    :type tau_trace: float
    :return: ``(delta_w, trace_e_next)``
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    trace_e = trace_e * math.exp(-1 / tau_trace) + eligibility / tau_trace
    return reward * trace_e, trace_e
