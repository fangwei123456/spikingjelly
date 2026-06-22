from typing import Callable, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import neuron, monitor, base


def stdp_linear_single_step(
    fc: nn.Linear,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <stdp_linear_single_step-cn>` | :ref:`English <stdp_linear_single_step-en>`

    ----

    .. _stdp_linear_single_step-cn:

    * **中文**

    对单步脉冲输入执行全连接层的 STDP 更新，返回更新后的 pre/post trace 与权重增量。

    :param fc: 需要执行 STDP 的全连接层
    :type fc: torch.nn.Linear
    :param in_spike: 输入脉冲，``shape = [batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数，要求为正数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数，要求为正数
    :type tau_post: float
    :param f_pre: 作用在当前权重上的 pre-update 调制函数
    :type f_pre: Callable
    :param f_post: 作用在当前权重上的 post-update 调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``fc.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    ----

    .. _stdp_linear_single_step-en:

    * **English**

    Apply one STDP update step to a linear layer driven by single-step spike
    inputs, and return the updated pre/post traces together with the weight
    increment.

    :param fc: Linear layer updated by STDP
    :type fc: torch.nn.Linear
    :param in_spike: Input spikes with ``shape = [batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes with ``shape = [batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Pre-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Post-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace. Expected to be positive
    :type tau_pre: float
    :param tau_post: Time constant of the post trace. Expected to be positive
    :type tau_post: float
    :param f_pre: Pre-update modulation function applied to the current weight
    :type f_pre: Callable
    :param f_post: Post-update modulation function applied to the current weight
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``fc.weight``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    """
    if trace_pre is None:
        trace_pre = 0.0

    if trace_post is None:
        trace_post = 0.0

    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike  # shape = [batch_size, N_in]
    trace_post = (
        trace_post - trace_post / tau_post + out_spike
    )  # shape = [batch_size, N_out]

    # [batch_size, N_out, N_in] -> [N_out, N_in]
    delta_w_pre = -f_pre(weight) * (
        trace_post.unsqueeze(2) * in_spike.unsqueeze(1)
    ).sum(0)
    delta_w_post = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ).sum(0)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def mstdp_linear_single_step(
    fc: nn.Linear,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <mstdp_linear_single_step-cn>` | :ref:`English <mstdp_linear_single_step-en>`

    ----

    .. _mstdp_linear_single_step-cn:

    * **中文**

    对单步脉冲输入执行全连接层的 reward-modulated STDP eligibility 计算。

    :param fc: 需要执行 mSTDP 的全连接层
    :type fc: torch.nn.Linear
    :param in_spike: 输入脉冲，``shape = [batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``，其中 ``eligibility`` 的形状为
        ``[batch_size, out_features, in_features]``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    ----

    .. _mstdp_linear_single_step-en:

    * **English**

    Compute the reward-modulated STDP eligibility tensor for a linear layer
    under single-step spike inputs.

    :param fc: Linear layer updated by mSTDP
    :type fc: torch.nn.Linear
    :param in_spike: Input spikes with ``shape = [batch_size, in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes with ``shape = [batch_size, out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Pre-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Post-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``, where ``eligibility`` has
        shape ``[batch_size, out_features, in_features]``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    """
    if trace_pre is None:
        trace_pre = 0.0

    if trace_post is None:
        trace_post = 0.0

    weight = fc.weight.data
    trace_pre = (
        trace_pre * math.exp(-1 / tau_pre) + in_spike
    )  # shape = [batch_size, C_in]
    trace_post = (
        trace_post * math.exp(-1 / tau_post) + out_spike
    )  # shape = [batch_size, C_out]

    # [batch_size, N_out, N_in]
    eligibility = f_post(weight) * (
        trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)
    ) - f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1))
    return trace_pre, trace_post, eligibility


def mstdpet_linear_single_step(
    fc: nn.Linear,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    tau_trace: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <mstdpet_linear_single_step-cn>` | :ref:`English <mstdpet_linear_single_step-en>`

    ----

    .. _mstdpet_linear_single_step-cn:

    * **中文**

    对单步脉冲输入执行全连接层的 mSTDP-ET eligibility 计算。

    :param fc: 需要执行 mSTDP-ET 的全连接层
    :type fc: torch.nn.Linear
    :param in_spike: 输入脉冲，``shape = [in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace。若为 ``None``，则使用标量 ``0.0`` 初始化
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数
    :type tau_post: float
    :param tau_trace: eligibility trace 的时间常数
    :type tau_trace: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``，其中 ``eligibility`` 的形状与 ``fc.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    ----

    .. _mstdpet_linear_single_step-en:

    * **English**

    Compute the mSTDP-ET eligibility update for a linear layer under single-step
    spike inputs.

    :param fc: Linear layer updated by mSTDP-ET
    :type fc: torch.nn.Linear
    :param in_spike: Input spikes with ``shape = [in_features]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes with ``shape = [out_features]``
    :type out_spike: torch.Tensor
    :param trace_pre: Pre-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Post-synaptic trace. If ``None``, it is initialized as scalar ``0.0``
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param tau_trace: Time constant of the eligibility trace
    :type tau_trace: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``, where ``eligibility`` has
        the same shape as ``fc.weight``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    """
    if trace_pre is None:
        trace_pre = 0.0

    if trace_post is None:
        trace_post = 0.0

    weight = fc.weight.data
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike

    eligibility = f_post(weight) * torch.outer(out_spike, trace_pre) - f_pre(
        weight
    ) * torch.outer(trace_post, in_spike)
    return trace_pre, trace_post, eligibility


def stdp_conv2d_single_step(
    conv: nn.Conv2d,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None],
    trace_post: Union[torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <stdp_conv2d_single_step-cn>` | :ref:`English <stdp_conv2d_single_step-en>`

    ----

    .. _stdp_conv2d_single_step-cn:

    * **中文**

    对单步脉冲输入执行二维卷积层的 STDP 更新。

    当前仅支持 ``dilation == (1, 1)`` 且 ``groups == 1`` 的卷积。

    :param conv: 需要执行 STDP 的二维卷积层
    :type conv: torch.nn.Conv2d
    :param in_spike: 输入脉冲，``shape = [batch_size, C_in, H_in, W_in]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace。若为 ``None``，则初始化为与 ``in_spike`` 同形状零张量
    :type trace_pre: Union[torch.Tensor, None]
    :param trace_post: post-synaptic trace。若为 ``None``，则初始化为与 ``out_spike`` 同形状零张量
    :type trace_post: Union[torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``conv.weight`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 ``conv.dilation != (1, 1)`` 或 ``conv.groups != 1`` 时抛出

    ----

    .. _stdp_conv2d_single_step-en:

    * **English**

    Apply one STDP update step to a 2D convolution layer driven by single-step
    spike inputs.

    Only convolutions with ``dilation == (1, 1)`` and ``groups == 1`` are
    currently supported.

    :param conv: 2D convolution layer updated by STDP
    :type conv: torch.nn.Conv2d
    :param in_spike: Input spikes with ``shape = [batch_size, C_in, H_in, W_in]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes with ``shape = [batch_size, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: Pre-synaptic trace. If ``None``, initialized as zeros with the same shape as ``in_spike``
    :type trace_pre: Union[torch.Tensor, None]
    :param trace_post: Post-synaptic trace. If ``None``, initialized as zeros with the same shape as ``out_spike``
    :type trace_post: Union[torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``conv.weight``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: Raised when ``conv.dilation != (1, 1)`` or ``conv.groups != 1``
    """
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv2d has not been implemented!"
        )
    if conv.groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv2d has not been implemented!"
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != "zeros":
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[
                :, :, h:h_end:stride_h, w:w_end:stride_w
            ]  # shape = [batch_size, C_in, h_out, w_out]
            post_spike = out_spike  # shape = [batch_size, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]  # shape = [batch_size_out, C_in]

            tr_pre = trace_pre[
                :, :, h:h_end:stride_h, w:w_end:stride_w
            ]  # shape = [batch_size, C_in, h_out, w_out]
            tr_post = trace_post  # shape = [batch_size, C_out, h_out, w_out]

            delta_w_pre = -(
                f_pre(weight)
                * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                .permute([1, 2, 0, 3, 4])
                .sum(dim=[2, 3, 4])
            )
            delta_w_post = f_post(weight) * (
                tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)
            ).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


def stdp_conv1d_single_step(
    conv: nn.Conv1d,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None],
    trace_post: Union[torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <stdp_conv1d_single_step-cn>` | :ref:`English <stdp_conv1d_single_step-en>`

    ----

    .. _stdp_conv1d_single_step-cn:

    * **中文**

    对单步脉冲输入执行一维卷积层的 STDP 更新。

    当前仅支持 ``dilation == (1,)`` 且 ``groups == 1`` 的卷积。

    :param conv: 需要执行 STDP 的一维卷积层
    :type conv: torch.nn.Conv1d
    :param in_spike: 输入脉冲，``shape = [batch_size, C_in, L_in]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace。若为 ``None``，则初始化为与 ``in_spike`` 同形状零张量
    :type trace_pre: Union[torch.Tensor, None]
    :param trace_post: post-synaptic trace。若为 ``None``，则初始化为与 ``out_spike`` 同形状零张量
    :type trace_post: Union[torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``conv.weight`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 ``conv.dilation != (1,)`` 或 ``conv.groups != 1`` 时抛出

    ----

    .. _stdp_conv1d_single_step-en:

    * **English**

    Apply one STDP update step to a 1D convolution layer driven by single-step
    spike inputs.

    Only convolutions with ``dilation == (1,)`` and ``groups == 1`` are
    currently supported.

    :param conv: 1D convolution layer updated by STDP
    :type conv: torch.nn.Conv1d
    :param in_spike: Input spikes with ``shape = [batch_size, C_in, L_in]``
    :type in_spike: torch.Tensor
    :param out_spike: Output spikes with ``shape = [batch_size, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: Pre-synaptic trace. If ``None``, initialized as zeros with the same shape as ``in_spike``
    :type trace_pre: Union[torch.Tensor, None]
    :param trace_post: Post-synaptic trace. If ``None``, initialized as zeros with the same shape as ``out_spike``
    :type trace_post: Union[torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``conv.weight``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: Raised when ``conv.dilation != (1,)`` or ``conv.groups != 1``
    """
    if conv.dilation != (1,):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv1d has not been implemented!"
        )
    if conv.groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv1d has not been implemented!"
        )

    stride_l = conv.stride[0]

    if conv.padding == (0,):
        pass
    else:
        pL = conv.padding[0]
        if conv.padding_mode != "zeros":
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pL, pL))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for l in range(conv.weight.shape[2]):
        l_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + l
        pre_spike = in_spike[
            :, :, l:l_end:stride_l
        ]  # shape = [batch_size, C_in, l_out]
        post_spike = out_spike  # shape = [batch_size, C_out, l_out]
        weight = conv.weight.data[:, :, l]  # shape = [batch_size_out, C_in]

        tr_pre = trace_pre[:, :, l:l_end:stride_l]  # shape = [batch_size, C_in, l_out]
        tr_post = trace_post  # shape = [batch_size, C_out, l_out]

        delta_w_pre = -(
            f_pre(weight)
            * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
            .permute([1, 2, 0, 3])
            .sum(dim=[2, 3])
        )
        delta_w_post = f_post(weight) * (
            tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)
        ).permute([1, 2, 0, 3]).sum(dim=[2, 3])
        delta_w[:, :, l] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


def stdp_multi_step(
    layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
):
    r"""
    **API Language:**
    :ref:`中文 <stdp_multi_step-cn>` | :ref:`English <stdp_multi_step-en>`

    ----

    .. _stdp_multi_step-cn:

    * **中文**

    对线性层、一维卷积层或二维卷积层执行多步 STDP 更新。

    该函数沿时间维遍历 ``in_spike`` 与 ``out_spike``，并在每个时间步调用对应的
    单步 STDP 规则，最终累加得到整段序列的权重增量。

    :param layer: 支持的突触层，目前为 ``nn.Linear``、``nn.Conv1d`` 或 ``nn.Conv2d``
    :type layer: Union[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]
    :param in_spike: 输入脉冲序列，时间维位于第 0 维
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲序列，时间维位于第 0 维
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace 的初始值
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace 的初始值
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``layer.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    ----

    .. _stdp_multi_step-en:

    * **English**

    Apply multi-step STDP updates to a linear layer, a 1D convolution layer, or
    a 2D convolution layer.

    The function iterates over the time dimension of ``in_spike`` and
    ``out_spike``, applies the matching single-step STDP rule at each time step,
    and accumulates the weight increment over the whole sequence.

    :param layer: Supported synaptic layer. Currently ``nn.Linear``, ``nn.Conv1d`` or ``nn.Conv2d``
    :type layer: Union[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]
    :param in_spike: Input spike sequence with the time axis at dimension 0
    :type in_spike: torch.Tensor
    :param out_spike: Output spike sequence with the time axis at dimension 0
    :type out_spike: torch.Tensor
    :param trace_pre: Initial value of the pre-synaptic trace
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Initial value of the post-synaptic trace
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``layer.weight``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    """
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]

    if isinstance(layer, nn.Linear):
        stdp_single_step = stdp_linear_single_step

    elif isinstance(layer, nn.Conv1d):
        stdp_single_step = stdp_conv1d_single_step

    elif isinstance(layer, nn.Conv2d):
        stdp_single_step = stdp_conv2d_single_step

    for t in range(T):
        trace_pre, trace_post, dw = stdp_single_step(
            layer,
            in_spike[t],
            out_spike[t],
            trace_pre,
            trace_post,
            tau_pre,
            tau_post,
            f_pre,
            f_post,
        )
        delta_w += dw

    return trace_pre, trace_post, delta_w


class STDPLearner(base.MemoryModule):
    def __init__(
        self,
        step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear],
        sn: neuron.BaseNode,
        tau_pre: float,
        tau_post: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ):
        r"""
        **API Language:**
        :ref:`中文 <STDPLearner.__init__-cn>` | :ref:`English <STDPLearner.__init__-en>`

        ----

        .. _STDPLearner.__init__-cn:

        * **中文**

        基于监视器的 STDP 学习器。

        该学习器通过 :class:`InputMonitor <spikingjelly.activation_based.monitor.InputMonitor>`
        和 :class:`OutputMonitor <spikingjelly.activation_based.monitor.OutputMonitor>`
        自动记录突触层输入脉冲与神经元输出脉冲，并在调用 :meth:`step` 时根据
        ``step_mode`` 选择单步或多步 STDP 规则。

        :param step_mode: ``'s'`` 表示单步 STDP，``'m'`` 表示多步 STDP
        :type step_mode: str
        :param synapse: 需要执行 STDP 的突触层，目前支持 ``nn.Linear``、``nn.Conv1d``、``nn.Conv2d``
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: 产生输出脉冲的脉冲神经元模块
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: pre trace 的时间常数
        :type tau_pre: float
        :param tau_post: post trace 的时间常数
        :type tau_post: float
        :param f_pre: pre 分支的权重调制函数
        :type f_pre: Callable
        :param f_post: post 分支的权重调制函数
        :type f_post: Callable

        ----

        .. _STDPLearner.__init__-en:

        * **English**

        Monitor-based STDP learner.

        The learner automatically records synaptic input spikes and neuronal
        output spikes with
        :class:`InputMonitor <spikingjelly.activation_based.monitor.InputMonitor>`
        and
        :class:`OutputMonitor <spikingjelly.activation_based.monitor.OutputMonitor>`.
        When :meth:`step` is called, it selects the single-step or multi-step
        STDP rule according to ``step_mode``.

        :param step_mode: ``'s'`` for single-step STDP and ``'m'`` for multi-step STDP
        :type step_mode: str
        :param synapse: Synaptic layer updated by STDP. Currently supports ``nn.Linear``, ``nn.Conv1d``, and ``nn.Conv2d``
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: Spiking neuron module that generates output spikes
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: Time constant of the pre trace
        :type tau_pre: float
        :param tau_post: Time constant of the post trace
        :type tau_post: float
        :param f_pre: Weight modulation function for the pre branch
        :type f_pre: Callable
        :param f_post: Weight modulation function for the post branch
        :type f_post: Callable
        """
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)

    def reset(self):
        r"""
        **API Language:**
        :ref:`中文 <STDPLearner.reset-cn>` | :ref:`English <STDPLearner.reset-en>`

        ----

        .. _STDPLearner.reset-cn:

        * **中文**

        重置学习器内部状态，并清空输入/输出脉冲监视器中已记录的数据。


        ----

        .. _STDPLearner.reset-en:

        * **English**

        Reset the learner state and clear all recorded data in the input/output spike monitors.
        """
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        r"""
        **API Language:**
        :ref:`中文 <STDPLearner.disable-cn>` | :ref:`English <STDPLearner.disable-en>`

        ----

        .. _STDPLearner.disable-cn:

        * **中文**

        禁用输入脉冲与输出脉冲监视器，使其停止记录新数据。


        ----

        .. _STDPLearner.disable-en:

        * **English**

        Disable the input and output spike monitors so they stop recording new data.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        r"""
        **API Language:**
        :ref:`中文 <STDPLearner.enable-cn>` | :ref:`English <STDPLearner.enable-en>`

        ----

        .. _STDPLearner.enable-cn:

        * **中文**

        启用输入脉冲与输出脉冲监视器，使其恢复记录。


        ----

        .. _STDPLearner.enable-en:

        * **English**

        Enable the input and output spike monitors so they resume recording.
        """
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: bool = True, scale: float = 1.0):
        r"""
        **API Language:**
        :ref:`中文 <STDPLearner.step-cn>` | :ref:`English <STDPLearner.step-en>`

        ----

        .. _STDPLearner.step-cn:

        * **中文**

        使用当前监视器中缓存的脉冲记录执行一次 STDP 权重更新。

        当 ``on_grad=True`` 时，函数会把 ``-delta_w`` 累加到 ``self.synapse.weight.grad``；
        当 ``on_grad=False`` 时，返回累计的 ``delta_w`` 而不写入梯度。

        :param on_grad: 是否将权重增量写入 ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: 对累计权重增量施加的缩放因子
        :type scale: float
        :return: 当 ``on_grad=False`` 时返回累计的权重增量；否则返回 ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: 当 ``self.step_mode`` 与突触层类型组合当前不受支持时抛出
        :raises ValueError: 当 ``self.step_mode`` 不是 ``'s'`` 或 ``'m'`` 时抛出

        ----

        .. _STDPLearner.step-en:

        * **English**

        Perform one STDP update using the spike records currently buffered in the monitors.

        When ``on_grad=True``, the function accumulates ``-delta_w`` into
        ``self.synapse.weight.grad``. When ``on_grad=False``, it returns the
        accumulated ``delta_w`` without writing gradients.

        :param on_grad: Whether to write the weight increment into ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: Scaling factor applied to the accumulated weight increment
        :type scale: float
        :return: The accumulated weight increment when ``on_grad=False``; otherwise ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: Raised when the current ``step_mode`` / synapse-type combination is unsupported
        :raises ValueError: Raised when ``self.step_mode`` is neither ``'s'`` nor ``'m'``
        """
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == "s":
            if isinstance(self.synapse, nn.Linear):
                stdp_f = stdp_linear_single_step
            elif isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Conv1d):
                stdp_f = stdp_conv1d_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == "m":
            if isinstance(self.synapse, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                stdp_f = stdp_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]

            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse,
                in_spike,
                out_spike,
                self.trace_pre,
                self.trace_post,
                self.tau_pre,
                self.tau_post,
                self.f_pre,
                self.f_post,
            )
            if scale != 1.0:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class MSTDPLearner(base.MemoryModule):
    def __init__(
        self,
        step_mode: str,
        batch_size: float,
        synapse: Union[nn.Conv2d, nn.Linear],
        sn: neuron.BaseNode,
        tau_pre: float,
        tau_post: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPLearner.__init__-cn>` | :ref:`English <MSTDPLearner.__init__-en>`

        ----

        .. _MSTDPLearner.__init__-cn:

        * **中文**

        reward-modulated STDP（mSTDP）学习器。

        该学习器维护每个样本对应的 eligibility，并在 :meth:`step` 中结合外部奖励
        ``reward`` 把 eligibility 转换为权重更新。

        :param step_mode: ``'s'`` 表示单步 mSTDP
        :type step_mode: str
        :param batch_size: 每次奖励调制时使用的 batch 大小
        :type batch_size: float
        :param synapse: 需要执行 mSTDP 的突触层
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: 产生输出脉冲的脉冲神经元模块
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: pre trace 的时间常数
        :type tau_pre: float
        :param tau_post: post trace 的时间常数
        :type tau_post: float
        :param f_pre: pre 分支的权重调制函数
        :type f_pre: Callable
        :param f_post: post 分支的权重调制函数
        :type f_post: Callable

        ----

        .. _MSTDPLearner.__init__-en:

        * **English**

        Reward-modulated STDP (mSTDP) learner.

        The learner maintains per-sample eligibility tensors and converts them
        into weight updates inside :meth:`step` using the external reward
        ``reward``.

        :param step_mode: ``'s'`` for single-step mSTDP
        :type step_mode: str
        :param batch_size: Batch size used when modulating eligibility with rewards
        :type batch_size: float
        :param synapse: Synaptic layer updated by mSTDP
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: Spiking neuron module that generates output spikes
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: Time constant of the pre trace
        :type tau_pre: float
        :param tau_post: Time constant of the post trace
        :type tau_post: float
        :param f_pre: Weight modulation function for the pre branch
        :type f_pre: Callable
        :param f_post: Weight modulation function for the post branch
        :type f_post: Callable
        """
        super().__init__()
        self.step_mode = step_mode
        self.batch_size = batch_size
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)

    def reset(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPLearner.reset-cn>` | :ref:`English <MSTDPLearner.reset-en>`

        ----

        .. _MSTDPLearner.reset-cn:

        * **中文**

        重置学习器内部状态，并清空监视器缓存。


        ----

        .. _MSTDPLearner.reset-en:

        * **English**

        Reset the learner state and clear the monitor buffers.
        """
        super(MSTDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPLearner.disable-cn>` | :ref:`English <MSTDPLearner.disable-en>`

        ----

        .. _MSTDPLearner.disable-cn:

        * **中文**

        禁用输入/输出监视器。


        ----

        .. _MSTDPLearner.disable-en:

        * **English**

        Disable the input/output monitors.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPLearner.enable-cn>` | :ref:`English <MSTDPLearner.enable-en>`

        ----

        .. _MSTDPLearner.enable-cn:

        * **中文**

        启用输入/输出监视器。


        ----

        .. _MSTDPLearner.enable-en:

        * **English**

        Enable the input/output monitors.
        """
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, reward, on_grad: bool = True, scale: float = 1.0):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPLearner.step-cn>` | :ref:`English <MSTDPLearner.step-en>`

        ----

        .. _MSTDPLearner.step-cn:

        * **中文**

        使用外部奖励 ``reward`` 对当前 eligibility 进行调制，并生成一次 mSTDP 权重更新。

        :param reward: 每个样本对应的奖励，通常 ``shape = [batch_size]``
        :type reward: torch.Tensor
        :param on_grad: 是否将结果写入 ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: 权重增量的缩放因子
        :type scale: float
        :return: 当 ``on_grad=False`` 时返回累计的权重增量；否则返回 ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: 当前仅部分支持特定 ``step_mode`` 与突触层组合
        :raises ValueError: 当 ``self.step_mode`` 不合法时抛出

        ----

        .. _MSTDPLearner.step-en:

        * **English**

        Modulate the current eligibility with the external reward ``reward`` and
        generate one mSTDP weight update.

        :param reward: Reward for each sample, typically with ``shape = [batch_size]``
        :type reward: torch.Tensor
        :param on_grad: Whether to write the result into ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: Scaling factor applied to the weight increment
        :type scale: float
        :return: The accumulated weight increment when ``on_grad=False``; otherwise ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: Only some ``step_mode`` / synapse-type combinations are currently supported
        :raises ValueError: Raised when ``self.step_mode`` is invalid
        """
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == "s":
            if isinstance(self.synapse, nn.Conv2d):
                # stdp_f = mstdp_conv2d_single_step
                raise NotImplementedError(self.synapse)
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = mstdp_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == "m":
            if isinstance(self.synapse, nn.Conv2d) or isinstance(
                self.synapse, nn.Linear
            ):
                # stdp_f = mstdp_multi_step
                raise NotImplementedError(self.synapse)
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            if not hasattr(self, "eligibility"):
                self.eligibility = torch.zeros(
                    self.batch_size,
                    *self.synapse.weight.shape,
                    device=self.synapse.weight.device,
                )

            dw = (reward.view(-1, 1, 1) * self.eligibility).sum(
                0
            )  # [batch_size, N_out, N_in] -> [N_out, N_in]

            if scale != 1.0:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]

            self.trace_pre, self.trace_post, self.eligibility = stdp_f(
                self.synapse,
                in_spike,
                out_spike,
                self.trace_pre,
                self.trace_post,
                self.tau_pre,
                self.tau_post,
                self.f_pre,
                self.f_post,
            )

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class MSTDPETLearner(base.MemoryModule):
    def __init__(
        self,
        step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear],
        sn: neuron.BaseNode,
        tau_pre: float,
        tau_post: float,
        tau_trace: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPETLearner.__init__-cn>` | :ref:`English <MSTDPETLearner.__init__-en>`

        ----

        .. _MSTDPETLearner.__init__-cn:

        * **中文**

        mSTDP-ET 学习器。

        与 :class:`MSTDPLearner` 相比，该学习器额外维护随时间衰减的 eligibility
        trace ``trace_e``，并在 :meth:`step` 中结合奖励生成最终权重更新。

        :param step_mode: ``'s'`` 表示单步 mSTDP-ET
        :type step_mode: str
        :param synapse: 需要执行 mSTDP-ET 的突触层
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: 产生输出脉冲的脉冲神经元模块
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: pre trace 的时间常数
        :type tau_pre: float
        :param tau_post: post trace 的时间常数
        :type tau_post: float
        :param tau_trace: eligibility trace 的时间常数
        :type tau_trace: float
        :param f_pre: pre 分支的权重调制函数
        :type f_pre: Callable
        :param f_post: post 分支的权重调制函数
        :type f_post: Callable

        ----

        .. _MSTDPETLearner.__init__-en:

        * **English**

        mSTDP-ET learner.

        Compared with :class:`MSTDPLearner`, this learner additionally maintains
        a temporally decaying eligibility trace ``trace_e`` and combines it with
        rewards inside :meth:`step` to produce the final weight update.

        :param step_mode: ``'s'`` for single-step mSTDP-ET
        :type step_mode: str
        :param synapse: Synaptic layer updated by mSTDP-ET
        :type synapse: Union[torch.nn.Conv2d, torch.nn.Linear]
        :param sn: Spiking neuron module that generates output spikes
        :type sn: spikingjelly.activation_based.neuron.BaseNode
        :param tau_pre: Time constant of the pre trace
        :type tau_pre: float
        :param tau_post: Time constant of the post trace
        :type tau_post: float
        :param tau_trace: Time constant of the eligibility trace
        :type tau_trace: float
        :param f_pre: Weight modulation function for the pre branch
        :type f_pre: Callable
        :param f_post: Weight modulation function for the post branch
        :type f_post: Callable
        """
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_trace = tau_trace
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)
        self.register_memory("trace_e", None)

    def reset(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPETLearner.reset-cn>` | :ref:`English <MSTDPETLearner.reset-en>`

        ----

        .. _MSTDPETLearner.reset-cn:

        * **中文**

        重置学习器内部状态，并清空监视器缓存。


        ----

        .. _MSTDPETLearner.reset-en:

        * **English**

        Reset the learner state and clear the monitor buffers.
        """
        super(MSTDPETLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPETLearner.disable-cn>` | :ref:`English <MSTDPETLearner.disable-en>`

        ----

        .. _MSTDPETLearner.disable-cn:

        * **中文**

        禁用输入/输出监视器。


        ----

        .. _MSTDPETLearner.disable-en:

        * **English**

        Disable the input/output monitors.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPETLearner.enable-cn>` | :ref:`English <MSTDPETLearner.enable-en>`

        ----

        .. _MSTDPETLearner.enable-cn:

        * **中文**

        启用输入/输出监视器。


        ----

        .. _MSTDPETLearner.enable-en:

        * **English**

        Enable the input/output monitors.
        """
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, reward, on_grad: bool = True, scale: float = 1.0):
        r"""
        **API Language:**
        :ref:`中文 <MSTDPETLearner.step-cn>` | :ref:`English <MSTDPETLearner.step-en>`

        ----

        .. _MSTDPETLearner.step-cn:

        * **中文**

        使用外部奖励 ``reward`` 对 eligibility trace 进行调制，并生成一次 mSTDP-ET 权重更新。

        :param reward: 奖励信号，通常为标量或 ``shape = [batch_size]`` 的张量
        :type reward: torch.Tensor
        :param on_grad: 是否将结果写入 ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: 权重增量的缩放因子
        :type scale: float
        :return: 当 ``on_grad=False`` 时返回累计的权重增量；否则返回 ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: 当前仅部分支持特定 ``step_mode`` 与突触层组合
        :raises ValueError: 当 ``self.step_mode`` 不合法时抛出

        ----

        .. _MSTDPETLearner.step-en:

        * **English**

        Modulate the eligibility trace with the external reward ``reward`` and
        generate one mSTDP-ET weight update.

        :param reward: Reward signal, typically a scalar or a tensor with ``shape = [batch_size]``
        :type reward: torch.Tensor
        :param on_grad: Whether to write the result into ``self.synapse.weight.grad``
        :type on_grad: bool
        :param scale: Scaling factor applied to the weight increment
        :type scale: float
        :return: The accumulated weight increment when ``on_grad=False``; otherwise ``None``
        :rtype: Optional[torch.Tensor]
        :raises NotImplementedError: Only some ``step_mode`` / synapse-type combinations are currently supported
        :raises ValueError: Raised when ``self.step_mode`` is invalid
        """
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == "s":
            if isinstance(self.synapse, nn.Conv2d):
                # stdp_f = mstdpet_conv2d_single_step
                raise NotImplementedError(self.synapse)
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = mstdpet_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == "m":
            if isinstance(self.synapse, nn.Conv2d) or isinstance(
                self.synapse, nn.Linear
            ):
                # stdp_f = mstdpet_multi_step
                raise NotImplementedError(self.synapse)
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            if not hasattr(self, "eligibility"):
                self.eligibility = torch.zeros(
                    *self.synapse.weight.shape, device=self.synapse.weight.device
                )

            if self.trace_e is None:
                self.trace_e = 0.0

            self.trace_e = (
                self.trace_e * math.exp(-1 / self.tau_trace)
                + self.eligibility / self.tau_trace
            )

            dw = reward * self.trace_e

            if scale != 1.0:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)

            self.trace_pre, self.trace_post, self.eligibility = stdp_f(
                self.synapse,
                in_spike,
                out_spike,
                self.trace_pre,
                self.trace_post,
                self.tau_pre,
                self.tau_post,
                self.tau_trace,
                self.f_pre,
                self.f_post,
            )

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
