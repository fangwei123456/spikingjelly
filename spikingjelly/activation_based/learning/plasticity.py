from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import base, monitor
from ..functional import learning as functional_learning


__all__ = [
    "stdp_linear_single_step",
    "mstdp_linear_single_step",
    "mstdpet_linear_single_step",
    "stdp_conv2d_single_step",
    "stdp_conv1d_single_step",
    "stdp_multi_step",
    "STDPLearner",
    "MSTDPLearner",
    "MSTDPETLearner",
]


def _check_open(learner) -> None:
    if not learner.in_spike_monitor.hooks or not learner.out_spike_monitor.hooks:
        raise RuntimeError("Learner is closed; create a new learner to resume.")


def _check_records(learner) -> None:
    synapse = learner.synapse
    expected_pre = None if learner.trace_pre is None else learner.trace_pre.shape
    expected_post = None if learner.trace_post is None else learner.trace_post.shape
    for state in (learner.trace_pre, learner.trace_post):
        if state is not None and state.device != synapse.weight.device:
            raise ValueError("Traces and weight must be on the same device.")
    ndim = 2 if isinstance(synapse, nn.Linear) else synapse.weight.ndim
    for pre, post in zip(
        learner.in_spike_monitor.records, learner.out_spike_monitor.records
    ):
        if learner.step_mode == "m":
            if pre.ndim != ndim + 1 or post.ndim != ndim + 1:
                raise ValueError(
                    "Multi-step spike dimensions do not match the synapse."
                )
            if pre.shape[0] != post.shape[0]:
                raise ValueError("Input and output time lengths differ.")
            if not pre.shape[0]:
                continue
            pre, post = pre[0], post[0]
        if pre.ndim != ndim or post.ndim != ndim:
            raise ValueError("Spike dimensions do not match the synapse.")
        channel = 0 if ndim == 1 else 1
        if (
            pre.shape[channel] != synapse.weight.shape[1]
            or post.shape[channel] != synapse.weight.shape[0]
            or (ndim > 1 and pre.shape[0] != post.shape[0])
        ):
            raise ValueError("Spike shapes do not match weight or batch dimensions.")
        pre_shape = pre.shape
        if isinstance(synapse, (nn.Conv1d, nn.Conv2d)):
            pre_shape = (
                *pre.shape[:2],
                *(
                    size
                    + sum(synapse._reversed_padding_repeated_twice[index : index + 2])
                    for size, index in zip(
                        pre.shape[2:],
                        range(
                            len(synapse._reversed_padding_repeated_twice) - 2, -1, -2
                        ),
                    )
                ),
            )
            expected = tuple(
                (size - kernel) // stride + 1
                for size, kernel, stride in zip(
                    pre_shape[2:], synapse.kernel_size, synapse.stride
                )
            )
            if post.shape[2:] != expected:
                raise ValueError("Output spatial shape does not match the synapse.")
        for expected, shape in (
            (expected_pre, pre_shape),
            (expected_post, post.shape),
        ):
            if expected is not None and expected != shape:
                raise ValueError(
                    "Trace shape differs; reset before changing batch size."
                )
        expected_pre, expected_post = pre_shape, post.shape
        if pre.device != synapse.weight.device or post.device != synapse.weight.device:
            raise ValueError("Spikes and weight must be on the same device.")


def _materialize_trace(trace, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(trace, torch.Tensor):
        return trace
    return torch.full_like(reference, 0.0 if trace is None else trace)


def _padded_conv_input(conv: Union[nn.Conv1d, nn.Conv2d], x: torch.Tensor):
    padding = conv._reversed_padding_repeated_twice
    if not any(padding):
        return x
    mode = "constant" if conv.padding_mode == "zeros" else conv.padding_mode
    return F.pad(x, padding, mode=mode)


def _accumulate_weight_gradient(synapse, delta_w):
    if synapse.weight.grad is None:
        synapse.weight.grad = -delta_w
    else:
        synapse.weight.grad = synapse.weight.grad - delta_w


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stdp_linear_single_step-cn>` | :ref:`English <stdp_linear_single_step-en>`

    ----

    .. _stdp_linear_single_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

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
    :param tau_pre: 单位为时间步，有限且 >= 1； pre trace 的时间常数，要求为正数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 >= 1； post trace 的时间常数，要求为正数
    :type tau_post: float
    :param f_pre: 作用在当前权重上的 pre-update 调制函数
    :type f_pre: Callable
    :param f_post: 作用在当前权重上的 post-update 调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``fc.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _stdp_linear_single_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param tau_pre: In time steps, finite and >= 1;  Time constant of the pre trace. Expected to be positive
    :type tau_pre: float
    :param tau_post: In time steps, finite and >= 1;  Time constant of the post trace. Expected to be positive
    :type tau_post: float
    :param f_pre: Pre-update modulation function applied to the current weight
    :type f_pre: Callable
    :param f_post: Post-update modulation function applied to the current weight
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``fc.weight``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    delta_w, (trace_pre, trace_post) = functional_learning.stdp_linear_step(
        in_spike,
        out_spike,
        (
            _materialize_trace(trace_pre, in_spike),
            _materialize_trace(trace_post, out_spike),
        ),
        fc.weight.data,
        tau_pre=tau_pre,
        tau_post=tau_post,
        f_pre=f_pre,
        f_post=f_post,
    )
    return trace_pre, trace_post, delta_w


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <mstdp_linear_single_step-cn>` | :ref:`English <mstdp_linear_single_step-en>`

    ----

    .. _mstdp_linear_single_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

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
    :param tau_pre: 单位为时间步，有限且 > 0； pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 > 0； post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``，其中 ``eligibility`` 的形状为
        ``[batch_size, out_features, in_features]``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _mstdp_linear_single_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param tau_pre: In time steps, finite and > 0;  Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: In time steps, finite and > 0;  Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``, where ``eligibility`` has
        shape ``[batch_size, out_features, in_features]``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    eligibility, (trace_pre, trace_post) = functional_learning.mstdp_linear_step(
        in_spike,
        out_spike,
        (
            _materialize_trace(trace_pre, in_spike),
            _materialize_trace(trace_post, out_spike),
        ),
        fc.weight.data,
        tau_pre=tau_pre,
        tau_post=tau_post,
        f_pre=f_pre,
        f_post=f_post,
    )
    return trace_pre, trace_post, eligibility


def mstdpet_linear_single_step(
    fc: nn.Linear,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <mstdpet_linear_single_step-cn>` | :ref:`English <mstdpet_linear_single_step-en>`

    ----

    .. _mstdpet_linear_single_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

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
    :param tau_pre: 单位为时间步，有限且 > 0； pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 > 0； post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``，其中 ``eligibility`` 的形状与 ``fc.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _mstdpet_linear_single_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param tau_pre: In time steps, finite and > 0;  Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: In time steps, finite and > 0;  Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, eligibility)``, where ``eligibility`` has
        the same shape as ``fc.weight``
    :rtype: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor], torch.Tensor]
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    eligibility, (trace_pre, trace_post) = functional_learning.mstdpet_linear_step(
        in_spike,
        out_spike,
        (
            _materialize_trace(trace_pre, in_spike),
            _materialize_trace(trace_post, out_spike),
        ),
        fc.weight.data,
        tau_pre=tau_pre,
        tau_post=tau_post,
        f_pre=f_pre,
        f_post=f_post,
    )
    return trace_pre, trace_post, eligibility


def stdp_conv2d_single_step(
    conv: nn.Conv2d,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stdp_conv2d_single_step-cn>` | :ref:`English <stdp_conv2d_single_step-en>`

    ----

    .. _stdp_conv2d_single_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

    对单步脉冲输入执行二维卷积层的 STDP 更新。

    当前仅支持 ``dilation == (1, 1)`` 且 ``groups == 1`` 的卷积。

    :param conv: 需要执行 STDP 的二维卷积层
    :type conv: torch.nn.Conv2d
    :param in_spike: 输入脉冲，``shape = [batch_size, C_in, H_in, W_in]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, C_out, H_out, W_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace；float 填充 padding 后输入形状，
        ``None`` 使用零；Tensor 须同形状
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace；float 填充 ``out_spike`` 形状，
        ``None`` 使用零；Tensor 须同形状
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: 单位为时间步，有限且 >= 1； pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 >= 1； post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``conv.weight`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 ``conv.dilation != (1, 1)`` 或 ``conv.groups != 1`` 时抛出

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _stdp_conv2d_single_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param trace_pre: Pre-synaptic trace; float fills the padded input shape,
        ``None`` uses zeros; tensors must match that shape
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Post-synaptic trace; float fills the ``out_spike`` shape,
        ``None`` uses zeros; tensors must match that shape
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: In time steps, finite and >= 1;  Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: In time steps, finite and >= 1;  Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``conv.weight``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: Raised when ``conv.dilation != (1, 1)`` or ``conv.groups != 1``
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv2d has not been implemented!"
        )
    if conv.groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv2d has not been implemented!"
        )
    in_spike = _padded_conv_input(conv, in_spike)
    delta_w, (trace_pre, trace_post) = functional_learning.stdp_conv2d_step(
        in_spike,
        out_spike,
        (
            _materialize_trace(trace_pre, in_spike),
            _materialize_trace(trace_post, out_spike),
        ),
        conv.weight.data,
        stride=conv.stride,
        tau_pre=tau_pre,
        tau_post=tau_post,
        f_pre=f_pre,
        f_post=f_post,
    )
    return trace_pre, trace_post, delta_w


def stdp_conv1d_single_step(
    conv: nn.Conv1d,
    in_spike: torch.Tensor,
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float,
    tau_post: float,
    f_pre: Callable = lambda x: x,
    f_post: Callable = lambda x: x,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <stdp_conv1d_single_step-cn>` | :ref:`English <stdp_conv1d_single_step-en>`

    ----

    .. _stdp_conv1d_single_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

    对单步脉冲输入执行一维卷积层的 STDP 更新。

    当前仅支持 ``dilation == (1,)`` 且 ``groups == 1`` 的卷积。

    :param conv: 需要执行 STDP 的一维卷积层
    :type conv: torch.nn.Conv1d
    :param in_spike: 输入脉冲，``shape = [batch_size, C_in, L_in]``
    :type in_spike: torch.Tensor
    :param out_spike: 输出脉冲，``shape = [batch_size, C_out, L_out]``
    :type out_spike: torch.Tensor
    :param trace_pre: pre-synaptic trace；float 填充 padding 后输入形状，
        ``None`` 使用零；Tensor 须同形状
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: post-synaptic trace；float 填充 ``out_spike`` 形状，
        ``None`` 使用零；Tensor 须同形状
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: 单位为时间步，有限且 >= 1； pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 >= 1； post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``conv.weight`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: 当 ``conv.dilation != (1,)`` 或 ``conv.groups != 1`` 时抛出

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _stdp_conv1d_single_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param trace_pre: Pre-synaptic trace; float fills the padded input shape,
        ``None`` uses zeros; tensors must match that shape
    :type trace_pre: Union[float, torch.Tensor, None]
    :param trace_post: Post-synaptic trace; float fills the ``out_spike`` shape,
        ``None`` uses zeros; tensors must match that shape
    :type trace_post: Union[float, torch.Tensor, None]
    :param tau_pre: In time steps, finite and >= 1;  Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: In time steps, finite and >= 1;  Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``conv.weight``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises NotImplementedError: Raised when ``conv.dilation != (1,)`` or ``conv.groups != 1``
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    if conv.dilation != (1,):
        raise NotImplementedError(
            "STDP with dilation != 1 for Conv1d has not been implemented!"
        )
    if conv.groups != 1:
        raise NotImplementedError(
            "STDP with groups != 1 for Conv1d has not been implemented!"
        )
    in_spike = _padded_conv_input(conv, in_spike)
    delta_w, (trace_pre, trace_post) = functional_learning.stdp_conv1d_step(
        in_spike,
        out_spike,
        (
            _materialize_trace(trace_pre, in_spike),
            _materialize_trace(trace_post, out_spike),
        ),
        conv.weight.data,
        stride=conv.stride,
        tau_pre=tau_pre,
        tau_post=tau_post,
        f_pre=f_pre,
        f_post=f_post,
    )
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
) -> tuple[
    Union[float, torch.Tensor, None], Union[float, torch.Tensor, None], torch.Tensor
]:
    r"""
    **API Language** - :ref:`中文 <stdp_multi_step-cn>` | :ref:`English <stdp_multi_step-en>`

    ----

    .. _stdp_multi_step-cn:

    * **中文**

    活动张量须使用与权重相同的浮点 dtype 和 device；输出 dtype/device
    与权重一致。先加入当前脉冲再计算更新，同一步脉冲是否抵消取决于调制函数。输入张量状态不被原地修改。

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
    :param tau_pre: 单位为时间步，有限且 >= 1； pre trace 的时间常数
    :type tau_pre: float
    :param tau_post: 单位为时间步，有限且 >= 1； post trace 的时间常数
    :type tau_post: float
    :param f_pre: pre 分支的权重调制函数
    :type f_pre: Callable
    :param f_post: post 分支的权重调制函数
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``，其中 ``delta_w`` 的形状与 ``layer.weight`` 相同
    :rtype: tuple[Union[float, torch.Tensor, None], Union[float, torch.Tensor, None], torch.Tensor]

    :raises ValueError: 参数、活动或状态形状违反支持范围。

    ----

    .. _stdp_multi_step-en:

    * **English**

    Activity must use the weight's floating dtype and device; outputs match them.
    Current spikes enter traces before the update, so simultaneous-spike cancellation
    depends on modulation functions. Tensor input states are not mutated in place.

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
    :param tau_pre: In time steps, finite and >= 1;  Time constant of the pre trace
    :type tau_pre: float
    :param tau_post: In time steps, finite and >= 1;  Time constant of the post trace
    :type tau_post: float
    :param f_pre: Weight modulation function for the pre branch
    :type f_pre: Callable
    :param f_post: Weight modulation function for the post branch
    :type f_post: Callable
    :return: ``(trace_pre, trace_post, delta_w)``, where ``delta_w`` has the same shape as ``layer.weight``
    :rtype: tuple[Union[float, torch.Tensor, None], Union[float, torch.Tensor, None], torch.Tensor]
    :raises ValueError: Parameters or activity/state shapes violate the supported range.
    """
    functional_learning._check_time_constants(tau_pre=tau_pre, tau_post=tau_post)
    if isinstance(layer, nn.Linear):
        stdp_single_step = stdp_linear_single_step
    elif isinstance(layer, nn.Conv1d):
        stdp_single_step = stdp_conv1d_single_step
    elif isinstance(layer, nn.Conv2d):
        stdp_single_step = stdp_conv2d_single_step
    else:
        raise NotImplementedError(layer)

    ndim = 3 if isinstance(layer, nn.Linear) else layer.weight.ndim + 1
    if in_spike.ndim != ndim or out_spike.ndim != ndim:
        raise ValueError("Multi-step spikes require a time dimension.")
    if in_spike.shape[0] != out_spike.shape[0]:
        raise ValueError("Input and output time lengths differ.")
    if (
        in_spike.shape[1] != out_spike.shape[1]
        or in_spike.shape[2] != layer.weight.shape[1]
        or out_spike.shape[2] != layer.weight.shape[0]
    ):
        raise ValueError("Sequence shapes do not match weight or batch dimensions.")
    if in_spike.shape[0] == 0:
        return trace_pre, trace_post, torch.zeros_like(layer.weight)

    delta_w = torch.zeros_like(layer.weight)
    for t in range(in_spike.shape[0]):
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
        synapse: Union[nn.Conv1d, nn.Conv2d, nn.Linear],
        sn: nn.Module,
        tau_pre: float,
        tau_post: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.__init__-cn>` | :ref:`English <STDPLearner.__init__-en>`

        ----

        .. _STDPLearner.__init__-cn:

        * **中文**

        局部突触可塑性学习器，支持单步及多步 Linear、Conv1d、Conv2d；卷积
        groups=1、dilation=1。显式监控指定突触的输入和指定模块的输出，不推断相邻神经元。只更新 weight，不更新
        bias。状态由输入初始化；reset() 清空状态和记录，不修改权重和启停状态；close() 永久释放钩子。

        :param step_mode: 单步为 "s"；多步为 "m"。
        :type step_mode: str
        :param synapse: 待更新突触，使用浮点权重。
        :type synapse: Union[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]
        :param sn: 输出实际 post 脉冲的模块；活动应与权重使用相同 dtype/device。
        :type sn: torch.nn.Module
        :param tau_pre: 迹时间常数，单位为时间步，有限且 >= 1；使用 1-1/tau 衰减。
        :type tau_pre: float
        :param tau_post: 迹时间常数，单位为时间步，有限且 >= 1；使用 1-1/tau 衰减。
        :type tau_post: float
        :param f_pre: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: 时间常数或 step_mode 不合法。
        :raises NotImplementedError: 突触类型或卷积配置不受支持。

        ----

        .. _STDPLearner.__init__-en:

        * **English**

        Local synaptic plasticity learner supporting single/multi-step Linear, Conv1d
        and Conv2d; convolution groups=1 and dilation=1. Explicitly monitors the
        supplied synapse input and module output; no adjacent neuron is inferred.
        Updates weight, not bias. States materialize from activity. reset() clears
        states/records without changing weights or enablement; close() permanently
        releases hooks.

        :param step_mode: "s" for single-step; "m" for multi-step.
        :type step_mode: str
        :param synapse: Synapse to update, with floating weights.
        :type synapse: Union[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]
        :param sn: Module producing actual post spikes; activity should match weight dtype/device.
        :type sn: torch.nn.Module
        :param tau_pre: Trace time constant in time steps, finite and >= 1; decay 1-1/tau.
        :type tau_pre: float
        :param tau_post: Trace time constant in time steps, finite and >= 1; decay 1-1/tau.
        :type tau_post: float
        :param f_pre: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: Invalid time constants or step_mode.
        :raises NotImplementedError: Unsupported synapse type or convolution configuration.
        """

        functional_learning._check_time_constants(
            tau_pre=tau_pre, tau_post=tau_post, exponential=False
        )
        if step_mode not in ("s", "m"):
            raise ValueError("Unsupported step_mode: " + str(step_mode))
        if not isinstance(synapse, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            raise NotImplementedError(
                "Unsupported plasticity synapse: " + str(type(synapse))
            )
        if isinstance(synapse, (nn.Conv1d, nn.Conv2d)):
            if any(d != 1 for d in synapse.dilation):
                raise NotImplementedError(
                    "STDP with dilation != 1 has not been implemented."
                )
            if synapse.groups != 1:
                raise NotImplementedError(
                    "STDP with groups != 1 has not been implemented."
                )
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(
            synapse, function_on_input=torch.Tensor.detach
        )
        self.out_spike_monitor = monitor.OutputMonitor(
            sn, function_on_output=torch.Tensor.detach
        )

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.reset-cn>` | :ref:`English <STDPLearner.reset-en>`

        ----

        .. _STDPLearner.reset-cn:

        * **中文**

        清空所有学习状态和活动记录，不修改权重、梯度、记录启停状态或已关闭状态。


        ----

        .. _STDPLearner.reset-en:

        * **English**

        Clears all learning states and activity records without changing weights,
        gradients, recording enablement or closed status.
        """
        super().reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.disable-cn>` | :ref:`English <STDPLearner.disable-en>`

        ----

        .. _STDPLearner.disable-cn:

        * **中文**

        停止后续记录，不清空已有状态或记录，不移除钩子。


        ----

        .. _STDPLearner.disable-en:

        * **English**

        Stops subsequent recording without clearing states/records or removing hooks.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.enable-cn>` | :ref:`English <STDPLearner.enable-en>`

        ----

        .. _STDPLearner.enable-cn:

        * **中文**

        启用后续记录，不清空已有状态或记录；已关闭的学习器不能重新启用。

        :raises RuntimeError: 学习器已关闭。

        ----

        .. _STDPLearner.enable-en:

        * **English**

        Enables subsequent recording without clearing states/records; a closed learner
        cannot be re-enabled.

        :raises RuntimeError: The learner is closed.
        """

        _check_open(self)
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    @torch.no_grad()
    def step(self, on_grad: bool = True, scale: float = 1.0) -> Optional[torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.step-cn>` | :ref:`English <STDPLearner.step-en>`

        ----

        .. _STDPLearner.step-cn:

        * **中文**

        在 no_grad 下计算并消费活动记录，保留学习状态；没有记录则返回 None，不推进状态。on_grad=True 将负的缩放更新量加到
        weight.grad，返回 None；否则返回缩放更新量。两种形式均不直接修改权重。更新量 shape/dtype/device
        与权重一致，无计算图。记录数量/形状错误在推进前失败；调制函数异常后需修正并 reset()。按
        batch、卷积空间和时间求和，不取平均；多步按固定权重累计，不等价于逐步改权重。

        :param on_grad: 默认 True，叠加到 weight.grad；False 只返回更新量。
        :type on_grad: bool
        :param scale: 更新量乘数，默认 1；不改变 trace 的推进。
        :type scale: float
        :return: on_grad=False 且有记录时返回更新量，否则 None。
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: 已关闭、记录数量不一致，或 reward 学习器积压多对记录。
        :raises ValueError: 活动、状态或奖励形状不合法。

        ----

        .. _STDPLearner.step-en:

        * **English**

        Computes under no_grad and consumes activity records, retaining learning states.
        With no records, returns None without advancing state. on_grad=True adds the
        negative scaled update to weight.grad and returns None; otherwise returns the
        scaled update. Neither form changes weights directly. Updates match the weight
        shape/dtype/device and have no graph. Count/shape errors fail before
        advancement; correct modulation failures and reset(). Sums over batch,
        convolution space and time without averaging; multi-step fixed-weight
        accumulation is not equivalent to updating weights each step.

        :param on_grad: True by default: add to weight.grad; False: only return the update.
        :type on_grad: bool
        :param scale: Update multiplier, 1 by default; does not change trace advancement.
        :type scale: float
        :return: Update if on_grad=False and records exist; otherwise None.
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: Closed, mismatched counts, or multiple pending reward activity pairs.
        :raises ValueError: Invalid activity, state or reward shape.
        """

        _check_open(self)
        length = len(self.in_spike_monitor.records)
        if length != len(self.out_spike_monitor.records):
            raise RuntimeError("Input and output monitor record counts differ.")
        if length == 0:
            return None
        if length > 1:
            _check_records(self)
        delta_w = None

        if self.step_mode == "m":
            stdp_f = stdp_multi_step
        elif self.step_mode != "s":
            raise ValueError(self.step_mode)
        elif isinstance(self.synapse, nn.Linear):
            stdp_f = stdp_linear_single_step
        elif isinstance(self.synapse, nn.Conv2d):
            stdp_f = stdp_conv2d_single_step
        elif isinstance(self.synapse, nn.Conv1d):
            stdp_f = stdp_conv1d_single_step
        else:
            raise NotImplementedError(self.synapse)

        for i in range(length):
            in_spike = self.in_spike_monitor.records[i]
            out_spike = self.out_spike_monitor.records[i]

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

            delta_w = dw if delta_w is None else delta_w + dw

        if on_grad:
            _accumulate_weight_gradient(self.synapse, delta_w)
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()
        if not on_grad:
            return delta_w

    def close(self) -> None:
        r"""
        **API Language** - :ref:`中文 <STDPLearner.close-cn>` | :ref:`English <STDPLearner.close-en>`

        ----

        .. _STDPLearner.close-cn:

        * **中文**

        永久移除钩子，清空活动记录和所有学习状态，不修改权重或梯度。重复调用安全；关闭后 enable() 和 step() 抛出
        RuntimeError，继续学习需重新创建学习器。


        ----

        .. _STDPLearner.close-en:

        * **English**

        Permanently removes hooks and clears records and all learning states without
        changing weights or gradients. Idempotent; enable() and step() raise
        RuntimeError afterwards. Create a new learner to resume.
        """
        self.in_spike_monitor.close()
        self.out_spike_monitor.close()
        self.reset()


class MSTDPLearner(base.MemoryModule):
    def __init__(
        self,
        step_mode: str,
        synapse: nn.Linear,
        sn: nn.Module,
        tau_pre: float,
        tau_post: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.__init__-cn>` | :ref:`English <MSTDPLearner.__init__-en>`

        ----

        .. _MSTDPLearner.__init__-cn:

        * **中文**

        局部突触可塑性学习器，支持单步 Linear，输入/输出为 [N,I]/[N,O]。显式监控指定突触的输入和指定模块的输出，不推断相邻神经元。只更新
        weight，不更新 bias。状态由输入初始化；reset() 清空状态和记录，不修改权重和启停状态；close() 永久释放钩子。

        :param step_mode: 单步为 "s"；不支持多步。
        :type step_mode: str
        :param synapse: 待更新突触，使用浮点权重。
        :type synapse: torch.nn.Linear
        :param sn: 输出实际 post 脉冲的模块；活动应与权重使用相同 dtype/device。
        :type sn: torch.nn.Module
        :param tau_pre: 迹时间常数，单位为时间步，有限且 > 0；使用 exp(-1/tau) 衰减。
        :type tau_pre: float
        :param tau_post: 迹时间常数，单位为时间步，有限且 > 0；使用 exp(-1/tau) 衰减。
        :type tau_post: float
        :param f_pre: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: 时间常数或 step_mode 不合法。
        :raises NotImplementedError: 突触类型或卷积配置不受支持。

        ----

        .. _MSTDPLearner.__init__-en:

        * **English**

        Local synaptic plasticity learner supporting single-step Linear with [N,I]/[N,O]
        spikes. Explicitly monitors the supplied synapse input and module output; no
        adjacent neuron is inferred. Updates weight, not bias. States materialize from
        activity. reset() clears states/records without changing weights or enablement;
        close() permanently releases hooks.

        :param step_mode: "s" for single-step; multi-step is unsupported.
        :type step_mode: str
        :param synapse: Synapse to update, with floating weights.
        :type synapse: torch.nn.Linear
        :param sn: Module producing actual post spikes; activity should match weight dtype/device.
        :type sn: torch.nn.Module
        :param tau_pre: Trace time constant in time steps, finite and > 0; decay exp(-1/tau).
        :type tau_pre: float
        :param tau_post: Trace time constant in time steps, finite and > 0; decay exp(-1/tau).
        :type tau_post: float
        :param f_pre: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: Invalid time constants or step_mode.
        :raises NotImplementedError: Unsupported synapse type or convolution configuration.
        """

        functional_learning._check_time_constants(
            tau_pre=tau_pre, tau_post=tau_post, exponential=True
        )
        if step_mode != "s":
            raise ValueError("Unsupported step_mode: " + str(step_mode))
        if not isinstance(synapse, nn.Linear):
            raise NotImplementedError(
                "Unsupported plasticity synapse: " + str(type(synapse))
            )
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(
            synapse, function_on_input=torch.Tensor.detach
        )
        self.out_spike_monitor = monitor.OutputMonitor(
            sn, function_on_output=torch.Tensor.detach
        )

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)
        self.register_memory("eligibility", None)

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.reset-cn>` | :ref:`English <MSTDPLearner.reset-en>`

        ----

        .. _MSTDPLearner.reset-cn:

        * **中文**

        清空所有学习状态和活动记录，不修改权重、梯度、记录启停状态或已关闭状态。


        ----

        .. _MSTDPLearner.reset-en:

        * **English**

        Clears all learning states and activity records without changing weights,
        gradients, recording enablement or closed status.
        """
        super().reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.disable-cn>` | :ref:`English <MSTDPLearner.disable-en>`

        ----

        .. _MSTDPLearner.disable-cn:

        * **中文**

        停止后续记录，不清空已有状态或记录，不移除钩子。


        ----

        .. _MSTDPLearner.disable-en:

        * **English**

        Stops subsequent recording without clearing states/records or removing hooks.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.enable-cn>` | :ref:`English <MSTDPLearner.enable-en>`

        ----

        .. _MSTDPLearner.enable-cn:

        * **中文**

        启用后续记录，不清空已有状态或记录；已关闭的学习器不能重新启用。

        :raises RuntimeError: 学习器已关闭。

        ----

        .. _MSTDPLearner.enable-en:

        * **English**

        Enables subsequent recording without clearing states/records; a closed learner
        cannot be re-enabled.

        :raises RuntimeError: The learner is closed.
        """

        _check_open(self)
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    @torch.no_grad()
    def step(
        self,
        reward: Union[float, torch.Tensor],
        on_grad: bool = True,
        scale: float = 1.0,
    ) -> Optional[torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.step-cn>` | :ref:`English <MSTDPLearner.step-en>`

        ----

        .. _MSTDPLearner.step-cn:

        * **中文**

        在 no_grad 下计算并消费活动记录，保留学习状态；没有记录则返回 None，不推进状态。on_grad=True 将负的缩放更新量加到
        weight.grad，返回 None；否则返回缩放更新量。两种形式均不直接修改权重。更新量 shape/dtype/device
        与权重一致，无计算图。记录数量/形状错误在推进前失败；调制函数异常后需修正并 reset()。每次恰好一对活动。reward[t] 作用于
        eligibility[t-1]；第一步更新为零。最终活动后的奖励需后续前向推进。按 batch 求和，更换 batch 大小前 reset()。

        :param reward: 标量或 [N] 奖励；标量广播到每个样本。
        :type reward: Union[float, torch.Tensor]
        :param on_grad: 默认 True，叠加到 weight.grad；False 只返回更新量。
        :type on_grad: bool
        :param scale: 更新量乘数，默认 1；不改变 trace 的推进。
        :type scale: float
        :return: on_grad=False 且有记录时返回更新量，否则 None。
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: 已关闭、记录数量不一致，或 reward 学习器积压多对记录。
        :raises ValueError: 活动、状态或奖励形状不合法。

        ----

        .. _MSTDPLearner.step-en:

        * **English**

        Computes under no_grad and consumes activity records, retaining learning states.
        With no records, returns None without advancing state. on_grad=True adds the
        negative scaled update to weight.grad and returns None; otherwise returns the
        scaled update. Neither form changes weights directly. Updates match the weight
        shape/dtype/device and have no graph. Count/shape errors fail before
        advancement; correct modulation failures and reset(). Exactly one pair per call.
        reward[t] acts on eligibility[t-1]; the first update is zero. Reward after final
        activity needs a subsequent forward step. Sums over batch; reset() before
        changing batch size.

        :param reward: Scalar or [N] reward; scalars apply to every sample.
        :type reward: Union[float, torch.Tensor]
        :param on_grad: True by default: add to weight.grad; False: only return the update.
        :type on_grad: bool
        :param scale: Update multiplier, 1 by default; does not change trace advancement.
        :type scale: float
        :return: Update if on_grad=False and records exist; otherwise None.
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: Closed, mismatched counts, or multiple pending reward activity pairs.
        :raises ValueError: Invalid activity, state or reward shape.
        """

        _check_open(self)
        count = len(self.in_spike_monitor.records)
        if count != len(self.out_spike_monitor.records):
            raise RuntimeError("Input and output monitor record counts differ.")
        if count == 0:
            return None
        if count != 1:
            raise RuntimeError(
                "Reward learners require exactly one activity pair per step."
            )
        if self.step_mode != "s":
            raise ValueError("Reward learners only support single-step mode.")
        pre = self.in_spike_monitor.records[0]
        post = self.out_spike_monitor.records[0]
        trace_pre, trace_post, eligibility = mstdp_linear_single_step(
            self.synapse,
            pre,
            post,
            self.trace_pre,
            self.trace_post,
            self.tau_pre,
            self.tau_post,
            self.f_pre,
            self.f_post,
        )
        reward = torch.as_tensor(
            reward, device=self.synapse.weight.device, dtype=self.synapse.weight.dtype
        ).detach()
        batch_size = pre.shape[0]
        if reward.ndim != 0 and reward.shape != (batch_size,):
            raise ValueError("MSTDP reward must be scalar or shaped [N].")
        if self.eligibility is not None and self.eligibility.shape != eligibility.shape:
            raise ValueError(
                "Eligibility shape differs; reset before changing batch size."
            )
        previous = (
            torch.zeros_like(eligibility)
            if self.eligibility is None
            else self.eligibility
        )
        if previous.device != self.synapse.weight.device:
            raise ValueError("Eligibility and weight must be on the same device.")
        delta_w = (reward.reshape(-1, 1, 1) * previous).sum(0)
        self.trace_pre, self.trace_post = trace_pre, trace_post
        self.eligibility = eligibility
        if scale != 1.0:
            delta_w *= scale
        if on_grad:
            _accumulate_weight_gradient(self.synapse, delta_w)
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()
        if not on_grad:
            return delta_w

    def close(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPLearner.close-cn>` | :ref:`English <MSTDPLearner.close-en>`

        ----

        .. _MSTDPLearner.close-cn:

        * **中文**

        永久移除钩子，清空活动记录和所有学习状态，不修改权重或梯度。重复调用安全；关闭后 enable() 和 step() 抛出
        RuntimeError，继续学习需重新创建学习器。


        ----

        .. _MSTDPLearner.close-en:

        * **English**

        Permanently removes hooks and clears records and all learning states without
        changing weights or gradients. Idempotent; enable() and step() raise
        RuntimeError afterwards. Create a new learner to resume.
        """
        self.in_spike_monitor.close()
        self.out_spike_monitor.close()
        self.reset()


class MSTDPETLearner(base.MemoryModule):
    def __init__(
        self,
        step_mode: str,
        synapse: nn.Linear,
        sn: nn.Module,
        tau_pre: float,
        tau_post: float,
        tau_trace: float,
        f_pre: Callable = lambda x: x,
        f_post: Callable = lambda x: x,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.__init__-cn>` | :ref:`English <MSTDPETLearner.__init__-en>`

        ----

        .. _MSTDPETLearner.__init__-cn:

        * **中文**

        局部突触可塑性学习器，支持单步 Linear，输入/输出为 [I]/[O]。显式监控指定突触的输入和指定模块的输出，不推断相邻神经元。只更新
        weight，不更新 bias。状态由输入初始化；reset() 清空状态和记录，不修改权重和启停状态；close() 永久释放钩子。

        :param step_mode: 单步为 "s"；不支持多步。
        :type step_mode: str
        :param synapse: 待更新突触，使用浮点权重。
        :type synapse: torch.nn.Linear
        :param sn: 输出实际 post 脉冲的模块；活动应与权重使用相同 dtype/device。
        :type sn: torch.nn.Module
        :param tau_pre: 迹时间常数，单位为时间步，有限且 > 0；使用 exp(-1/tau) 衰减。
        :type tau_pre: float
        :param tau_post: 迹时间常数，单位为时间步，有限且 > 0；使用 exp(-1/tau) 衰减。
        :type tau_post: float
        :param tau_trace: eligibility trace 时间常数，单位为时间步，有限且 > 0。
        :type tau_trace: float
        :param f_pre: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: 权重调制函数，默认恒等映射；返回与权重形状兼容的张量。
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: 时间常数或 step_mode 不合法。
        :raises NotImplementedError: 突触类型或卷积配置不受支持。

        ----

        .. _MSTDPETLearner.__init__-en:

        * **English**

        Local synaptic plasticity learner supporting single-step Linear with [I]/[O]
        spikes. Explicitly monitors the supplied synapse input and module output; no
        adjacent neuron is inferred. Updates weight, not bias. States materialize from
        activity. reset() clears states/records without changing weights or enablement;
        close() permanently releases hooks.

        :param step_mode: "s" for single-step; multi-step is unsupported.
        :type step_mode: str
        :param synapse: Synapse to update, with floating weights.
        :type synapse: torch.nn.Linear
        :param sn: Module producing actual post spikes; activity should match weight dtype/device.
        :type sn: torch.nn.Module
        :param tau_pre: Trace time constant in time steps, finite and > 0; decay exp(-1/tau).
        :type tau_pre: float
        :param tau_post: Trace time constant in time steps, finite and > 0; decay exp(-1/tau).
        :type tau_post: float
        :param tau_trace: Eligibility-trace time constant in time steps, finite and > 0.
        :type tau_trace: float
        :param f_pre: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_pre: Callable[[torch.Tensor], torch.Tensor]
        :param f_post: Weight modulation, identity by default; returns a tensor compatible with the
            weight shape.
        :type f_post: Callable[[torch.Tensor], torch.Tensor]
        :raises ValueError: Invalid time constants or step_mode.
        :raises NotImplementedError: Unsupported synapse type or convolution configuration.
        """

        functional_learning._check_time_constants(
            tau_pre=tau_pre, tau_post=tau_post, exponential=True
        )
        if step_mode != "s":
            raise ValueError("Unsupported step_mode: " + str(step_mode))
        if not isinstance(synapse, nn.Linear):
            raise NotImplementedError(
                "Unsupported plasticity synapse: " + str(type(synapse))
            )
        functional_learning._check_time_constants(tau_trace=tau_trace, exponential=True)
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_trace = tau_trace
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(
            synapse, function_on_input=torch.Tensor.detach
        )
        self.out_spike_monitor = monitor.OutputMonitor(
            sn, function_on_output=torch.Tensor.detach
        )

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)
        self.register_memory("trace_e", None)
        self.register_memory("eligibility", None)

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.reset-cn>` | :ref:`English <MSTDPETLearner.reset-en>`

        ----

        .. _MSTDPETLearner.reset-cn:

        * **中文**

        清空所有学习状态和活动记录，不修改权重、梯度、记录启停状态或已关闭状态。


        ----

        .. _MSTDPETLearner.reset-en:

        * **English**

        Clears all learning states and activity records without changing weights,
        gradients, recording enablement or closed status.
        """
        super().reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.disable-cn>` | :ref:`English <MSTDPETLearner.disable-en>`

        ----

        .. _MSTDPETLearner.disable-cn:

        * **中文**

        停止后续记录，不清空已有状态或记录，不移除钩子。


        ----

        .. _MSTDPETLearner.disable-en:

        * **English**

        Stops subsequent recording without clearing states/records or removing hooks.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.enable-cn>` | :ref:`English <MSTDPETLearner.enable-en>`

        ----

        .. _MSTDPETLearner.enable-cn:

        * **中文**

        启用后续记录，不清空已有状态或记录；已关闭的学习器不能重新启用。

        :raises RuntimeError: 学习器已关闭。

        ----

        .. _MSTDPETLearner.enable-en:

        * **English**

        Enables subsequent recording without clearing states/records; a closed learner
        cannot be re-enabled.

        :raises RuntimeError: The learner is closed.
        """

        _check_open(self)
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    @torch.no_grad()
    def step(
        self,
        reward: Union[float, torch.Tensor],
        on_grad: bool = True,
        scale: float = 1.0,
    ) -> Optional[torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.step-cn>` | :ref:`English <MSTDPETLearner.step-en>`

        ----

        .. _MSTDPETLearner.step-cn:

        * **中文**

        在 no_grad 下计算并消费活动记录，保留学习状态；没有记录则返回 None，不推进状态。on_grad=True 将负的缩放更新量加到
        weight.grad，返回 None；否则返回缩放更新量。两种形式均不直接修改权重。更新量 shape/dtype/device
        与权重一致，无计算图。记录数量/形状错误在推进前失败；调制函数异常后需修正并 reset()。每次恰好一对活动。reward[t] 作用于
        eligibility[t-1]；第一步更新为零。最终活动后的奖励需后续前向推进。先计算
        trace_e[t]=exp(-1/tau_trace)*trace_e[t-1]+eligibility[t-1]/tau_trace，再乘
        reward[t]，最后计算当前 eligibility。

        :param reward: 标量奖励。
        :type reward: Union[float, torch.Tensor]
        :param on_grad: 默认 True，叠加到 weight.grad；False 只返回更新量。
        :type on_grad: bool
        :param scale: 更新量乘数，默认 1；不改变 trace 的推进。
        :type scale: float
        :return: on_grad=False 且有记录时返回更新量，否则 None。
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: 已关闭、记录数量不一致，或 reward 学习器积压多对记录。
        :raises ValueError: 活动、状态或奖励形状不合法。

        ----

        .. _MSTDPETLearner.step-en:

        * **English**

        Computes under no_grad and consumes activity records, retaining learning states.
        With no records, returns None without advancing state. on_grad=True adds the
        negative scaled update to weight.grad and returns None; otherwise returns the
        scaled update. Neither form changes weights directly. Updates match the weight
        shape/dtype/device and have no graph. Count/shape errors fail before
        advancement; correct modulation failures and reset(). Exactly one pair per call.
        reward[t] acts on eligibility[t-1]; the first update is zero. Reward after final
        activity needs a subsequent forward step. First computes
        trace_e[t]=exp(-1/tau_trace)*trace_e[t-1]+eligibility[t-1]/tau_trace, multiplies
        by reward[t], then computes current eligibility.

        :param reward: Scalar reward.
        :type reward: Union[float, torch.Tensor]
        :param on_grad: True by default: add to weight.grad; False: only return the update.
        :type on_grad: bool
        :param scale: Update multiplier, 1 by default; does not change trace advancement.
        :type scale: float
        :return: Update if on_grad=False and records exist; otherwise None.
        :rtype: Optional[torch.Tensor]
        :raises RuntimeError: Closed, mismatched counts, or multiple pending reward activity pairs.
        :raises ValueError: Invalid activity, state or reward shape.
        """

        _check_open(self)
        count = len(self.in_spike_monitor.records)
        if count != len(self.out_spike_monitor.records):
            raise RuntimeError("Input and output monitor record counts differ.")
        if count == 0:
            return None
        if count != 1:
            raise RuntimeError(
                "Reward learners require exactly one activity pair per step."
            )
        if self.step_mode != "s":
            raise ValueError("Reward learners only support single-step mode.")
        pre = self.in_spike_monitor.records[0]
        post = self.out_spike_monitor.records[0]
        trace_pre, trace_post, eligibility = mstdpet_linear_single_step(
            self.synapse,
            pre,
            post,
            self.trace_pre,
            self.trace_post,
            self.tau_pre,
            self.tau_post,
            self.f_pre,
            self.f_post,
        )
        reward = torch.as_tensor(
            reward, device=self.synapse.weight.device, dtype=self.synapse.weight.dtype
        ).detach()
        if reward.ndim != 0:
            raise ValueError("MSTDPET reward must be scalar.")
        for state in (self.eligibility, self.trace_e):
            if state is not None and state.shape != self.synapse.weight.shape:
                raise ValueError("Eligibility state shape differs from weight.")
        previous = (
            torch.zeros_like(eligibility)
            if self.eligibility is None
            else self.eligibility
        )
        trace_e = (
            torch.zeros_like(eligibility) if self.trace_e is None else self.trace_e
        )
        delta_w, trace_e = functional_learning.mstdpet_reward_step(
            reward,
            previous,
            trace_e,
            tau_trace=self.tau_trace,
        )
        self.trace_e = trace_e
        self.trace_pre, self.trace_post = trace_pre, trace_post
        self.eligibility = eligibility
        if scale != 1.0:
            delta_w *= scale
        if on_grad:
            _accumulate_weight_gradient(self.synapse, delta_w)
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()
        if not on_grad:
            return delta_w

    def close(self) -> None:
        r"""
        **API Language** - :ref:`中文 <MSTDPETLearner.close-cn>` | :ref:`English <MSTDPETLearner.close-en>`

        ----

        .. _MSTDPETLearner.close-cn:

        * **中文**

        永久移除钩子，清空活动记录和所有学习状态，不修改权重或梯度。重复调用安全；关闭后 enable() 和 step() 抛出
        RuntimeError，继续学习需重新创建学习器。


        ----

        .. _MSTDPETLearner.close-en:

        * **English**

        Permanently removes hooks and clears records and all learning states without
        changing weights or gradients. Idempotent; enable() and step() raise
        RuntimeError afterwards. Create a new learner to resume.
        """
        self.in_spike_monitor.close()
        self.out_spike_monitor.close()
        self.reset()
