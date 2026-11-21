import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "ReplaceforGrad",
    "GradwithTrace",
    "SpikeTraceOp",
    "OTTTSequential",
]


class ReplaceforGrad(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <ReplaceforGrad-cn>` | :ref:`English <ReplaceforGrad-en>`

    ----

    .. _ReplaceforGrad-cn:

    * **中文**

    在 OTTT 在线训练中用于替换前向值与梯度路径的自定义自动求导函数。
    前向返回 ``x_r``，反向时梯度同时传播给 ``x`` 和 ``x_r``。

    ----

    .. _ReplaceforGrad-en:

    * **English**

    Custom autograd Function for replacing forward values and gradient paths in OTTT online training.
    Forward returns ``x_r``, backward passes gradients to both ``x`` and ``x_r``.
    """

    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


def _trace_forward(module: nn.Module, trace: Tensor) -> Tensor:
    # Use batch statistics for the auxiliary training pass without committing
    # running statistics again. Leave unrelated buffers and lazy branches alone.
    batchnorms = [
        bn
        for bn in module.modules()
        if isinstance(bn, nn.modules.batchnorm._BatchNorm)
        and bn.training
        and bn.track_running_stats
        and not isinstance(bn.running_mean, nn.parameter.UninitializedBuffer)
        and not isinstance(bn.running_var, nn.parameter.UninitializedBuffer)
    ]
    try:
        for bn in batchnorms:
            bn.track_running_stats = False
        return module(trace)
    finally:
        for bn in batchnorms:
            bn.track_running_stats = True


def _grad_with_trace(module: nn.Module, x):
    spike, trace = x
    with torch.no_grad():
        out = module(spike)
    out_for_grad = _trace_forward(module, ReplaceforGrad.apply(spike, trace))
    return ReplaceforGrad.apply(out_for_grad, out)


def _apply_to_spike_and_trace(module: nn.Module, x):
    spike, trace = x
    spike = module(spike)
    with torch.no_grad():
        trace = _trace_forward(module, trace)
    return [spike, trace]


class GradwithTrace(nn.Module):
    def __init__(self, module):
        r"""
        **API Language** - :ref:`中文 <GradwithTrace.__init__-cn>` | :ref:`English <GradwithTrace.__init__-en>`

        ----

        .. _GradwithTrace.__init__-cn:

        * **中文**

        用于随时间在线训练时，根据神经元的迹计算梯度
        出处：'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        :param module: 需要包装的模块
        :type module: torch.nn.Module

        ----

        .. _GradwithTrace.__init__-en:

        * **English**

        Used for online training through time, calculate gradients by the traces of neurons
        Reference: 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        :param module: the module that requires wrapping
        :type module: torch.nn.Module

        ----
        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        r"""
        **API Language** - :ref:`中文 <GradwithTrace.forward-cn>` | :ref:`English <GradwithTrace.forward-en>`

        ----

        .. _GradwithTrace.forward-cn:

        * **中文**

        对于已初始化的 BatchNorm，只有脉冲前向更新运行统计量；trace 训练前向
        仍使用 batch 统计量，但暂时停止跟踪运行统计量。其他 buffer 和 Python 状态
        保留原有副作用，不被隔离。此保护仅覆盖前向调用；包装模块内部的
        activation checkpoint 必须自行为反向重计算提供统计量保护。

        :param x: ``[spike, trace]``，其中 ``spike`` 用于前向值，``trace`` 用于梯度路径
        :type x: torch.Tensor
        :return: 包装模块的输出，前向值来自 ``spike``，反向梯度来自 ``trace``
        :rtype: torch.Tensor

        ----

        .. _GradwithTrace.forward-en:

        * **English**

        For initialized BatchNorm modules, only the spike forward updates running
        statistics. The trace training forward still uses batch statistics, with
        running-statistic tracking temporarily disabled. Other buffers and Python
        state retain their original side effects and are not isolated. This guard
        covers the forward call only; activation checkpoints inside the wrapped
        module must provide their own running-statistic guard for recomputation.

        :param x: ``[spike, trace]`` where ``spike`` provides forward values and ``trace`` provides gradient paths
        :type x: torch.Tensor
        :return: Wrapped-module output with forward value from ``spike`` and backward gradient from ``trace``
        :rtype: torch.Tensor
        """
        return _grad_with_trace(self.module, x)


class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        r"""
        **API Language** - :ref:`中文 <SpikeTraceOp.__init__-cn>` | :ref:`English <SpikeTraceOp.__init__-en>`

        ----

        .. _SpikeTraceOp.__init__-cn:

        * **中文**

        对脉冲和迹进行相同的运算，如Dropout，AvgPool等

        :param module: 需要包装的模块
        :type module: torch.nn.Module

        ----

        .. _SpikeTraceOp.__init__-en:

        * **English**

        perform the same operations for spike and trace, such as Dropout, Avgpool, etc.

        :param module: the module that requires wrapping
        :type module: torch.nn.Module

        ----
        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        r"""
        **API Language** - :ref:`中文 <SpikeTraceOp.forward-cn>` | :ref:`English <SpikeTraceOp.forward-en>`

        ----

        .. _SpikeTraceOp.forward-cn:

        * **中文**

        对于已初始化的 BatchNorm，只有脉冲前向更新运行统计量；trace 训练前向
        仍使用 batch 统计量，但暂时停止跟踪运行统计量。其他 buffer 和 Python 状态
        保留原有副作用，不被隔离。此保护仅覆盖前向调用；包装模块内部的
        activation checkpoint 必须自行为反向重计算提供统计量保护。

        :param x: ``[spike, trace]`` 输入对
        :type x: torch.Tensor
        :return: 对 ``spike`` 与 ``trace`` 施加相同算子后的结果 ``[spike, trace]``
        :rtype: list[torch.Tensor]

        ----

        .. _SpikeTraceOp.forward-en:

        * **English**

        For initialized BatchNorm modules, only the spike forward updates running
        statistics. The trace training forward still uses batch statistics, with
        running-statistic tracking temporarily disabled. Other buffers and Python
        state retain their original side effects and are not isolated. This guard
        covers the forward call only; activation checkpoints inside the wrapped
        module must provide their own running-statistic guard for recomputation.

        :param x: ``[spike, trace]`` input pair
        :type x: torch.Tensor
        :return: ``[spike, trace]`` after applying the same operator to both
        :rtype: list[torch.Tensor]
        """
        return _apply_to_spike_and_trace(self.module, x)


class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        r"""
        **API Language** - :ref:`中文 <OTTTSequential.__init__-cn>` | :ref:`English <OTTTSequential.__init__-en>`

        ----

        .. _OTTTSequential.__init__-cn:

        * **中文**

        用于 OTTT（Online Training Through Time）的顺序容器，扩展自 ``nn.Sequential``。
        在 ``forward`` 中，若输入为 ``[spike, trace]`` 列表形式，则：若下一模块为
        :class:`~spikingjelly.activation_based.neuron.online_learning.OTTTLIFNode`，
        只对 ``spike`` 调用一次并用其返回的 ``[spike, trace]`` 作为新的状态对；
        否则将有参数的模块包装为 :class:`GradwithTrace`，将无参数的模块包装为
        :class:`SpikeTraceOp`。

        :param args: 需要顺序执行的模块
        :type args: nn.Module

        ----

        .. _OTTTSequential.__init__-en:

        * **English**

        Sequential container for OTTT (Online Training Through Time), extending ``nn.Sequential``.
        During ``forward``, if the input is a ``[spike, trace]`` list, an
        ``OTTTLIFNode`` is called once on the spike tensor and its returned
        ``[spike, trace]`` pair replaces the state. Other modules with parameters
        are wrapped by :class:`GradwithTrace`; parameter-free modules use
        :class:`SpikeTraceOp`.

        :param args: Modules to be executed sequentially
        :type args: nn.Module
        """
        super().__init__(*args)

    def forward(self, input):
        r"""
        **API Language** - :ref:`中文 <OTTTSequential.forward-cn>` | :ref:`English <OTTTSequential.forward-en>`

        ----

        .. _OTTTSequential.forward-cn:

        * **中文**

        :param input: 常规张量输入，或 ``[spike, trace]`` 形式输入
        :type input: Union[torch.Tensor, list[torch.Tensor]]
        :return: 顺序执行后的输出
        :rtype: Union[torch.Tensor, list[torch.Tensor]]

        ----

        .. _OTTTSequential.forward-en:

        * **English**

        :param input: Regular tensor input, or ``[spike, trace]`` style input
        :type input: Union[torch.Tensor, list[torch.Tensor]]
        :return: Output after sequential execution
        :rtype: Union[torch.Tensor, list[torch.Tensor]]
        """
        # Import at call time to avoid the layer <-> neuron import cycle.
        from ..neuron.online_learning import OTTTLIFNode

        for module in self:
            if isinstance(input, list):
                if isinstance(module, OTTTLIFNode):
                    input = module(input[0])
                elif next(module.parameters(), None) is None:
                    input = _apply_to_spike_and_trace(module, input)
                else:
                    input = _grad_with_trace(module, input)
                continue
            input = module(input)
        return input
