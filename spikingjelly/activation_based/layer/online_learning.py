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
    **API Language:**
    :ref:`中文 <ReplaceforGrad-cn>` | :ref:`English <ReplaceforGrad-en>`

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


class GradwithTrace(nn.Module):
    def __init__(self, module):
        r"""
        **API Language:**
        :ref:`中文 <GradwithTrace.__init__-cn>` | :ref:`English <GradwithTrace.__init__-en>`

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
        :return: None
        :rtype: None
        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        r"""
        **API Language:**
        :ref:`中文 <GradwithTrace.forward-cn>` | :ref:`English <GradwithTrace.forward-en>`

        ----

        .. _GradwithTrace.forward-cn:
        * **中文**

        :param x: ``[spike, trace]``，其中 ``spike`` 用于前向值，``trace`` 用于梯度路径
        :type x: torch.Tensor
        :return: 包装模块的输出，前向值来自 ``spike``，反向梯度来自 ``trace``
        :rtype: torch.Tensor

        ----

        .. _GradwithTrace.forward-en:
        * **English**

        :param x: ``[spike, trace]`` where ``spike`` provides forward values and ``trace`` provides gradient paths
        :type x: torch.Tensor
        :return: Wrapped-module output with forward value from ``spike`` and backward gradient from ``trace``
        :rtype: torch.Tensor
        """
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]

        with torch.no_grad():
            out = self.module(spike).detach()

        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)

        x = ReplaceforGrad.apply(out_for_grad, out)

        return x


class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        r"""
        **API Language:**
        :ref:`中文 <SpikeTraceOp.__init__-cn>` | :ref:`English <SpikeTraceOp.__init__-en>`

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
        :return: None
        :rtype: None
        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        r"""
        **API Language:**
        :ref:`中文 <SpikeTraceOp.forward-cn>` | :ref:`English <SpikeTraceOp.forward-en>`

        ----

        .. _SpikeTraceOp.forward-cn:
        * **中文**

        :param x: ``[spike, trace]`` 输入对
        :type x: torch.Tensor
        :return: 对 ``spike`` 与 ``trace`` 施加相同算子后的结果 ``[spike, trace]``
        :rtype: list[torch.Tensor]

        ----

        .. _SpikeTraceOp.forward-en:
        * **English**

        :param x: ``[spike, trace]`` input pair
        :type x: torch.Tensor
        :return: ``[spike, trace]`` after applying the same operator to both
        :rtype: list[torch.Tensor]
        """
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]

        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)

        x = [spike, trace]

        return x


class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        r"""
        **API Language:**
        :ref:`中文 <OTTTSequential.__init__-cn>` | :ref:`English <OTTTSequential.__init__-en>`

        ----

        .. _OTTTSequential.__init__-cn:
        * **中文**

        用于 OTTT（Online Training Through Time）的顺序容器，扩展自 ``nn.Sequential``。
        在 ``forward`` 中，若输入为 ``[spike, trace]`` 列表形式，则自动将有参数的模块包装为 :class:`GradwithTrace`，
        将无参数的模块包装为 :class:`SpikeTraceOp`，以实现在线训练中的梯度传递。

        :param args: 需要顺序执行的模块
        :type args: nn.Module
        :return: ``None``
        :rtype: None

        ----

        .. _OTTTSequential.__init__-en:
        * **English**

        Sequential container for OTTT (Online Training Through Time), extending ``nn.Sequential``.
        During ``forward``, if the input is a ``[spike, trace]`` list, modules with parameters are
        automatically wrapped by :class:`GradwithTrace`, while parameter-free modules are wrapped by
        :class:`SpikeTraceOp`, enabling gradient propagation for online training.

        :param args: Modules to be executed sequentially
        :type args: nn.Module
        :return: ``None``
        :rtype: None
        """
        super().__init__(*args)

    def forward(self, input):
        r"""
        **API Language:**
        :ref:`中文 <OTTTSequential.forward-cn>` | :ref:`English <OTTTSequential.forward-en>`

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
        for module in self:
            if not isinstance(input, list):
                input = module(input)
            else:
                if len(list(module.parameters())) > 0:  # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                else:  # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
                input = module(input)
        return input
