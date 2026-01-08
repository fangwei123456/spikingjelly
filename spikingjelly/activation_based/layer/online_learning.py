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
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class GradwithTrace(nn.Module):
    def __init__(self, module):
        """
        * :ref:`API in English <GradwithTrace-en>`

        .. _GradwithTrace-cn:

        :param module: 需要包装的模块

        用于随时间在线训练时，根据神经元的迹计算梯度
        出处：'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _GradwithTrace-en:

        :param module: the module that requires wrapping

        Used for online training through time, calculate gradients by the traces of neurons
        Reference: 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
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
        """
        * :ref:`API in English <SpikeTraceOp-en>`

        .. _SpikeTraceOp-cn:

        :param module: 需要包装的模块

        对脉冲和迹进行相同的运算，如Dropout，AvgPool等

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _SpikeTraceOp-en:

        :param module: the module that requires wrapping

        perform the same operations for spike and trace, such as Dropout, Avgpool, etc.

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]

        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)

        x = [spike, trace]

        return x


class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(input, list):
                input = module(input)
            else:
                if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                else: # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
                input = module(input)
        return input
