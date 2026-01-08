import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import base, functional


__all__ = [
    'SynapseFilter',
    'PrintShapeModule',
    'VotingLayer',
    'Delay',
]


class SynapseFilter(base.MemoryModule):

    def __init__(self, tau=100.0, learnable=False, step_mode='s'):
        """
        * :ref:`API in English <LowPassSynapse.__init__-en>`

        .. _LowPassSynapse.__init__-cn:

        :param tau: time 突触上电流衰减的时间常数

        :param learnable: 时间常数在训练过程中是否是可学习的。若为 ``True``，则 ``tau`` 会被设定成时间常数的初始值

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        具有滤波性质的突触。突触的输出电流满足，当没有脉冲输入时，输出电流指数衰减：

        .. math::
            \\tau \\frac{\\mathrm{d} I(t)}{\\mathrm{d} t} = - I(t)

        当有新脉冲输入时，输出电流自增1：

        .. math::
            I(t) = I(t) + 1

        记输入脉冲为 :math:`S(t)`，则离散化后，统一的电流更新方程为：

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \\frac{1}{\\tau} I(t-1) + S(t)

        这种突触能将输入脉冲进行平滑，简单的示例代码和输出结果：

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label='in spike')
            pyplot.xlabel('t')
            pyplot.ylabel('spike')
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label='out i')
            pyplot.xlabel('t')
            pyplot.ylabel('i')
            pyplot.legend()
            pyplot.show()

        .. image:: ../_static/API/activation_based/layer/SynapseFilter.png

        输出电流不仅取决于当前时刻的输入，还取决于之前的输入，使得该突触具有了一定的记忆能力。

        这种突触偶有使用，例如：

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        另一种视角是将其视为一种输入为脉冲，并输出其电压的LIF神经元。并且该神经元的发放阈值为 :math:`+\\infty` 。

        神经元最后累计的电压值一定程度上反映了该神经元在整个仿真过程中接收脉冲的数量，从而替代了传统的直接对输出脉冲计数（即发放频率）来表示神经元活跃程度的方法。因此通常用于最后一层，在以下文章中使用：

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        * :ref:`中文API <LowPassSynapse.__init__-cn>`

        .. _LowPassSynapse.__init__-en:

        :param tau: time constant that determines the decay rate of current in the synapse

        :param learnable: whether time constant is learnable during training. If ``True``, then ``tau`` will be the
            initial value of time constant

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        The synapse filter that can filter input current. The output current will decay when there is no input spike:

        .. math::
            \\tau \\frac{\\mathrm{d} I(t)}{\\mathrm{d} t} = - I(t)

        The output current will increase 1 when there is a new input spike:

        .. math::
            I(t) = I(t) + 1

        Denote the input spike as :math:`S(t)`, then the discrete current update equation is as followed:

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \\frac{1}{\\tau} I(t-1) + S(t)

        This synapse can smooth input. Here is the example and output:

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label='in spike')
            pyplot.xlabel('t')
            pyplot.ylabel('spike')
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label='out i')
            pyplot.xlabel('t')
            pyplot.ylabel('i')
            pyplot.legend()
            pyplot.show()

        .. image:: ../_static/API/activation_based/layer/SynapseFilter.png

        The output current is not only determined by the present input but also by the previous input, which makes this
        synapse have memory.

        This synapse is sometimes used, e.g.:

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        Another view is regarding this synapse as a LIF neuron with a :math:`+\\infty` threshold voltage.

        The final output of this synapse (or the final voltage of this LIF neuron) represents the accumulation of input
        spikes, which substitute for traditional firing rate that indicates the excitatory level. So, it can be used in
        the last layer of the network, e.g.:

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        """
        super().__init__()
        self.step_mode = step_mode
        self.learnable = learnable
        assert tau > 1
        if learnable:
            init_w = - math.log(tau - 1)
            self.w = nn.Parameter(torch.as_tensor(init_w))
        else:
            self.tau = tau

        self.register_memory('out_i', 0.)

    def extra_repr(self):
        if self.learnable:
            with torch.no_grad():
                tau = 1. / self.w.sigmoid()
        else:
            tau = self.tau

        return f'tau={tau}, learnable={self.learnable}, step_mode={self.step_mode}'

    @staticmethod
    @torch.jit.script
    def js_single_step_forward_learnable(x: torch.Tensor, w: torch.Tensor, out_i: torch.Tensor):
        inv_tau = w.sigmoid()
        out_i = out_i - (1. - x) * out_i * inv_tau + x
        return out_i

    @staticmethod
    @torch.jit.script
    def js_single_step_forward(x: torch.Tensor, tau: float, out_i: torch.Tensor):
        inv_tau = 1. / tau
        out_i = out_i - (1. - x) * out_i * inv_tau + x
        return out_i

    def single_step_forward(self, x: Tensor):
        if isinstance(self.out_i, float):
            out_i_init = self.out_i
            self.out_i = torch.zeros_like(x.data)
            if out_i_init != 0.:
                torch.fill_(self.out_i, out_i_init)

        if self.learnable:
            self.out_i = self.js_single_step_forward_learnable(x, self.w, self.out_i)
        else:
            self.out_i = self.js_single_step_forward(x, self.tau, self.out_i)
        return self.out_i


class PrintShapeModule(nn.Module):
    def __init__(self, ext_str='PrintShapeModule'):
        """
        * :ref:`API in English <PrintModule.__init__-en>`

        .. _PrintModule.__init__-cn:

        :param ext_str: 额外打印的字符串
        :type ext_str: str

        只打印 ``ext_str`` 和输入的 ``shape``，不进行任何操作的网络层，可以用于debug。

        * :ref:`中文API <PrintModule.__init__-cn>`

        .. _PrintModule.__init__-en:

        :param ext_str: extra strings for printing
        :type ext_str: str

        This layer will not do any operation but print ``ext_str`` and the shape of input, which can be used for debugging.

        """
        super().__init__()
        self.ext_str = ext_str

    def forward(self, x: Tensor):
        print(self.ext_str, x.shape)
        return x


class VotingLayer(nn.Module, base.StepModule):
    def __init__(self, voting_size: int = 10, step_mode='s'):
        """
        * :ref:`API in English <VotingLayer-en>`

        .. _VotingLayer-cn:

        :param voting_size: 决定一个类别的投票数量
        :type voting_size: int
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        投票层，对 ``shape = [..., C * voting_size]`` 的输入在最后一维上做 ``kernel_size = voting_size, stride = voting_size`` 的平均池化

        * :ref:`中文 API <VotingLayer-cn>`

        .. _VotingLayer-en:

        :param voting_size: the voting numbers for determine a class
        :type voting_size: int
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Applies average pooling with ``kernel_size = voting_size, stride = voting_size`` on the last dimension of the input with ``shape = [..., C * voting_size]``

        """
        super().__init__()
        self.voting_size = voting_size
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f'voting_size={self.voting_size}, step_mode={self.step_mode}'

    def single_step_forward(self, x: torch.Tensor):
        return F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, self.single_step_forward)


class Delay(base.MemoryModule):
    def __init__(self, delay_steps: int, step_mode='s'):
        """
        * :ref:`API in English <DelayModule.__init__-en>`

        .. _DelayModule.__init__-cn:

        :param delay_steps: 延迟的时间步数
        :type delay_steps: int
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        延迟层，可以用来延迟输入，使得 ``y[t] = x[t - delay_steps]``。缺失的数据用0填充。

        代码示例：

        .. code-block:: python

            delay_layer = Delay(delay=1, step_mode='m')
            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay_layer(x)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

        输出为：

        .. code-block:: bash

            x=
            tensor([[0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])


        * :ref:`中文API <DelayModule.__init__-cn>`

        .. _DelayModule.__init__-en:

        :param delay_steps: the number of delayed time-steps
        :type delay_steps: int
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        A delay layer that can delay inputs and makes ``y[t] = x[t - delay_steps]``. The nonexistent data will be regarded as 0.

        Codes example:

        .. code-block:: python

            delay_layer = Delay(delay=1, step_mode='m')
            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay_layer(x)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

        The outputs are:

        .. code-block:: bash

            x=
            tensor([[0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])
        """
        super().__init__()
        assert delay_steps >= 0 and isinstance(delay_steps, int)
        self._delay_steps = delay_steps
        self.step_mode = step_mode

        self.register_memory('queue', [])
        # used for single step mode

    @property
    def delay_steps(self):
        return self._delay_steps

    def single_step_forward(self, x: torch.Tensor):
        self.queue.append(x)
        if self.queue.__len__() > self.delay_steps:
            return self.queue.pop(0)
        else:
            return torch.zeros_like(x)

    def multi_step_forward(self, x_seq: torch.Tensor):
        return functional.delay(x_seq, self.delay_steps)
