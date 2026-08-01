import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import base, functional


__all__ = [
    "SynapseFilter",
    "PrintShapeModule",
    "VotingLayer",
    "Delay",
    "SpikeCountToBinary",
    "TemporalBinSum",
]


class SynapseFilter(base.MemoryModule):
    def __init__(self, tau=100.0, learnable=False, step_mode="s"):
        r"""
        **API Language** - :ref:`中文 <SynapseFilter.__init__-cn>` | :ref:`English <SynapseFilter.__init__-en>`

        ----

        .. _SynapseFilter.__init__-cn:

        * **中文**

        具有滤波性质的突触。突触的输出电流满足，当没有脉冲输入时，输出电流指数衰减：

        .. math::
            \tau \frac{\mathrm{d} I(t)}{\mathrm{d} t} = - I(t)

        当有新脉冲输入时，输出电流自增1：

        .. math::
            I(t) = I(t) + 1

        记输入脉冲为 :math:`S(t)`，则离散化后，统一的电流更新方程为：

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \frac{1}{\tau} I(t-1) + S(t)

        输出电流不仅取决于当前时刻的输入，还取决于之前的输入，使得该突触具有了一定的记忆能力。

        这种突触偶有使用，例如：

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        另一种视角是将其视为一种输入为脉冲，并输出其电压的LIF神经元。并且该神经元的发放阈值为 :math:`+\infty` 。

        神经元最后累计的电压值一定程度上反映了该神经元在整个仿真过程中接收脉冲的数量，从而替代了传统的直接对输出脉冲计数（即发放频率）来表示神经元活跃程度的方法。因此通常用于最后一层，在以下文章中使用：

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        :param tau: time 突触上电流衰减的时间常数
        :type tau: float

        :param learnable: 时间常数在训练过程中是否是可学习的。若为 ``True``，则 ``tau`` 会被设定成时间常数的初始值
        :type learnable: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _SynapseFilter.__init__-en:

        * **English**

        The synapse filter that can filter input current. The output current will decay when there is no input spike:

        .. math::
            \tau \frac{\mathrm{d} I(t)}{\mathrm{d} t} = - I(t)

        The output current will increase 1 when there is a new input spike:

        .. math::
            I(t) = I(t) + 1

        Denote the input spike as :math:`S(t)`, then the discrete current update equation is as followed:

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \frac{1}{\tau} I(t-1) + S(t)

        The output current is not only determined by the present input but also by the previous input, which makes this
        synapse have memory.

        This synapse is sometimes used, e.g.:

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        Another view is regarding this synapse as a LIF neuron with a :math:`+\infty` threshold voltage.

        The final output of this synapse (or the final voltage of this LIF neuron) represents the accumulation of input
        spikes, which substitute for traditional firing rate that indicates the excitatory level. So, it can be used in
        the last layer of the network, e.g.:

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        :param tau: time constant that determines the decay rate of current in the synapse
        :type tau: float

        :param learnable: whether time constant is learnable during training. If ``True``, then ``tau`` will be the initial value of time constant
        :type learnable: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label="in spike")
            pyplot.xlabel("t")
            pyplot.ylabel("spike")
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label="out i")
            pyplot.xlabel("t")
            pyplot.ylabel("i")
            pyplot.legend()
            pyplot.show()

        .. image:: ../_static/API/activation_based/layer/SynapseFilter.png
        """
        super().__init__()
        self.step_mode = step_mode
        self.learnable = learnable
        assert tau > 1
        if learnable:
            init_w = -math.log(tau - 1)
            self.w = nn.Parameter(torch.as_tensor(init_w))
        else:
            self.tau = tau

        self.register_memory("out_i", 0.0)

    def extra_repr(self):
        if self.learnable:
            with torch.no_grad():
                tau = 1.0 / self.w.sigmoid()
        else:
            tau = self.tau

        return f"tau={tau}, learnable={self.learnable}, step_mode={self.step_mode}"

    def single_step_forward(self, x: Tensor):
        if isinstance(self.out_i, float):
            self.out_i = torch.full_like(x, self.out_i)
        inv_tau = self.w.sigmoid() if self.learnable else 1.0 / self.tau
        self.out_i = functional.synapse_filter_step(x, self.out_i, inv_tau)
        return self.out_i


class PrintShapeModule(nn.Module):
    def __init__(self, ext_str="PrintShapeModule"):
        r"""
        **API Language** - :ref:`中文 <PrintShapeModule.__init__-cn>` | :ref:`English <PrintShapeModule.__init__-en>`

        ----

        .. _PrintShapeModule.__init__-cn:

        * **中文**

        只打印 ``ext_str`` 和输入的 ``shape``，不进行任何操作的网络层，可以用于debug。

        :param ext_str: 额外打印的字符串
        :type ext_str: str

        ----

        .. _PrintShapeModule.__init__-en:

        * **English**

        This layer will not do any operation but print ``ext_str`` and the shape of input, which can be used for debugging.

        :param ext_str: extra strings for printing
        :type ext_str: str
        """
        super().__init__()
        self.ext_str = ext_str

    def forward(self, x: Tensor):
        print(self.ext_str, x.shape)
        return x


class VotingLayer(nn.Module, base.StepModule):
    def __init__(self, voting_size: int = 10, step_mode="s"):
        r"""
        **API Language** - :ref:`中文 <VotingLayer.__init__-cn>` | :ref:`English <VotingLayer.__init__-en>`

        ----

        .. _VotingLayer.__init__-cn:

        * **中文**

        投票层，对 ``shape = [..., C * voting_size]`` 的输入在最后一维上做 ``kernel_size = voting_size, stride = voting_size`` 的平均池化

        :param voting_size: 决定一个类别的投票数量
        :type voting_size: int

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _VotingLayer.__init__-en:

        * **English**

        Applies average pooling with ``kernel_size = voting_size, stride = voting_size`` on the last dimension of the input with ``shape = [..., C * voting_size]``

        :param voting_size: the voting numbers for determine a class
        :type voting_size: int

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.voting_size = voting_size
        self.step_mode = step_mode

    def extra_repr(self):
        return (
            super().extra_repr()
            + f"voting_size={self.voting_size}, step_mode={self.step_mode}"
        )

    def single_step_forward(self, x: torch.Tensor):
        return F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(
            1
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return functional.seq_to_ann_forward(x, self.single_step_forward)


class Delay(base.MemoryModule):
    def __init__(self, delay_steps: int, step_mode="s"):
        r"""
        **API Language** - :ref:`中文 <Delay.__init__-cn>` | :ref:`English <Delay.__init__-en>`

        ----

        .. _Delay.__init__-cn:

        * **中文**

        延迟层，可以用来延迟输入，使得 ``y[t] = x[t - delay_steps]``。缺失的数据用0填充。

        :param delay_steps: 延迟的时间步数
        :type delay_steps: int

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        * :ref:`中文API <Delay.__init__-cn>`

        ----

        .. _Delay.__init__-en:

        * **English**

        A delay layer that can delay inputs and makes ``y[t] = x[t - delay_steps]``. The nonexistent data will be regarded as 0.

        :param delay_steps: the number of delayed time-steps
        :type delay_steps: int

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            delay_layer = Delay(delay_steps=1, step_mode="m")
            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay_layer(x)
            print("x=")
            print(x)
            print("y=")
            print(y)
            y.sum().backward()
            print("x.grad=")
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

        self.register_memory("queue", [])
        # used for single step mode

    @property
    def delay_steps(self):
        return self._delay_steps

    def single_step_forward(self, x: torch.Tensor):
        y, queue_next = functional.delay_step(x, tuple(self.queue), self.delay_steps)
        self.queue[:] = queue_next
        return y

    def multi_step_forward(self, x_seq: torch.Tensor):
        return functional.delay(x_seq, self.delay_steps)


class SpikeCountToBinary(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        num_slots: int,
        always_convert: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <SpikeCountToBinary.__init__-cn>` | :ref:`English <SpikeCountToBinary.__init__-en>`

        ----

        .. _SpikeCountToBinary.__init__-cn:

        * **中文**

        将每个非负整数脉冲计数展开为 ``num_slots`` 个二值事件槽。该模块仅支持
        多步模式，时间维固定为第 0 维。默认训练模式返回原输入、推理模式执行
        展开；设置 ``always_convert=True`` 后始终展开。

        展开激活时使用硬二值前向和直通估计。每个事件槽对输入计数的代理导数为
        ``1 / num_slots``，因此反向传播会把同一 bin 的槽梯度取平均。若展开后
        使用同一个无偏置权重层并由 :class:`TemporalBinSum` 求和，则前向、输入
        梯度和权重梯度均与直接对整数计数执行权重运算等价。偏置、归一化和
        非线性应放在 :class:`TemporalBinSum` 之后。

        :param num_slots: 每个逻辑时间步包含的二值事件槽数量
        :type num_slots: int
        :param always_convert: 为 ``True`` 时始终展开；为 ``False`` 时仅在推理
            模式展开，默认为 ``False``
        :type always_convert: bool
        :raises ValueError: 当 ``num_slots < 1`` 时抛出

        ----

        .. _SpikeCountToBinary.__init__-en:

        * **English**

        Expand each non-negative integer spike count into ``num_slots`` binary
        event slots. This module only supports multi-step mode and fixes the time
        dimension at dimension 0. By default, training mode returns the input
        unchanged and evaluation mode performs the expansion. Set
        ``always_convert=True`` to always expand.

        Active expansion uses hard binary values in forward and a straight-through
        estimator whose surrogate derivative is ``1 / num_slots`` for each event
        slot. Backward therefore averages the slot gradients in each bin. When the
        same bias-free weight layer processes every slot and
        :class:`TemporalBinSum` sums the results, the forward values, input
        gradients, and weight gradients match a direct weight operation on the
        integer counts. Bias, normalization, and nonlinear operations must follow
        :class:`TemporalBinSum`.

        :param num_slots: Number of binary event slots per logical timestep
        :type num_slots: int
        :param always_convert: ``True`` to always expand, or ``False`` to expand
            only in evaluation mode; defaults to ``False``
        :type always_convert: bool
        :raises ValueError: If ``num_slots < 1``
        """
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1.")
        self.num_slots = num_slots
        self.always_convert = always_convert
        self.step_mode = "m"

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SpikeCountToBinary.forward-cn>` | :ref:`English <SpikeCountToBinary.forward-en>`

        ----

        .. _SpikeCountToBinary.forward-cn:

        * **中文**

        :param x_seq: 形状为 ``[T, N, *]``、``T > 0`` 且取值位于
            ``[0, num_slots]`` 的浮点整数值脉冲计数
        :type x_seq: torch.Tensor
        :return: 转换激活时返回形状为 ``[T * num_slots, N, *]`` 的二值脉冲；
            否则原样返回 ``x_seq``。输出保持输入的 dtype 和 device
        :rtype: torch.Tensor
        :raises ValueError: 转换激活且 ``x_seq`` 不满足 ``[T, N, *]``、
            ``T > 0`` 时抛出

        ----

        .. _SpikeCountToBinary.forward-en:

        * **English**

        :param x_seq: Floating-point integer-valued spike counts in
            ``[0, num_slots]`` with shape ``[T, N, *]`` and ``T > 0``
        :type x_seq: torch.Tensor
        :return: Binary spikes with shape ``[T * num_slots, N, *]`` when
            conversion is active; otherwise ``x_seq`` unchanged. The output keeps
            the input dtype and device
        :rtype: torch.Tensor
        :raises ValueError: If active conversion receives an input that does not
            satisfy shape ``[T, N, *]`` and ``T > 0``
        """
        if not self.always_convert and self.training:
            return x_seq

        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("x_seq must have shape [T, N, *] with T > 0.")

        counts = x_seq.unsqueeze(1)
        slots = torch.arange(self.num_slots, device=x_seq.device).view(
            1, self.num_slots, *((1,) * (x_seq.ndim - 1))
        )
        hard_spikes = (slots < counts).to(x_seq.dtype)
        proxy = counts / self.num_slots
        return (hard_spikes + (proxy - proxy.detach())).flatten(0, 1)


class TemporalBinSum(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        num_slots: int,
        always_convert: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <TemporalBinSum.__init__-cn>` | :ref:`English <TemporalBinSum.__init__-en>`

        ----

        .. _TemporalBinSum.__init__-cn:

        * **中文**

        沿第 0 维将每 ``num_slots`` 个连续时间步求和。最后一个 bin 不完整时
        在时间维末尾补零。该模块仅支持多步模式。默认训练模式返回原输入、
        推理模式执行求和；设置 ``always_convert=True`` 后始终求和。

        与 :class:`SpikeCountToBinary` 配对时，两者必须使用相同的
        ``num_slots`` 和 ``always_convert``。展开与求和之间的权重层必须禁用
        bias；归一化和非线性应在求和后执行一次。

        :param num_slots: 每个输出逻辑时间步聚合的物理事件槽数量
        :type num_slots: int
        :param always_convert: 为 ``True`` 时始终聚合；为 ``False`` 时仅在推理
            模式聚合，默认为 ``False``
        :type always_convert: bool
        :raises ValueError: 当 ``num_slots < 1`` 时抛出

        ----

        .. _TemporalBinSum.__init__-en:

        * **English**

        Sum each ``num_slots`` consecutive timesteps along dimension 0. An
        incomplete final bin is zero-padded at the end of the time dimension.
        This module only supports multi-step mode. By default, training mode
        returns the input unchanged and evaluation mode performs the reduction.
        Set ``always_convert=True`` to always reduce.

        When paired with :class:`SpikeCountToBinary`, both modules must use the
        same ``num_slots`` and ``always_convert``. The weight layer between
        expansion and reduction must disable bias; normalization and nonlinear
        operations must run once after the reduction.

        :param num_slots: Number of physical event slots aggregated into each
            output logical timestep
        :type num_slots: int
        :param always_convert: ``True`` to always reduce, or ``False`` to reduce
            only in evaluation mode; defaults to ``False``
        :type always_convert: bool
        :raises ValueError: If ``num_slots < 1``
        """
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1.")
        self.num_slots = num_slots
        self.always_convert = always_convert
        self.step_mode = "m"

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <TemporalBinSum.forward-cn>` | :ref:`English <TemporalBinSum.forward-en>`

        ----

        .. _TemporalBinSum.forward-cn:

        * **中文**

        :param x_seq: 形状为 ``[T, N, *]``、``T > 0`` 的浮点物理事件槽序列
        :type x_seq: torch.Tensor
        :return: 聚合激活时返回形状为
            ``[ceil(T / num_slots), N, *]`` 的序列；否则原样返回 ``x_seq``。
            输出保持输入的 dtype 和 device
        :rtype: torch.Tensor
        :raises ValueError: 聚合激活且 ``x_seq`` 不满足 ``[T, N, *]``、
            ``T > 0`` 时抛出

        ----

        .. _TemporalBinSum.forward-en:

        * **English**

        :param x_seq: Floating-point physical event-slot sequence with shape
            ``[T, N, *]`` and ``T > 0``
        :type x_seq: torch.Tensor
        :return: A sequence with shape ``[ceil(T / num_slots), N, *]`` when
            aggregation is active; otherwise ``x_seq`` unchanged. The output keeps
            the input dtype and device
        :rtype: torch.Tensor
        :raises ValueError: If active aggregation receives an input that does not
            satisfy shape ``[T, N, *]`` and ``T > 0``
        """
        if not self.always_convert and self.training:
            return x_seq

        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("x_seq must have shape [T, N, *] with T > 0.")

        remainder = x_seq.shape[0] % self.num_slots
        if remainder:
            padding = x_seq.new_zeros((self.num_slots - remainder, *x_seq.shape[1:]))
            x_seq = torch.cat((x_seq, padding))
        return x_seq.reshape(-1, self.num_slots, *x_seq.shape[1:]).sum(dim=1)
