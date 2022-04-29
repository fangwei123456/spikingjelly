import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import base, functional
from torch import Tensor
from torch.nn.common_types import _size_2_t
from typing import Callable
from torch.nn.modules.batchnorm import _BatchNorm


class NeuNorm(base.MemoryModule):
    def __init__(self, in_channels, height, width, k=0.9, shared_across_channels=False):
        """
        * :ref:`API in English <NeuNorm.__init__-en>`

        .. _NeuNorm.__init__-cn:

        :param in_channels: 输入数据的通道数

        :param height: 输入数据的宽

        :param width: 输入数据的高

        :param k: 动量项系数

        :param shared_across_channels: 可学习的权重 ``w`` 是否在通道这一维度上共享。设置为 ``True`` 可以大幅度节省内存

        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_ 中提出\\
        的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如：

        ``Conv2d -> LIF -> NeuNorm``

        要求输入的尺寸是 ``[batch_size, in_channels, height, width]``。

        ``in_channels`` 是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        ``k`` 是动量项系数，相当于论文中的 :math:`k_{\\tau 2}`。

        论文中的 :math:`\\frac{v}{F}` 会根据 :math:`k_{\\tau 2} + vF = 1` 自动算出。

        * :ref:`中文API <NeuNorm.__init__-cn>`

        .. _NeuNorm.__init__-en:

        :param in_channels: channels of input

        :param height: height of input

        :param width: height of width

        :param k: momentum factor

        :param shared_across_channels: whether the learnable parameter ``w`` is shared over channel dim. If set ``True``,
            the consumption of memory can decrease largely

        The NeuNorm layer is proposed in `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_.

        It should be placed after spiking neurons behind convolution layer, e.g.,

        ``Conv2d -> LIF -> NeuNorm``

        The input should be a 4-D tensor with ``shape = [batch_size, in_channels, height, width]``.

        ``in_channels`` is the channels of input，which is :math:`F` in the paper.

        ``k`` is the momentum factor，which is :math:`k_{\\tau 2}` in the paper.

        :math:`\\frac{v}{F}` will be calculated by :math:`k_{\\tau 2} + vF = 1` autonomously.

        """
        super().__init__()
        self.register_memory('x', 0.)
        self.k0 = k
        self.k1 = (1. - self.k0) / in_channels ** 2
        if shared_across_channels:
            self.w = nn.Parameter(Tensor(1, height, width))
        else:
            self.w = nn.Parameter(Tensor(in_channels, height, width))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, in_spikes: Tensor):
        self.x = self.k0 * self.x + self.k1 * in_spikes.sum(dim=1,
                                                            keepdim=True)  # x.shape = [batch_size, 1, height, width]
        return in_spikes - self.w * self.x

    def extra_repr(self) -> str:
        return f'shape={self.w.shape}'


class DCT(nn.Module):
    def __init__(self, kernel_size):
        """
        * :ref:`API in English <DCT.__init__-en>`

        .. _DCT.__init__-cn:

        :param kernel_size: 进行分块DCT变换的块大小

        将输入的 ``shape = [*, W, H]`` 的数据进行分块DCT变换的层，``*`` 表示任意额外添加的维度。变换只在最后2维进行，要求 ``W`` 和 ``H`` 都能\\
        整除 ``kernel_size``。

        ``DCT`` 是 ``AXAT`` 的一种特例。

        * :ref:`中文API <DCT.__init__-cn>`

        .. _DCT.__init__-en:

        :param kernel_size: block size for DCT transform

        Apply Discrete Cosine Transform on input with ``shape = [*, W, H]``, where ``*`` means any number of additional dimensions.
        DCT will only applied in the last two dimensions. ``W`` and ``H`` should be divisible by ``kernel_size``.

        Note that ``DCT`` is a special case of ``AXAT``.
        """
        super().__init__()
        self.kernel = torch.zeros(size=[kernel_size, kernel_size])
        for i in range(0, kernel_size):
            for j in range(kernel_size):
                if i == 0:
                    self.kernel[i][j] = math.sqrt(1 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)
                else:
                    self.kernel[i][j] = math.sqrt(2 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)

    def forward(self, x: Tensor):
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        x_shape = x.shape
        x = x.view(-1, x_shape[-2], x_shape[-1])
        ret = torch.zeros_like(x)
        for i in range(0, x_shape[-2], self.kernel.shape[0]):
            for j in range(0, x_shape[-1], self.kernel.shape[0]):
                ret[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]] \
                    = self.kernel.matmul(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]]).matmul(
                    self.kernel.t())
        return ret.view(x_shape)


class AXAT(nn.Module):
    def __init__(self, in_features, out_features):
        """
        * :ref:`API in English <AXAT.__init__-en>`

        .. _AXAT.__init__-cn:

        :param in_features: 输入数据的最后2维的尺寸。输入应该是 ``shape = [*, in_features, in_features]``
        :param out_features: 输出数据的最后2维的尺寸。输出数据为 ``shape = [*, out_features, out_features]``

        对输入数据 :math:`X` 在最后2维进行线性变换 :math:`AXA^{T}` 的操作，:math:`A` 是 ``shape = [out_features, in_features]`` 的矩阵。

        将输入的数据看作是批量个 ``shape = [in_features, in_features]`` 的矩阵.

        * :ref:`中文API <AXAT.__init__-cn>`

        .. _AXAT.__init__-en:

        :param in_features: feature number of input at last two dimensions. The input should be ``shape = [*, in_features, in_features]``

        :param out_features: feature number of output at last two dimensions. The output will be ``shape = [*, out_features, out_features]``

        Apply :math:`AXA^{T}` transform on input :math:`X` at the last two dimensions. :math:`A` is a tensor with ``shape = [out_features, in_features]``.

        The input will be regarded as a batch of tensors with ``shape = [in_features, in_features]``.
        """
        super().__init__()
        self.A = nn.Parameter(Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor):
        x_shape = list(x.shape)
        x = x.view(-1, x_shape[-2], x_shape[-1])
        x = self.A.matmul(x).matmul(self.A.t())
        x_shape[-1] = x.shape[-1]
        x_shape[-2] = x.shape[-2]
        return x.view(x_shape)


class Dropout(base.MemoryModule):
    def __init__(self, p=0.5):
        """
        * :ref:`API in English <Dropout.__init__-en>`

        .. _Dropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float

        与 ``torch.nn.Dropout`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        .. tip::

            这种Dropout最早由 `Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures
            <https://arxiv.org/abs/1903.06379>`_ 一文进行详细论述：

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.

        * :ref:`中文API <Dropout.__init__-cn>`

        .. _Dropout.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float

        This layer is almost same with ``torch.nn.Dropout``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        .. admonition:: Tip
            :class: tip

            This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
            Network Architectures <https://arxiv.org/abs/1903.06379>`_:

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.
        """
        super().__init__()
        assert 0 <= p < 1
        self.register_memory('mask', None)
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def create_mask(self, x: Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x


class Dropout2d(Dropout):
    def __init__(self, p=0.2):
        """
        * :ref:`API in English <Dropout2d.__init__-en>`

        .. _Dropout2d.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float

        与 ``torch.nn.Dropout2d`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        关于SNN中Dropout的更多信息，参见 :ref:`layer.Dropout <Dropout.__init__-cn>`。

        * :ref:`中文API <Dropout2d.__init__-cn>`

        .. _Dropout2d.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float

        This layer is almost same with ``torch.nn.Dropout2d``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        For more information about Dropout in SNN, refer to :ref:`layer.Dropout <Dropout.__init__-en>`.
        """
        super().__init__(p)

    def create_mask(self, x: Tensor):
        self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)


class MultiStepDropout(Dropout):
    def __init__(self, p=0.5):
        """
        * :ref:`API in English <MultiStepDropout.__init__-en>`

        .. _MultiStepDropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float

        :class:`spikingjelly.clock_driven.layer.Dropout` 的多步版本。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepDropout.__init__-cn>`

        .. _MultiStepDropout.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float

        The multi-step version of :class:`spikingjelly.clock_driven.layer.Dropout`.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.
        """
        super().__init__(p)

    def forward(self, x_seq: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x_seq[0])

            return x_seq * self.mask
        else:
            return x_seq


class MultiStepDropout2d(Dropout2d):
    def __init__(self, p=0.5):
        """
        * :ref:`API in English <MultiStepDropout2d.__init__-en>`

        .. _MultiStepDropout2d.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float

        :class:`spikingjelly.clock_driven.layer.Dropout2d` 的多步版本。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepDropout2d.__init__-cn>`

        .. _MultiStepDropout2d.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float

        The multi-step version of :class:`spikingjelly.clock_driven.layer.Dropout2d`.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.
        """
        super().__init__(p)

    def forward(self, x_seq: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x_seq[0])

            return x_seq * self.mask
        else:
            return x_seq


class SynapseFilter(base.MemoryModule):
    def __init__(self, tau=100.0, learnable=False):
        """
        * :ref:`API in English <LowPassSynapse.__init__-en>`

        .. _LowPassSynapse.__init__-cn:

        :param tau: time 突触上电流衰减的时间常数

        :param learnable: 时间常数在训练过程中是否是可学习的。若为 ``True``，则 ``tau`` 会被设定成时间常数的初始值

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

        .. image:: ./_static/API/clock_driven/layer/SynapseFilter.png

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

        .. image:: ./_static/API/clock_driven/layer/SynapseFilter.png

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

        return f'tau={tau}, learnable={self.learnable}'

    def forward(self, in_spikes: Tensor):
        if self.learnable:
            inv_tau = self.w.sigmoid()
        else:
            inv_tau = 1. / self.tau

        self.out_i = self.out_i - (1 - in_spikes) * self.out_i * inv_tau + in_spikes

        return self.out_i


class ChannelsPool(nn.Module):
    def __init__(self, pool: nn.MaxPool1d or nn.AvgPool1d):
        """
        * :ref:`API in English <ChannelsPool.__init__-en>`

        .. _ChannelsPool.__init__-cn:

        :param pool: ``nn.MaxPool1d`` 或 ``nn.AvgPool1d``，池化层

        使用 ``pool`` 将输入的4-D数据在第1个维度上进行池化。

        示例代码：

        .. code-block:: python

            >>> cp = ChannelsPool(torch.nn.MaxPool1d(2, 2))
            >>> x = torch.rand(size=[2, 8, 4, 4])
            >>> y = cp(x)
            >>> y.shape
            torch.Size([2, 4, 4, 4])

        * :ref:`中文API <ChannelsPool.__init__-cn>`

        .. _ChannelsPool.__init__-en:

        :param pool: ``nn.MaxPool1d`` or ``nn.AvgPool1d``, the pool layer

        Use ``pool`` to pooling 4-D input at dimension 1.

        Examples:

        .. code-block:: python

            >>> cmp = ChannelsPool(torch.nn.MaxPool1d(2, 2))
            >>> x = torch.rand(size=[2, 8, 4, 4])
            >>> y = cp(x)
            >>> y.shape
            torch.Size([2, 4, 4, 4])
        """
        super().__init__()
        self.pool = pool

    def forward(self, x: Tensor):
        x_shape = x.shape
        return self.pool(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view((x_shape[0], -1) + x_shape[2:])


class DropConnectLinear(base.MemoryModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.5, samples_num: int = 1024,
                 invariant: bool = False, activation: None or nn.Module = nn.ReLU()) -> None:
        """
        * :ref:`API in English <DropConnectLinear.__init__-en>`

        .. _DropConnectLinear.__init__-cn:

        :param in_features: 每个输入样本的特征数
        :type in_features: int
        :param out_features: 每个输出样本的特征数
        :type out_features: int
        :param bias: 若为 ``False``，则本层不会有可学习的偏置项。
            默认为 ``True``
        :type bias: bool
        :param p: 每个连接被断开的概率。默认为0.5
        :type p: float
        :param samples_num: 在推理时，从高斯分布中采样的数据数量。默认为1024
        :type samples_num: int
        :param invariant: 若为 ``True``，线性层会在第一次执行前向传播时被按概率断开，断开后的线性层会保持不变，直到 ``reset()`` 函数
            被调用，线性层恢复为完全连接的状态。完全连接的线性层，调用 ``reset()`` 函数后的第一次前向传播时被重新按概率断开。 若为
            ``False``，在每一次前向传播时线性层都会被重新完全连接再按概率断开。 阅读 :ref:`layer.Dropout <Dropout.__init__-cn>` 以
            获得更多关于此参数的信息。
            默认为 ``False``
        :type invariant: bool
        :param activation: 在线性层后的激活层
        :type activation: None or nn.Module

        DropConnect，由 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
        一文提出。DropConnect与Dropout非常类似，区别在于DropConnect是以概率 ``p`` 断开连接，而Dropout是将输入以概率置0。

        .. Note::

            在使用DropConnect进行推理时，输出的tensor中的每个元素，都是先从高斯分布中采样，通过激活层激活，再在采样数量上进行平均得到的。
            详细的流程可以在 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
            一文中的 `Algorithm 2` 找到。激活层 ``activation`` 在中间的步骤起作用，因此我们将其作为模块的成员。

        * :ref:`中文API <DropConnectLinear.__init__-cn>`

        .. _DropConnectLinear.__init__-en:

        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        :type bias: bool
        :param p: probability of an connection to be zeroed. Default: 0.5
        :type p: float
        :param samples_num: number of samples drawn from the Gaussian during inference. Default: 1024
        :type samples_num: int
        :param invariant: If set to ``True``, the connections will be dropped at the first time of forward and the dropped
            connections will remain unchanged until ``reset()`` is called and the connections recovery to fully-connected
            status. Then the connections will be re-dropped at the first time of forward after ``reset()``. If set to
            ``False``, the connections will be re-dropped at every forward. See :ref:`layer.Dropout <Dropout.__init__-en>`
            for more information to understand this parameter. Default: ``False``
        :type invariant: bool
        :param activation: the activation layer after the linear layer
        :type activation: None or nn.Module

        DropConnect, which is proposed by `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_,
        is similar with Dropout but drop connections of a linear layer rather than the elements of the input tensor with
        probability ``p``.

        .. admonition:: Note
            :class: note

            When inference with DropConnect, every elements of the output tensor are sampled from a Gaussian distribution,
            activated by the activation layer and averaged over the sample number ``samples_num``.
            See `Algorithm 2` in `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
            for more details. Note that activation is an intermediate process. This is the reason why we include
            ``activation`` as a member variable of this module.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.p = p  # 置0的概率
        self.register_memory('dropped_w', None)
        if self.bias is not None:
            self.register_memory('dropped_b', None)

        self.samples_num = samples_num
        self.invariant = invariant
        self.activation = activation

    def reset_parameters(self) -> None:
        """
        * :ref:`API in English <DropConnectLinear.reset_parameters-en>`

        .. _DropConnectLinear.reset_parameters-cn:

        :return: None
        :rtype: None

        初始化模型中的可学习参数。

        * :ref:`中文API <DropConnectLinear.reset_parameters-cn>`

        .. _DropConnectLinear.reset_parameters-en:

        :return: None
        :rtype: None

        Initialize the learnable parameters of this module.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reset(self):
        """
        * :ref:`API in English <DropConnectLinear.reset-en>`

        .. _DropConnectLinear.reset-cn:

        :return: None
        :rtype: None

        将线性层重置为完全连接的状态，若 ``self.activation`` 也是一个有状态的层，则将其也重置。

        * :ref:`中文API <DropConnectLinear.reset-cn>`

        .. _DropConnectLinear.reset-en:

        :return: None
        :rtype: None

        Reset the linear layer to fully-connected status. If ``self.activation`` is also stateful, this function will
        also reset it.
        """
        super().reset()
        if hasattr(self.activation, 'reset'):
            self.activation.reset()

    def drop(self, batch_size: int):
        mask_w = (torch.rand_like(self.weight.unsqueeze(0).repeat([batch_size] + [1] * self.weight.dim())) > self.p)
        # self.dropped_w = mask_w.to(self.weight) * self.weight  # shape = [batch_size, out_features, in_features]
        self.dropped_w = self.weight * mask_w

        if self.bias is not None:
            mask_b = (torch.rand_like(self.bias.unsqueeze(0).repeat([batch_size] + [1] * self.bias.dim())) > self.p)
            # self.dropped_b = mask_b.to(self.bias) * self.bias
            self.dropped_b = self.bias * mask_b

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            if self.invariant:
                if self.dropped_w is None:
                    self.drop(input.shape[0])
            else:
                self.drop(input.shape[0])
            if self.bias is None:
                ret = torch.bmm(self.dropped_w, input.unsqueeze(-1)).squeeze(-1)
            else:
                ret = torch.bmm(self.dropped_w, input.unsqueeze(-1)).squeeze(-1) + self.dropped_b
            if self.activation is None:
                return ret
            else:
                return self.activation(ret)
        else:
            mu = (1 - self.p) * F.linear(input, self.weight, self.bias)  # shape = [batch_size, out_features]
            if self.bias is None:
                sigma2 = self.p * (1 - self.p) * F.linear(input.square(), self.weight.square())
            else:
                sigma2 = self.p * (1 - self.p) * F.linear(input.square(), self.weight.square(), self.bias.square())
            dis = torch.distributions.normal.Normal(mu, sigma2.sqrt())
            samples = dis.sample(torch.Size([self.samples_num]))

            if self.activation is None:
                ret = samples
            else:
                ret = self.activation(samples)
            return ret.mean(dim=0)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, p={self.p}, invariant={self.invariant}'


class MultiStepContainer(nn.Sequential):
    def __init__(self, *args):
        """
        * :ref:`API in English <MultiStepContainer.reset-en>`

        .. _MultiStepContainer.reset-cn:

        :param args: 单个或多个网络模块
        :type args: torch.nn.Module

        将单步模块包装成多步模块的包装器。


        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepContainer.reset-cn>`

        .. _MultiStepContainer.reset-en:

        :param args: one or many modules
        :type args: torch.nn.Module

        A container that wraps single-step modules to a multi-step modules.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.



        """
        super().__init__(*args)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: Tensor
        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: Tensor
        """

        return functional.multi_step_forward(x_seq, self)


class SeqToANNContainer(nn.Sequential):
    def __init__(self, *args):
        """
        * :ref:`API in English <SeqToANNContainer.__init__-en>`

        .. _SeqToANNContainer.__init__-cn:

        :param *args: 无状态的单个或多个ANN网络层

        包装无状态的ANN以处理序列数据的包装器。``shape=[T, batch_size, ...]`` 的输入会被拼接成 ``shape=[T * batch_size, ...]`` 再送入被包装的模块。输出结果会被再拆成 ``shape=[T, batch_size, ...]``。

        示例代码

        .. code-block:: python
            with torch.no_grad():
                T = 16
                batch_size = 8
                x = torch.rand([T, batch_size, 4])
                fc = SeqToANNContainer(nn.Linear(4, 2), nn.Linear(2, 3))
                print(fc(x).shape)
                # torch.Size([16, 8, 3])

        * :ref:`中文API <SeqToANNContainer.__init__-cn>`

        .. _SeqToANNContainer.__init__-en:

        :param *args: one or many stateless ANN layers

        A container that contain sataeless ANN to handle sequential data. This container will concatenate inputs ``shape=[T, batch_size, ...]`` at time dimension as ``shape=[T * batch_size, ...]``, and send the reshaped inputs to contained ANN. The output will be split to ``shape=[T, batch_size, ...]``.

        Examples:

        .. code-block:: python
            with torch.no_grad():
                T = 16
                batch_size = 8
                x = torch.rand([T, batch_size, 4])
                fc = SeqToANNContainer(nn.Linear(4, 2), nn.Linear(2, 3))
                print(fc(x).shape)
                # torch.Size([16, 8, 3])
        """
        super().__init__(*args)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: Tensor
        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: Tensor
        """
        return functional.seq_to_ann_forward(x_seq, self)


class STDPLearner(base.MemoryModule):
    def __init__(self,
                 tau_pre: float, tau_post: float,
                 f_pre, f_post
                 ) -> None:
        """
        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.clock_driven import layer, neuron, functional
            from matplotlib import pyplot as plt
            import numpy as np
            def f_pre(x):
                return x.abs() + 0.1

            def f_post(x):
                return - f_pre(x)

            fc = nn.Linear(1, 1, bias=False)

            stdp_learner = layer.STDPLearner(100., 100., f_pre, f_post)
            trace_pre = []
            trace_post = []
            w = []
            T = 256
            s_pre = torch.zeros([T, 1])
            s_post = torch.zeros([T, 1])
            s_pre[0: T // 2] = (torch.rand_like(s_pre[0: T // 2]) > 0.95).float()
            s_post[0: T // 2] = (torch.rand_like(s_post[0: T // 2]) > 0.9).float()

            s_pre[T // 2:] = (torch.rand_like(s_pre[T // 2:]) > 0.8).float()
            s_post[T // 2:] = (torch.rand_like(s_post[T // 2:]) > 0.95).float()

            for t in range(T):
                stdp_learner.stdp(s_pre[t], s_post[t], fc, 1e-2)
                trace_pre.append(stdp_learner.trace_pre.item())
                trace_post.append(stdp_learner.trace_post.item())
                w.append(fc.weight.item())

            plt.style.use('science')
            fig = plt.figure(figsize=(10, 6))
            s_pre = s_pre[:, 0].numpy()
            s_post = s_post[:, 0].numpy()
            t = np.arange(0, T)
            plt.subplot(5, 1, 1)
            plt.eventplot((t * s_pre)[s_pre == 1.], lineoffsets=0, colors='r')
            plt.yticks([])
            plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
            plt.xticks([])
            plt.xlim(0, T)
            plt.subplot(5, 1, 2)
            plt.plot(t, trace_pre)
            plt.ylabel('$tr_{pre}$', rotation=0, labelpad=10)
            plt.xticks([])
            plt.xlim(0, T)

            plt.subplot(5, 1, 3)
            plt.eventplot((t * s_post)[s_post == 1.], lineoffsets=0, colors='r')
            plt.yticks([])
            plt.ylabel('$S_{post}$', rotation=0, labelpad=10)
            plt.xticks([])
            plt.xlim(0, T)
            plt.subplot(5, 1, 4)
            plt.plot(t, trace_post)
            plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
            plt.xticks([])
            plt.xlim(0, T)
            plt.subplot(5, 1, 5)
            plt.plot(t, w)
            plt.ylabel('$w$', rotation=0, labelpad=10)
            plt.xlim(0, T)

            plt.show()

        .. image:: ./_static/API/clock_driven/layer/STDPLearner.*

        """
        super().__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        self.register_memory('trace_pre', 0.)
        self.register_memory('trace_post', 0.)
        self.f_pre = f_pre
        self.f_post = f_post

    @torch.no_grad()
    def stdp(self, s_pre: Tensor, s_post: Tensor, module: nn.Module, learning_rate: float):
        if isinstance(module, nn.Linear):
            # update trace
            self.trace_pre += - self.trace_pre / self.tau_pre + s_pre
            self.trace_post += - self.trace_post / self.tau_post + s_post

            # update weight
            delta_w_pre = self.f_pre(module.weight) * s_pre
            delta_w_post = self.f_post(module.weight) * s_post.unsqueeze(1)
            module.weight += (delta_w_pre + delta_w_post) * learning_rate
        else:
            raise NotImplementedError


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


class ConvBatchNorm2d(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        """
        A fused Conv2d-BatchNorm2d module. See :class:`torch.nn.Conv2d` and :class:`torch.nn.BatchNorm2d` for params information.

        Examples:

        .. code-block:: python

            convbn = ConvBatchNorm2d(3, 64, kernel_size=3, padding=1)
            x = torch.rand([16, 3, 224, 224])
            with torch.no_grad():
                convbn.eval()
                conv = convbn.get_fused_conv()
                conv.eval()
                print((convbn(x) - conv(x)).abs().max())

                k_weight = 1.5
                b_weight = 0.4
                k_bias = 0.8
                b_bias = 0.1

                conv.weight.data *= k_weight
                conv.weight.data += b_weight
                conv.bias.data *= k_bias
                conv.bias.data += b_bias

                convbn.scale_fused_weight(k_weight, b_weight)
                convbn.scale_fused_bias(k_bias, b_bias)

                print((convbn(x) - conv(x)).abs().max())
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=eps, momentum=momentum, affine=affine,
                                 track_running_stats=track_running_stats)

    def forward(self, x: Tensor):
        return self.bn(self.conv(x))

    def get_fused_weight(self):
        """
        :return: the weight of this fused module
        :rtype: Tensor
        """
        return functional.fused_conv2d_weight_of_convbn2d(self.conv, self.bn)

    def get_fused_bias(self):
        """
        :return: the bias of this fused module
        :rtype: Tensor
        """
        return functional.fused_conv2d_bias_of_convbn2d(self.conv, self.bn)

    @torch.no_grad()
    def scale_fused_weight(self, k=None, b=None):
        """
        :param k: scale factor
        :type k: float or None
        :param b: bias factor
        :type b: float or None

        Set the `weight` of this fused module to `weight * k + b`
        """
        functional.scale_fused_conv2d_weight_of_convbn2d(self.conv, self.bn, k, b)

    @torch.no_grad()
    def scale_fused_bias(self, k=None, b=None):
        """
        :param k: scale factor
        :type k: float or None
        :param b: bias factor
        :type b: float or None

        Set the `bias` of this fused module to `bias * k + b`
        """
        functional.scale_fused_conv2d_bias_of_convbn2d(self.conv, self.bn, k, b)

    def get_fused_conv(self):
        return functional.fuse_convbn2d(self.conv, self.bn)


class ElementWiseRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, element_wise_function: Callable):
        """
        :param sub_module: the contained module
        :type sub_module: torch.nn.Module
        :param element_wise_function: the user-defined element-wise function, which should have the format ``z=f(x, y)``
        :type element_wise_function: Callable

        A container that use a element-wise recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = f(x[t], y[t-1])

        where :math:`f` is the user-defined element-wise function. We set :math:`y[-1] = 0`.

        .. admonition:: Note
            :class: note

            The shape inputs and outputs of ``sub_module`` must be the same.

        Codes example:

        .. code-block:: python

            T = 8
            net = ElementWiseRecurrentContainer(neuron.IFNode(v_reset=None), element_wise_function=lambda x, y: x + y)
            print(net)
            x = torch.zeros([T])
            x[0] = 1.5
            for t in range(T):
                print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

            functional.reset_net(net)
        """
        super().__init__()
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory('y', None)

    def forward(self, x: Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.y, x))
        return self.y

    def extra_repr(self) -> str:
        return f'element-wise function={self.element_wise_function}'


class LinearRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        :param sub_module: the contained module
        :type sub_module: torch.nn.Module
        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: If set to ``False``, the layer will not learn an additive bias
        :type bias: bool

        A container that use a linear recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = \\begin{pmatrix} x[t] \\\\ y[t-1]\\end{pmatrix} W^{T} + b

        where :math:`W, b` are the weight and bias of the linear connection. We set :math:`y[-1] = 0`.

        :math:`x[t]` should have the shape ``[N, *, in_features]``, and :math:`y[t]` has the shape ``[N, *, out_features]``.

        .. admonition:: Tip
            :class: tip

            The recurrent connection is implement by ``torch.nn.Linear(in_features + out_features, in_features, bias)``.

        .. code-block:: python

            in_features = 4
            out_features = 2
            T = 8
            N = 2
            net = LinearRecurrentContainer(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    neuron.LIFNode(),
                ),
                in_features, out_features)
            print(net)
            x = torch.rand([T, N, in_features])
            for t in range(T):
                print(t, net(x[t]))

            functional.reset_net(net)

        """
        super().__init__()
        self.sub_module_out_features = out_features
        self.rc = nn.Linear(in_features + out_features, in_features, bias)
        self.sub_module = sub_module
        self.register_memory('y', None)

    def forward(self, x: Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.sub_module_out_features]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.zeros(out_shape).to(x)
        x = torch.cat((x, self.y), dim=-1)
        self.y = self.sub_module(self.rc(x))
        return self.y
class _MultiStepThresholdDependentBatchNormBase(_BatchNorm):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.v_th = v_th
        assert self.affine, "ThresholdDependentBatchNorm needs to set `affine = True`!"
        torch.nn.init.constant_(self.weight, alpha * v_th)

    def forward(self, x_seq):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
        y = super().forward(y)
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class MultiStepThresholdDependentBatchNorm1d(_MultiStepThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm1d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm1d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(
                f'expected 3D or 4D input with shape [T, N, C] or [T, N, C, M], but got input with shape {x.shape}')


class MultiStepThresholdDependentBatchNorm2d(_MultiStepThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm2d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm2d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')


class MultiStepThresholdDependentBatchNorm3d(_MultiStepThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm3d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm3d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f'expected 6D input with shape [T, N, C, D, H, W], but got input with shape {x.shape}')


class MultiStepTemporalWiseAttention(nn.Module):
    def __init__(self, T: int, reduction: int = 16, dimension: int = 4):
        """
        * :ref:`API in English <MultiStepTemporalWiseAttention.__init__-en>`

        .. _MultiStepTemporalWiseAttention.__init__-cn:

        :param T: 输入数据的时间步长

        :param reduction: 压缩比

        :param dimension: 输入数据的维度。当输入数据为[T, N, C, H, W]时， dimension = 4；输入数据维度为[T, N, L]时，dimension = 2。

        `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_ 中提出
        的MultiStepTemporalWiseAttention层。MultiStepTemporalWiseAttention层必须放在二维卷积层之后脉冲神经元之前，例如：

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        输入的尺寸是 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` ，经过MultiStepTemporalWiseAttention层，输出为 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` 。

        ``reduction`` 是压缩比，相当于论文中的 :math:`r`。

        * :ref:`中文API <MultiStepTemporalWiseAttention.__init__-cn>`

        .. _MultiStepTemporalWiseAttention.__init__-en:

        :param T: timewindows of input

        :param reduction: reduction ratio

        :param dimension: Dimensions of input. If the input dimension is [T, N, C, H, W], dimension = 4; when the input dimension is [T, N, L], dimension = 2.

        The MultiStepTemporalWiseAttention layer is proposed in `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_.

        It should be placed after the convolution layer and before the spiking neurons, e.g.,

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        The dimension of the input is ``[T, N, C, H, W]`` or  ``[T, N, L]`` , after the MultiStepTemporalWiseAttention layer, the output dimension is ``[T, N, C, H, W]`` or  ``[T, N, L]`` .

        ``reduction`` is the reduction ratio，which is :math:`r` in the paper.

        """
        super().__init__()
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'

        self.dimension = dimension

        # Sequence
        if self.dimension == 2:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        elif self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

        assert T >= reduction, 'reduction cannot be greater than T'

        # Excitation
        self.sharedMLP = nn.Sequential(
            nn.Linear(T, T // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(T // reduction, T, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(f'expected 3D or 5D input with shape [T, N, M] or [T, N, C, H, W], but got input with shape {x_seq.shape}')
        x_seq = x_seq.transpose(0, 1)
        avgout = self.sharedMLP(self.avg_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]]))
        maxout = self.sharedMLP(self.max_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]]))
        scores = self.sigmoid(avgout + maxout)
        if self.dimension == 2:
            y_seq = x_seq * scores[:, :, None]
        elif self.dimension == 4:
            y_seq = x_seq * scores[:, :, None, None, None]
        y_seq = y_seq.transpose(0, 1)
        return y_seq
