import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SpikingFlow.clock_driven import accelerating

class NeuNorm(nn.Module):
    def __init__(self, in_channels, k=0.9):
        '''
        * :ref:`API in English <NeuNorm.__init__-en>`

        .. _NeuNorm.__init__-cn:

        .. warning::
            可能是错误的实现。测试的结果表明，增加NeuNorm后的收敛速度和正确率反而下降了。

        :param in_channels: 输入数据的通道数

        :param k: 动量项系数

        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_ 中提出\\
        的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如：

        ``Conv2d -> LIF -> NeuNorm``

        要求输入的尺寸是 ``[batch_size, in_channels, W, H]``。

        ``in_channels`` 是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        ``k`` 是动量项系数，相当于论文中的 :math:`k_{\\tau 2}`。

        论文中的 :math:`\\frac{v}{F}` 会根据 :math:`k_{\\tau 2} + vF = 1` 自动算出。

        * :ref:`中文API <NeuNorm.__init__-cn>`

        .. _NeuNorm.__init__-en:

        .. admonition:: Warning
            :class: warning

            There may be some wrong in code implement. Our experiment results show that networks with NeuNorm perform worse.

        :param in_channels: channels of input

        :param k: momentum factor

        The NeuNorm layer is proposed in `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_.

        It should be placed after spiking neurons behind convolution layer, e.g.,

        ``Conv2d -> LIF -> NeuNorm``

        The input should be a 4-D tensor with ``shape = [batch_size, in_channels, W, H]``.

        ``in_channels`` is the channels of input，which is :math:`F` in the paper.

        ``k`` is the momentum factor，which is :math:`k_{\\tau 2}` in the paper.

        :math:`\\frac{v}{F}` will be calculated by :math:`k_{\\tau 2} + vF = 1` autonomously.

        '''
        super().__init__()
        self.x = 0
        self.k0 = k
        self.k1 = (1 - self.k0) / in_channels**2
        self.w = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, in_spikes: torch.Tensor):
        self.x = self.k0 * self.x + (self.k1 * in_spikes.sum(dim=1).unsqueeze(1))
        return in_spikes - self.w * self.x

    def reset(self):
        '''
        * :ref:`API in English <NeuNorm.reset-en>`

        .. _NeuNorm.reset-cn:

        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。

        * :ref:`中文API <NeuNorm.reset-cn>`

        .. _NeuNorm.reset-en:

        :return: None

        This layer is stateful. This function will reset all stateful variables.
        '''
        self.x = 0

class DCT(nn.Module):
    def __init__(self, kernel_size):
        '''
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
        '''
        super().__init__()
        self.kernel = torch.zeros(size=[kernel_size, kernel_size])
        for i in range(0, kernel_size):
            for j in range(kernel_size):
                if i == 0:
                    self.kernel[i][j] = math.sqrt(1 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)
                else:
                    self.kernel[i][j] = math.sqrt(2 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)

    def forward(self, x: torch.Tensor):
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        x_shape = x.shape
        x = x.view(-1, x_shape[-2], x_shape[-1])
        ret = torch.zeros_like(x)
        for i in range(0, x_shape[-2], self.kernel.shape[0]):
            for j in range(0, x_shape[-1], self.kernel.shape[0]):
                ret[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]] \
                    = self.kernel.matmul(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]]).matmul(self.kernel.t())
        return ret.view(x_shape)

class AXAT(nn.Module):
    def __init__(self, in_features, out_features):
        '''
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
        '''
        super().__init__()
        self.A = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))


    def forward(self, x: torch.Tensor):
        x_shape = list(x.shape)
        x = x.view(-1, x_shape[-2], x_shape[-1])
        x = self.A.matmul(x).matmul(self.A.t())
        x_shape[-1] = x.shape[-1]
        x_shape[-2] = x.shape[-2]
        return x.view(x_shape)

class Dropout(nn.Module):
    def __init__(self, p=0.5, behind_spiking_layer=False):
        '''
        * :ref:`API in English <Dropout.__init__-en>`

        .. _Dropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :param behind_spiking_layer: 本层是否位于输出脉冲数据的层，例如 ``neuron.LIFNode`` 层之后。若为 ``True``，则计算会有一定的加速

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
        :param behind_spiking_layer: whether this layer is behind a spiking layer, such as ``neuron.LIFNode``. If ``True``,
            the calculation will be accelerated

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
        '''
        super().__init__()
        assert 0 < p < 1
        self.mask = None
        self.p = p
        self.behind_spiking_layer = behind_spiking_layer

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.mask = F.dropout(torch.ones_like(x), self.p, training=True).bool()
            return accelerating.mul(x, self.mask, self.behind_spiking_layer)
        else:
            return x


    def reset(self):
        '''
        * :ref:`API in English <Dropout.reset-en>`

        .. _Dropout.reset-cn:

        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。

        * :ref:`中文API <Dropout.reset-cn>`

        .. _Dropout.reset-en:

        :return: None

        This layer is stateful. This function will reset all stateful variables.
        '''
        self.mask = None

class Dropout2d(nn.Module):
    def __init__(self, p=0.2, behind_spiking_layer=False):
        '''
        * :ref:`API in English <Dropout2d.__init__-en>`

        .. _Dropout2d.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :param behind_spiking_layer: 本层是否位于输出脉冲数据的层，例如 ``neuron.LIFNode`` 层后。若为 ``True``，则计算会有一定的加速

        与 ``torch.nn.Dropout2d`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        关于SNN中Dropout的更多信息，参见 :ref:`layer.Dropout <Dropout.__init__-cn>`。

        * :ref:`中文API <Dropout2d.__init__-cn>`

        .. _Dropout2d.__init__-en:

        :param p: probability of an element to be zeroed
        :param behind_spiking_layer: whether this layer is behind a spiking layer, such as ``neuron.LIFNode``. If ``True``,
            the calculation will be accelerated
            
        This layer is almost same with ``torch.nn.Dropout2d``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        For more information about Dropout in SNN, refer to :ref:`layer.Dropout <Dropout.__init__-en>`.
        '''
        super().__init__()
        assert 0 < p < 1
        self.mask = None
        self.p = p
        self.behind_spiking_layer = behind_spiking_layer

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.mask = F.dropout2d(torch.ones_like(x), self.p, training=True).bool()
            return accelerating.mul(x, self.mask, self.behind_spiking_layer)
        else:
            return x

    def reset(self):
        '''
        * :ref:`API in English <Dropout2d.reset-en>`

        .. _Dropout2d.reset-cn:

        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。

        * :ref:`中文API <Dropout2d.reset-cn>`

        .. _Dropout2d.reset-en:

        :return: None

        This layer is stateful. This function will reset all stateful variables.
        '''
        self.mask = None

class SynapseFilter(nn.Module):
    def __init__(self, tau=100.0, learnable=False):
        '''
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

        .. image:: ./_static/API/LowPassSynapseFilter.png

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

        .. image:: ./_static/API/LowPassSynapseFilter.png

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

        '''
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.ones(size=[1]) / tau)
        else:
            self.tau = 1 / tau
        self.out_i = 0

    def extra_repr(self):
        return 'tau={}'.format(
            1 / self.tau
        )

    def forward(self, in_spikes: torch.Tensor):
        self.out_i = self.out_i - (1 - in_spikes) * self.out_i * self.tau + in_spikes
        return self.out_i

    def reset(self):
        '''
        * :ref:`API in English <LowPassSynapse.reset-en>`

        .. _LowPassSynapse.reset-cn:

        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。

        * :ref:`中文API <LowPassSynapse.reset-cn>`

        .. _LowPassSynapse.reset-en:

        :return: None

        This layer is stateful. This function will reset all stateful variables.
        '''
        self.out_i = 0

class ChannelsMaxPool(nn.Module):
    def __init__(self, pool: nn.MaxPool1d):
        '''
        * :ref:`API in English <ChannelsMaxPool.__init__-en>`

        .. _ChannelsMaxPool.__init__-cn:

        :param pool: ``nn.Maxpool1d``，池化层

        使用 ``pool`` 将输入的4-D数据在第1个维度上进行池化。

        示例代码：

        .. code-block:: python

            >>> cmp = ChannelsMaxPool(torch.nn.MaxPool1d(2, 2))
            >>> x = torch.rand(size=[2, 8, 4, 4])
            >>> y = cmp(x)
            >>> y.shape
            torch.Size([2, 4, 4, 4])

        * :ref:`中文API <ChannelsMaxPool.__init__-cn>`

        .. _ChannelsMaxPool.__init__-en:

        :param pool: ``nn.Maxpool1d``, the pool layer

        Use ``pool`` to pooling 4-D input at dimension 1.

        Examples:

        .. code-block:: python

            >>> cmp = ChannelsMaxPool(torch.nn.MaxPool1d(2, 2))
            >>> x = torch.rand(size=[2, 8, 4, 4])
            >>> y = cmp(x)
            >>> y.shape
            torch.Size([2, 4, 4, 4])
        '''
        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        return self.pool(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view((x_shape[0], -1) + x_shape[2:])