时间驱动：编码器
=======================================
本教程作者： `Grasshlw <https://github.com/Grasshlw>`_, `Yanqi-Chen <https://github.com/Yanqi-Chen>`_, `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 :class:`spikingjelly.clock_driven.encoding` ，介绍编码器。

编码器基类
-----------------

在 :class:`spikingjelly.clock_driven.encoding` 中，存在2个基类编码器：

    1.无状态的编码器 :class:`spikingjelly.clock_driven.encoding.StatelessEncoder`

    2.有状态的编码器 :class:`spikingjelly.clock_driven.encoding.StatefulEncoder`



所定义的编码器都继承自这2个编码器基类之一。

无状态的编码器没有隐藏状态，输入数据 ``x[t]`` 可直接编码得到输出脉冲 ``spike[t]``；而有状态的编码器 ``encoder = StatefulEncoder(T)``，
编码器会在首次调用 ``forward`` 时使用 ``encode`` 函数对 ``T`` 个时刻的输入序列 ``x`` 进行编码得到 ``spike``，在第 ``t`` 次调用
``forward`` 时会输出 ``spike[t % T]``，可以从其前向传播的代码 :class:`spikingjelly.clock_driven.encoding.StatefulEncoder.forward` 看到这种操作：

.. code-block:: python

        def forward(self, x: torch.Tensor):
            if self.spike is None:
                self.encode(x)

            t = self.t
            self.t += 1
            if self.t >= self.T:
                self.t = 0
            return self.spike[t]

与SpikingJelly中的其他有状态module一样，调用 ``reset()`` 函数可以将有状态编码器进行重新初始化。

泊松编码器
-----------------
泊松编码器 :class:`spikingjelly.clock_driven.encoding.PoissonEncoder` 是无状态的编码器。泊松编码器将输入数据 ``x`` 编码为发放次数分布符合泊松过程的脉冲序列。泊松过程又被称为泊松流，当一个脉冲流满足独立增量性、增
量平稳性和普通性时，这样的脉冲流就是一个泊松流。更具体地说，在整个脉冲流中，互不相交的区间里出现脉冲的个数是相互独立的，且在任意一个区间中，出现脉冲的个数
与区间的起点无关，与区间的长度有关。因此，为了实现泊松编码，我们令一个时间步长的脉冲发放概率 :math:`p=x`, 其中 :math:`x` 需归一化到[0,1]。

示例：输入图像为 `lena512.bmp <https://www.ece.rice.edu/~wakin/images/lena512.bmp>`_ ，仿真20个时间步长，得到20个脉冲矩阵。

.. code-block:: python

    import torch
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # 读入lena图像
    lena_img = np.array(Image.open('lena512.bmp')) / 255
    x = torch.from_numpy(lena_img)

    pe = encoding.PoissonEncoder()

    # 仿真20个时间步长，将图像编码为脉冲矩阵并输出
    w, h = x.shape
    out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
    T = 20
    for t in range(T):
        out_spike[t] = pe(x)

    plt.figure()
    plt.imshow(x, cmap='gray')
    plt.axis('off')

    visualizing.plot_2d_spiking_feature_map(out_spike.float().numpy(), 4, 5, 30, 'PoissonEncoder')
    plt.axis('off')
    plt.show()

lena原灰度图和编码后20个脉冲矩阵如下：

.. image:: ../_static/tutorials/clock_driven/2_encoding/3.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/2_encoding/4.*
    :width: 100%

对比原灰度图和编码后的脉冲矩阵，可发现脉冲矩阵很接近原灰度图的轮廓，可见泊松编码器性能的优越性。

同样对lena灰度图进行编码，仿真512个时间步长，将每一步得到的脉冲矩阵叠加，得到第1、128、256、384、512步叠加得到的结果并画图：

.. code-block:: python

    # 仿真512个时间不长，将编码的脉冲矩阵逐次叠加，得到第1、128、256、384、512次叠加的结果并输出
    superposition = torch.full((w, h), 0, dtype=torch.float)
    superposition_ = torch.full((5, w, h), 0, dtype=torch.float)
    T = 512
    for t in range(T):
        superposition += pe(x).float()
        if t == 0 or t == 127 or t == 255 or t == 387 or t == 511:
            superposition_[int((t + 1) / 128)] = superposition

    # 归一化
    for i in range(5):
        min_ = superposition_[i].min()
        max_ = superposition_[i].max()
        superposition_[i] = (superposition_[i] - min_) / (max_ - min_)

    # 画图
    visualizing.plot_2d_spiking_feature_map(superposition_.numpy(), 1, 5, 30, 'PoissonEncoder')
    plt.axis('off')

    plt.show()

叠加后的图像如下：

.. image:: ../_static/tutorials/clock_driven/2_encoding/5.*
    :width: 100%

可见当仿真足够的步长，泊松编码器得到的脉冲叠加后几乎可以重构出原始图像。

周期编码器
-----------------
周期编码器 :class:`spikingjelly.clock_driven.encoding.PoissonEncoder` 是周期性输出给定的脉冲序列的编码器。``PeriodicEncoder`` 在
初始化时可以设定好要输出的脉冲序列 ``spike`` ，也可以随时调用 :class:`spikingjelly.clock_driven.encoding.PoissonEncoder.encode` 重
新设定。

.. code-block:: python

    class PeriodicEncoder(BaseEncoder):
        def __init__(self, spike: torch.Tensor):
            super().__init__(spike.shape[0])
            self.encode(spike)
        def encode(self, spike: torch.Tensor):
            self.spike = spike
            self.T = spike.shape[0]

示例：给定3个神经元，时间步长为5的脉冲序列，分别为 ``01000`` 、 ``10000`` 、 ``00001`` 。初始化周期编码器，输出20个时间步长的仿真脉冲数据。

.. code-block:: python

    spike = torch.full((5, 3), 0)
    spike[1, 0] = 1
    spike[0, 1] = 1
    spike[4, 2] = 1

    pe = encoding.PeriodicEncoder(spike)

    # 输出周期性编码器的编码结果
    out_spike = torch.full((20, 3), 0)
    for t in range(out_spike.shape[0]):
        out_spike[t] = pe(spike)

    visualizing.plot_1d_spikes(out_spike.float().numpy(), 'PeriodicEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

.. image:: ../_static/tutorials/clock_driven/2_encoding/1.*
    :width: 100%

延迟编码器
-------------------
延迟编码器 :class:`spikingjelly.clock_driven.encoding.LatencyEncoder` 是根据输入数据 ``x`` ，延迟发放脉冲的编码器。当刺激强度越大，发放
时间就越早，且存在最大脉冲发放时间。因此对于每一个输入数据 ``x``，都能得到一段时间步长为最大脉冲发放时间的脉冲序列，每段序列有且仅有一个脉冲发放。

脉冲发放时间 :math:`t_f` 与刺激强度 :math:`x \in [0, 1]` 满足以下二式：
当编码类型为线性时（ ``function_type='linear'`` )

.. math::
    t_f(x) = (T - 1)(1 - x)

当编码类型为对数时（ ``function_type='log'`` ）

.. math::
    t_f(x) = (T - 1) - ln(\alpha * x + 1)

其中， :math:`T` 为最大脉冲发放时间， :math:`x` 需归一化到 :math:`[0,1]`。

考虑第二个式子， :math:`\alpha` 需满足：

.. math::
    (T - 1) - ln(\alpha * 1 + 1) = 0

这会导致该编码器很可能发生溢出，因为

.. math::
    \alpha = e^{T - 1} - 1

:math:`\alpha` 会随着 :math:`T` 增大而指数增长，最终造成溢出。

示例：随机生成6个 ``x`` ，分别为6个神经元的刺激强度，并设定最大脉冲发放时间为20，对以上输入数据进行编码。

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # 随机生成6个神经元的刺激强度，设定最大脉冲时间为20
    N = 6
    x = torch.rand([N])
    T = 20

    # 将输入数据编码为脉冲序列
    le = encoding.LatencyEncoder(T)

    # 输出延迟编码器的编码结果
    out_spike = torch.zeros([T, N])
    for t in range(T):
        out_spike[t] = le(x)

    print(x)
    visualizing.plot_1d_spikes(out_spike.numpy(), 'LatencyEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

当随机生成的6个刺激强度分别为 ``0.6650`` 、 ``0.3704`` 、 ``0.8485`` 、 ``0.0247`` 、 ``0.5589`` 和 ``0.1030`` 时，得到的脉冲序列如下：

.. image:: ../_static/tutorials/clock_driven/2_encoding/2.*
    :width: 100%

带权相位编码器
--------------

一种基于二进制表示的编码方法。

将输入数据按照二进制各位展开，从高位到低位遍历输入进行脉冲编码。相比于频率编码，每一位携带的信息量更多。编码相位数为 :math:`K` 时，可以对于处于区间 :math:`[0, 1-2^{-K}]` 的数进行编码。以下为原始论文 [#kim2018deep]_ 中 :math:`K=8` 的示例：

+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| Phase (K=8)                      | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              |
+==================================+================+================+================+================+================+================+================+================+
| Spike weight :math:`\omega(t)`   | 2\ :sup:`-1`   | 2\ :sup:`-2`   | 2\ :sup:`-3`   | 2\ :sup:`-4`   | 2\ :sup:`-5`   | 2\ :sup:`-6`   | 2\ :sup:`-7`   | 2\ :sup:`-8`   |
+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| 192/256                          | 1              | 1              | 0              | 0              | 0              | 0              | 0              | 0              |
+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| 1/256                            | 0              | 0              | 0              | 0              | 0              | 0              | 0              | 1              |
+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| 128/256                          | 1              | 0              | 0              | 0              | 0              | 0              | 0              | 0              |
+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| 255/256                          | 1              | 1              | 1              | 1              | 1              | 1              | 1              | 1              |
+----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+

.. [#kim2018deep] Kim J, Kim H, Huh S, et al. Deep neural networks with weighted spikes[J]. Neurocomputing, 2018, 311: 373-386.