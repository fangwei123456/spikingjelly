时间驱动：编码器
=======================================
本教程作者： `Grasshlw <https://github.com/Grasshlw>`_

本节教程主要关注 ``spikingjelly.clock_driven.encoding`` ，介绍编码器。

编码器基类
-----------------

在 ``spikingjelly.clock_driven`` 中，所定义的编码器都继承自编码器基类 ``BaseEncoder`` ，该编码器继承 ``torch.nn.Module`` ，
定义三个方法， 其一 ``forward`` 将输入数据 ``x`` 编码为脉冲；其二 ``step`` 针对多数编码器， ``x`` 被编码成一定长度的脉冲序列，
需进行多步输出，则用 ``step`` 获取每一步的脉冲数据；其三 ``reset`` 将编码器的状态变量设置为初始状态。

.. code-block:: python

    class BaseEncoder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            raise NotImplementedError

        def step(self):
            raise NotImplementedError

        def reset(self):
            pass

周期编码器
-----------------

周期编码器是周期性输出给定的脉冲序列的编码器。与输入的数据无关，类 ``PeriodicEncoder`` 在初始化时已设定好要输出的脉冲序列 ``out_spike`` ，
并可在使用过程中通过方法 ``set_out_spike`` 重新设定。

.. code-block:: python

    class PeriodicEncoder(BaseEncoder):
        def __init__(self, out_spike):
            super().__init__()
            assert out_spike.dtype == torch.bool
            self.out_spike = out_spike
            self.T = out_spike.shape[0]
            self.index = 0

示例：给定3个神经元，时间步长为5的脉冲序列，分别为 ``01000`` 、 ``10000`` 、 ``00001`` 。初始化周期编码器，输出20个时间步长的仿真脉冲数据。

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # 给定脉冲序列
    set_spike = torch.full((3, 5), 0, dtype=torch.bool)
    set_spike[0, 1] = 1
    set_spike[1, 0] = 1
    set_spike[2, 4] = 1

    pe = encoding.PeriodicEncoder(set_spike.transpose(0, 1))

    # 输出周期性编码器的编码结果
    out_spike = torch.full((3, 20), 0, dtype=torch.bool)
    for t in range(out_spike.shape[1]):
        out_spike[:, t] = pe.step()

    plt.style.use(['science', 'muted'])
    visualizing.plot_1d_spikes(out_spike.float().numpy(), 'PeriodicEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

.. image:: ../_static/tutorials/clock_driven/2_encoding/1.*
    :width: 100%

延迟编码器
-------------------

延迟编码器是根据输入数据 ``x`` ，延迟发放脉冲的编码器。当刺激强度越大，发放时间就越早，且存在最大脉冲发放时间。因此对于每一个输入数据 ``x`` ，
都能得到一段时间步长为最大脉冲发放时间的脉冲序列，每段序列有且仅有一个脉冲发放。

脉冲发放时间 :math:`t_i` 与刺激强度 :math:`x_i` 满足以下二式：
当编码类型为线性时（ ``function_type='linear'`` )

.. math::
    t_i = (t_{max} - 1) * (1 - x_i)

当编码类型为对数时（ ``function_type='log'`` ）

.. math::
    t_i = (t_{max} - 1) - ln(\alpha * x_i + 1)

其中， :math:`t_{max}` 为最大脉冲发放时间， :math:`x_i` 需归一化到 :math:`[0,1]`。

考虑第二个式子， :math:`\alpha` 需满足：

.. math::
    (t_{max} - 1) - ln(\alpha * 1 + 1) = 0

这会导致该编码器很可能发生溢出，因为

.. math::
    \alpha = e^{t_{max} - 1} - 1

:math:`\alpha` 会随着 :math:`t_{max}` 增大而指数增长，最终造成溢出。

示例：随机生成6个 ``x`` ，分别为6个神经元的刺激强度，并设定最大脉冲发放时间为20，对以上输入数据进行编码。

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # 随机生成6个神经元的刺激强度，设定最大脉冲时间为20
    x = torch.rand(6)
    max_spike_time = 20

    # 将输入数据编码为脉冲序列
    le = encoding.LatencyEncoder(max_spike_time)
    le(x)

    # 输出延迟编码器的编码结果
    out_spike = torch.full((6, 20), 0, dtype=torch.bool)
    for t in range(max_spike_time):
        out_spike[:, t] = le.step()

    print(x)
    plt.style.use(['science', 'muted'])
    visualizing.plot_1d_spikes(out_spike.float().numpy(), 'LatencyEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

当随机生成的6个刺激强度分别为 ``0.6650`` 、 ``0.3704`` 、 ``0.8485`` 、 ``0.0247`` 、 ``0.5589`` 和 ``0.1030`` 时，得到的脉冲序列如下：

.. image:: ../_static/tutorials/clock_driven/2_encoding/2.*
    :width: 100%

泊松编码器
-----------------
泊松编码器将输入数据 ``x`` 编码为发放次数分布符合泊松过程的脉冲序列。泊松过程又被称为泊松流，当一个脉冲流满足独立增量性、增量平稳性和普通性时，
这样的脉冲流就是一个泊松流。更具体地说，在整个脉冲流中，互不相交的区间里出现脉冲的个数是相互独立的，且在任意一个区间中，出现脉冲的个数与区间的起点无关，
与区间的长度有关。因此，为了实现泊松编码，我们令一个时间步长的脉冲发放概率 :math:`p=x`, 其中 :math:`x` 需归一化到[0,1]。

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
    plt.style.use(['science', 'muted'])
    plt.imshow(x, cmap='gray')
    plt.axis('off')

    visualizing.plot_2d_spiking_feature_map(out_spike.float().numpy(), 4, 5, 30, 'PoissonEncoder')
    plt.axis('off')

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

高斯协调曲线编码器
------------------------

对于有 ``M`` 个特征的输入数据，高斯协调曲线编码器使用 ``tuning_curve_num`` 个神经元去编码输入数据的每一个特征，将每个特征编码为这 ``tuning_curve_num`` 个
神经元的脉冲发放时刻，因此可认为编码器有 ``M`` × ``tuning_curve_num`` 个神经元在工作。

对于第 :math:`i` 个特征 :math:`X^i`，取值范围为 :math:`X^i_{min}<=X^i<=X^i_{max}`。根据特征最大和最小值可计算出 ``tuning_curve_num`` 条高斯曲线 :math:`G^i_j` 的均值和方差：

.. math::
    \mu^i_j = x^i_{min} + \frac{2j-3}{2} \frac{x^i_{max} - x^i_{min}}{m - 2}
    \sigma^i_j = \frac{1}{\beta} \frac{x^i_{max} - x^i_{min}}{m - 2}

其中 :math:`\beta` 通常取值 :math:`1.5`。对于同一个特征，所有高斯曲线形状完全相同，对称轴位置不同。

生成高斯曲线后，则计算每个输入对应的高斯函数值，并将这些函数值线性转换为 ``[0, max_spike_time - 1]`` 之间的脉冲发放时间。此外，对于最后时刻发放的脉冲，
被认为是没有脉冲发放。

根据以上步骤，完成对输入数据的编码。

间隔编码器
-------------

间隔编码器是每隔 ``T`` 个时间步长发放一次脉冲的编码器。该编码器较为简单，此处不再详述。