Clock driven: Encoder
=======================================
Author: `Grasshlw <https://github.com/Grasshlw>`_, `Yanqi-Chen <https://github.com/Yanqi-Chen>`_, `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

This tutorial focuses on :class:`spikingjelly.clock_driven.encoding` and introduces several encoders.

The Base Class of Encoder
----------------------------------------

All encodes are based on two base encoders:

    1.The stateless base encoder :class:`spikingjelly.clock_driven.encoding.StatelessEncoder`

    2.The stateful base encoder :class:`spikingjelly.clock_driven.encoding.StatefulEncoder`

There are no hidden states in the stateless encoder, and the spikes ``spike[t]`` will be encoded from the input data
``x[t]`` at time-step ``t``. While the stateful encoder ``encoder = StatefulEncoder(T)`` will use ``encode`` function
to encode the input sequence ``x`` containing ``T`` time-steps data to ``spike`` at the first time of ``forward``, and
will output ``spike[t % T]`` at the``t``-th calling ``forward``. The codes of :class:`spikingjelly.clock_driven.encoding.StatefulEncoder.forward` are:

.. code-block:: python

        def forward(self, x: torch.Tensor):
            if self.spike is None:
                self.encode(x)

            t = self.t
            self.t += 1
            if self.t >= self.T:
                self.t = 0
            return self.spike[t]

Poisson Encoder
-----------------
The Poisson encoder :class:`spikingjelly.clock_driven.encoding.PoissonEncoder` is a stateless encoder. It converts the input data ``x`` into a spike with the same shape, which conforms to a Poisson process, i.e., the number of spikes during a certain period follows a Poisson distribution.
A Poisson process is also called a Poisson flow. When a spike flow satisfies the requirements of independent increment,
incremental stability and commonality, such a spike flow is a Poisson flow. More specifically, in the entire spike
stream, the number of spikes appearing in disjoint intervals is independent of each other, and in any interval,
the number of spikes is related to the length of the interval while not the starting point of the interval.
Therefore, in order to realize Poisson encoding, we set the firing probability of a
time step :math:`p=x`, where :math:`x` needs to be normalized to [0, 1].

Example: The input image is `lena512.bmp <https://www.ece.rice.edu/~wakin/images/lena512.bmp>`_ , and 20 time
steps are simulated to obtain 20 spike matrices.

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

The original grayscale image of Lena and 20 resulted spike matrices are as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/3.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/2_encoding/4.*
    :width: 100%

Comparing the original grayscale image to the spike matrix, it can be found that the spike matrix is
very close to the contour of the original grayscale image, which shows the superiority of the
Poisson encoder.

After simulating the Poisson encoder with the Lena grayscale image for 512 time steps, we superimpose the spike matrix obtained
in each step, and obtain the result of the superposition of steps 1, 128, 256, 384, and 512, and draw the picture:

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

The superimposed images are as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/5.*
    :width: 100%

It can be seen that when the simulation is sufficiently long, the original image can almost be reconstructed with the
superimposed images composed of spikes obtained by the Poisson encoder.

Periodic Encoder
-----------------

Periodic encoder :class:`spikingjelly.clock_driven.encoding.PoissonEncoder` is an encoder that periodically outputs spikes
from a given spike sequence. ``spike`` is set at the initialization of ``PeriodicEncoder``, and we can also use :class:`spikingjelly.clock_driven.encoding.PoissonEncoder.encode` to set a new ``spike``.

.. code-block:: python

    class PeriodicEncoder(BaseEncoder):
        def __init__(self, spike: torch.Tensor):
            super().__init__(spike.shape[0])
            self.encode(spike)
        def encode(self, spike: torch.Tensor):
            self.spike = spike
            self.T = spike.shape[0]

Example: Considering three neurons and spike sequences with 5 time steps, which are ``01000``, ``10000``, and ``00001`` respectively,
we initialize a periodic encoder and output simulated spike data with 20 time steps.

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

Latency encoder
-------------------

The latency encoder :class:`spikingjelly.clock_driven.encoding.LatencyEncoder` is an encoder that delays the delivery of spikes based on the input data ``x``. When the stimulus intensity is greater, the firing time is earlier, and there is a maximum spike latency.
Therefore, for each input data ``x``, a spike sequence with a period of the maximum spike latency can be
obtained.

The spike firing time :math:`t_f` and the stimulus intensity :math:`x \in [0, 1]` satisfy the following formulas. When the encoding type is
linear (``function_type='linear'``)

.. math::
    t_f(x) = (T - 1)(1 - x)
    
When the encoding type is logarithmic (``function_type='log'`` )

.. math::
    t_i = (t_{max} - 1) - ln(\alpha * x_i + 1)

In the formulas, :math:`t_{max}` is the maximum spike latency, and :math:`x_i` needs to be normalized to :math:`[0, 1]`.

Consider the second formula, :math:`\alpha` needs to satisfy:

.. math::
    (T - 1) - ln(\alpha * 1 + 1) = 0

This may cause the encoder to overflow:

.. math::
    \alpha = e^{T - 1} - 1

because :math:`\alpha` will increase exponentially as :math:`T` increases.

Example: Randomly generate six ``x``, each of which is the stimulation intensity of 6 neurons, and set the maximum spike
latency to 20, then use ``LatencyEncoder`` to encode the above input data.

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

When the randomly generated stimulus intensities are ``0.6650``, ``0.3704``, ``0.8485``, ``0.0247``, ``0.5589``, and ``0.1030``, the spike
sequence obtained is as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/2.*
    :width: 100%


Weighted phase encoder
------------------------

Weighted phase encoder is based on binary representations of floats. 

Inputs are decomposed to fractional bits and the spikes correspond to the binary value from the leftmost bit to the rightmost bit. Compared to rate coding, each spike in phase coding carries more information. When phase is :math:`K`, number lies in the interval :math:`[0, 1-2^{-K}]` can be encoded. Example when :math:`K=8` in original paper [#kim2018deep]_ is illustrated here:

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
