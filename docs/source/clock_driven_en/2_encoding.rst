Clock driven: Encoder
=======================================
Author: `Grasshlw <https://github.com/Grasshlw>`_, `Yanqi-Chen <https://github.com/Yanqi-Chen>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

This tutorial focuses on ``spikingjelly.clock_driven.encoding`` and introduces several encoders.

The base class of Encoder
----------------------------------------

In ``spikingjelly.clock_driven``, the defined encoders are inherited from the base class of encoder ``BaseEncoder``.
The  ``BaseEncoder`` inherits ``torch.nn.Module``, and defines three methods.
The ``forward`` method converts the input data ``x`` into spikes.
In the ``step`` method, ``x`` is encoded into a spike sequence, and ``step``  is used to obtain the spike data of each step for multiple steps.
The ``reset`` method sets the state variable of an encoder to the initial state.

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

Periodic encoder
-----------------

Periodic encoder is an encoder that periodically outputs spikes from a given sequence. Regardless of the input data, the
class ``PeriodicEncoder`` has defined the output spike sequence ``out_spike`` when initialization, and can be reset by
the method ``set_out_spike`` during application.

.. code-block:: python

    class PeriodicEncoder(BaseEncoder):
        def __init__(self, out_spike):
            super().__init__()
            assert out_spike.dtype == torch.bool
            self.out_spike = out_spike
            self.T = out_spike.shape[0]
            self.index = 0

Example: Considering three neurons and spike sequences with 5 time steps, which are ``01000``, ``10000``, and ``00001`` respectively,
we initialize a periodic encoder and output simulated spike data with 20 time steps.

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # Given spike sequence
    set_spike = torch.full((3, 5), 0, dtype=torch.bool)
    set_spike[0, 1] = 1
    set_spike[1, 0] = 1
    set_spike[2, 4] = 1

    pe = encoding.PeriodicEncoder(set_spike.transpose(0, 1))

    # Output the coding result of the periodic encoder
    out_spike = torch.full((3, 20), 0, dtype=torch.bool)
    for t in range(out_spike.shape[1]):
        out_spike[:, t] = pe.step()

    plt.style.use(['science', 'muted'])
    visualizing.plot_1d_spikes(out_spike.float().numpy(), 'PeriodicEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

.. image:: ../_static/tutorials/clock_driven/2_encoding/1.*
    :width: 100%

Latency encoder
-------------------

The latency encoder is an encoder that delays the delivery of spikes based on the input data ``x``. When the stimulus
intensity is greater, the firing time is earlier, and there is a maximum spike latency.
Therefore, for each input data ``x``, a spike sequence with a period of the maximum spike latency can be
obtained.

The spike firing time :math:`t_i` and the stimulus intensity :math:`x_i` satisfy the following formulas. When the encoding type is
linear (``function_type='linear'``)

.. math::
    t_i = (t_{max} - 1) * (1 - x_i)

When the encoding type is logarithmic (``function_type='log'`` )

.. math::
    t_i = (t_{max} - 1) - ln(\alpha * x_i + 1)

In the formulas, :math:`t_{max}` is the maximum spike latency, and :math:`x_i` needs to be normalized to :math:`[0, 1]`.

Consider the second formula, :math:`\alpha` needs to satisfy:

.. math::
    (t_{max} - 1) - ln(\alpha * 1 + 1) = 0

This may cause the encoder to overflow:

.. math::
    \alpha = e^{t_{max} - 1} - 1

because :math:`\alpha` will increase exponentially as :math:`t_{max}` increases.

Example: Randomly generate six ``x``, each of which is the stimulation intensity of 6 neurons, and set the maximum spike
latency to 20, then use ``LatencyEncoder`` to encode the above input data.

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # Randomly generate stimulation intensity of 6 neurons, set the maximum spike time to 20
    x = torch.rand(6)
    max_spike_time = 20

    # Encode input data into spike sequence
    le = encoding.LatencyEncoder(max_spike_time)
    le(x)

    # Output the encoding result of the delayed encoder
    out_spike = torch.full((6, 20), 0, dtype=torch.bool)
    for t in range(max_spike_time):
        out_spike[:, t] = le.step()

    print(x)
    plt.style.use(['science', 'muted'])
    visualizing.plot_1d_spikes(out_spike.float().numpy(), 'LatencyEncoder', 'Simulating Step', 'Neuron Index',
                               plot_firing_rate=False)
    plt.show()

When the randomly generated stimulus intensities are ``0.6650``, ``0.3704``, ``0.8485``, ``0.0247``, ``0.5589``, and ``0.1030``, the spike
sequence obtained is as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/2.*
    :width: 100%

Poisson encoder
-----------------
The Poisson encoder converts the input data ``x`` into a spike sequence, which conforms to a Poisson process,
i.e., the number of spikes during a certain period follows a Poisson distribution.
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

    # Read in Lena image
    lena_img = np.array(Image.open('lena512.bmp')) / 255
    x = torch.from_numpy(lena_img)

    pe = encoding.PoissonEncoder()

    # Simulate 20 time steps, encode the image into a spike matrix and output
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

    # Simulate 512 time steps, superimpose the coded spike matrix one by one to obtain the 1, 128, 256, 384, 512th superposition results and output
    superposition = torch.full((w, h), 0, dtype=torch.float)
    superposition_ = torch.full((5, w, h), 0, dtype=torch.float)
    T = 512
    for t in range(T):
        superposition += pe(x).float()
        if t == 0 or t == 127 or t == 255 or t == 387 or t == 511:
            superposition_[int((t + 1) / 128)] = superposition

    # Normalized
    for i in range(5):
        min_ = superposition_[i].min()
        max_ = superposition_[i].max()
        superposition_[i] = (superposition_[i] - min_) / (max_ - min_)

    # plot
    visualizing.plot_2d_spiking_feature_map(superposition_.numpy(), 1, 5, 30, 'PoissonEncoder')
    plt.axis('off')

    plt.show()

The superimposed images are as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/5.*
    :width: 100%

It can be seen that when the simulation is sufficiently long, the original image can almost be reconstructed with the
superimposed images composed of spikes obtained by the Poisson encoder.

Gaussian tuning curve encoder
------------------------------------

For input data with ``M`` features, the Gaussian tuning curve encoder uses ``tuning_curve_num`` neurons
to encode each feature of the input data, and encodes each feature as the firing time of
these ``tuning_curve_num`` neurons.
Therefore, the encoder has ``M`` × ``tuning_curve_num`` neurons to work properly.

For feature :math:`X^i`, the value range is :math:`X^i_{min}<=X^i<=X^i_{max}`. According to the maximum and minimum features,
the mean and standard deviation of Gaussian curve :math:`G_i^j` can be calculated as follows:

.. math::
    \mu^i_j = x^i_{min} + \frac{2j-3}{2} \frac{x^i_{max} - x^i_{min}}{m - 2},
    \sigma^i_j = \frac{1}{\beta} \frac{x^i_{max} - x^i_{min}}{m - 2}

where :math:`\beta` is usually :math:`1.5`.
For one feature, all ``tuning_curve_num`` Gaussian curves have the same shape, while the axes of symmetry are different.

After the Gaussian curve is generated, the output of the Gaussian function corresponding to each input is calculated, and
these outputs are linearly converted into firing timestamps between ``[0, max_spike_time - 1]``.
In addition, the spikes fired at the last moment are ignored as they never happen.

According to the above steps, the encoding of the input data is completed.

Interval encoder
-------------------

The interval encoder is an encoder that emits a spike every ``T`` time steps. The encoder is relatively simple and
will not be detailed here.

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
