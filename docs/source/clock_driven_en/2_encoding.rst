Time driven: Encoder
=======================================
Author: `Grasshlw <https://github.com/Grasshlw>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

This tutorial focuses on ``spikingjelly.clock_driven.encoding`` and introduces the encoder.

Encoder base class
--------------------

In ``spikingjelly.clock_driven``, the defined encoders are inherited from the encoder base class ``BaseEncoder``, the encoder
inherits ``torch.nn.Module``, defines three methods, the first ``forward`` encodes the input data ``x`` into pulse.
The second ``step`` is for most encoders, ``x`` is encoded into a pulse sequence of a certain length, and multi-step
output is required, ``step`` is used to obtain the pulse data of each step. The third ``reset`` sets the state variable
of the encoder to the initial state.

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

Periodic encoder is an encoder that periodically outputs a given pulse sequence. Regardless of the input data, the
class ``PeriodicEncoder`` has set the pulse sequence ``out_spike`` to be output during initialization, and can be reset by
the method ``set_out_spike`` during use.

.. code-block:: python

    class PeriodicEncoder(BaseEncoder):
        def __init__(self, out_spike):
            super().__init__()
            assert out_spike.dtype == torch.bool
            self.out_spike = out_spike
            self.T = out_spike.shape[0]
            self.index = 0

Example: Given 3 neurons and a pulse sequence with a time-step of 5, they are ``01000``, ``10000``, and ``00001`` respectively.
Initialize the periodic encoder and output 20 time-step simulation pulse data.

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # Given pulse sequence
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

Delay encoder
-------------------

The delayed encoder is an encoder that delays the delivery of pulses based on the input data ``x``. When the stimulus
intensity is greater, the firing time is earlier, and there is a maximum pulse firing time.
Therefore, for each input data ``x``, a pulse sequence with a period of time as the maximum pulse firing time can be
obtained, and each sequence has only one pulse firing.

The pulse firing time :math:`t_i` and the stimulus intensity :math:`x_i` satisfy the following two formulas, when the coding type is
linear (``function_type='linear'``)

.. math::
    t_i = (t_{max} - 1) * (1 - x_i)

When the encoding type is logarithmic (``function_type='log'`` )

.. math::
    t_i = (t_{max} - 1) - ln(\alpha * x_i + 1)

Among them, :math:`t_{max}` is the maximum pulse firing time, and :math:`x_i` needs to be normalized to :math:`[0,1]`.

Consider the second formula, :math:`\alpha` needs to satisfy:

.. math::
    (t_{max} - 1) - ln(\alpha * 1 + 1) = 0

This will cause the encoder to likely overflow because:

.. math::
    \alpha = e^{t_{max} - 1} - 1

:math:`\alpha` will increase exponentially as :math:`t_{max}` increases, eventually causing overflow.

Example: Randomly generate six ``x``, each of which is the stimulation intensity of 6 neurons, and set the maximum pulse
firing time to 20, and encode the above input data.

.. code-block:: python

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from spikingjelly.clock_driven import encoding
    from spikingjelly import visualizing

    # Randomly generate stimulation intensity of 6 neurons, set the maximum pulse time to 20
    x = torch.rand(6)
    max_spike_time = 20

    # Encode input data into pulse sequence
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

When the randomly generated 6 stimulus intensities are ``0.6650``, ``0.3704``, ``0.8485``, ``0.0247``, ``0.5589``, and ``0.1030``, the pulse
sequence obtained is as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/2.*
    :width: 100%

Poisson encoder
-----------------
The Poisson encoder encodes the input data ``x`` into a pulse sequence whose firing times distribution conforms to the
Poisson process. The Poisson process is also called Poisson flow. When a pulse flow satisfies independent increment,
incremental stability and commonality, such a pulse flow is a poisson flow. More specifically, in the entire pulse
stream, the number of pulses appearing in disjoint intervals is independent of each other, and in any interval,
the number of pulses appearing has nothing to do with the starting point of the interval, but is related to the
length of the interval. Therefore, in order to realize Poisson coding, we set the pulse firing probability of a
time step :math:`p=x`, where :math:`x` needs to be normalized to [0, 1].

Example: The input image is `lena512.bmp <https://www.ece.rice.edu/~wakin/images/lena512.bmp>`_ , and 20 time
steps are simulated to obtain 20 pulse matrices.

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

    # Simulate 20 time steps, encode the image into a pulse matrix and output
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

The original grayscale image of Lena and the encoded 20 pulse matrix are as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/3.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/2_encoding/4.*
    :width: 100%

Comparing the original grayscale image and the encoded pulse matrix, it can be found that the pulse matrix is
very close to the contour of the original grayscale image, which shows the superiority of the
Poisson encoder performance.

Also encode the Lena grayscale image, simulate 512 time steps, superimpose the pulse matrix obtained
in each step, and get the result of the superposition of steps 1, 128, 256, 384, and 512 and draw the picture:

.. code-block:: python

    # Simulate 512 time steps, superimpose the coded pulse matrix one by one to obtain the 1, 128, 256, 384, 512th superposition results and output
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

The superimposed image is as follows:

.. image:: ../_static/tutorials/clock_driven/2_encoding/5.*
    :width: 100%

It can be seen that when the simulation step is sufficient, the original image can almost be reconstructed after the
pulses obtained by the Poisson encoder are superimposed.

Gaussian coordination curve encoder
------------------------------------

For input data with ``M`` features, the Gaussian coordination curve encoder uses ``tuning_curve_num`` neurons
to encode each feature of the input data, and encodes each feature as the pulse firing time of
these ``tuning_curve_num`` neurons, so it can be considered that the encoder has ``M`` × ``tuning_curve_num`` neurons are working.

For the :math:`i` feature :math:`X^i`, the value range is :math:`X^i_{min}<=X^i<=X^i_{max}`. According to the maximum and minimum features,
the mean and variance of ``tuning_curve_num`` Gaussian curves Gij can be calculated:

.. math::
    \mu^i_j = x^i_{min} + \frac{2j-3}{2} \frac{x^i_{max} - x^i_{min}}{m - 2}
    \sigma^i_j = \frac{1}{\beta} \frac{x^i_{max} - x^i_{min}}{m - 2}

Where :math:`\beta` is usually :math:`1.5`, for the same feature, all Gaussian curves have the same shape, and the symmetry axis positions are different.

After the Gaussian curve is generated, the Gaussian function value corresponding to each input is calculated, and
these function values are linearly converted into the pulse firing time between ``[0, max_spike_time - 1]``.
In addition, for the pulses delivered at the last moment, it is considered that there is no pulse delivery.

According to the above steps, the encoding of the input data is completed.

Interval encoder
-------------------

The interval encoder is an encoder that emits a pulse every ``T`` time steps. The encoder is relatively simple and
will not be detailed here.
