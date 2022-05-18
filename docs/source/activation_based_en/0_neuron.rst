Clock driven: Neurons
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

This tutorial focuses on :class:`spikingjelly.clock_driven.neuron` and introduces spiking neurons and clock-driven
simulation methods.

Spiking Nneuron Model
-----------------------------------------------
In ``spikingjelly``, we define the neuron which can only output spikes, i.e. 0 or 1, as a "spiking neuron".
Networks that use spiking neurons are called Spiking Neural Networks (SNNs).
:class:`spikingjelly.clock_driven.neuron` defines various common spiking neuron models.
We take :class:`spikingjelly.clock_driven.neuron.LIFNode` as an example to introduce spiking neurons.

First, we need to import the relevant modules:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.clock_driven import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt

And then we create a new LIF neurons layer:

.. code-block:: python

    lif = neuron.LIFNode()

The LIF neurons layer has some parameters, which are explained in detail in the API documentation:

    - **tau** -- membrane time constant

    - **v_threshold** -- the threshold voltage of the neuron

    - **v_reset** -- the reset voltage of the neuron. If it is not ``None``, when the neuron releases a spike, the voltage will be reset to ``v_reset``; if it is set to ``None``, the voltage will be subtracted from ``v_threshold``

    - **surrogate_function** -- the surrogate function used to calculate the gradient of the spike function during back propagation

The ``surrogate_function`` behaves exactly the same as the step function during forward propagation,
and we will introduce its working principle for back propagation later. We can just ignore it now.

You may be curious about the number of neurons in this layer. For most neurons layers in :class:`spikingjelly.clock_driven.neuron`,
the number of neurons is automatically determined according to the ``shape`` of the received input after initialization or re-initialization by calling the ``reset()`` function.

Similar to neurons in RNN, spiking neurons are also stateful (they have memory).
The state variable of a spiking neuron is generally its membrane potential :math:`V_{t}`.
Therefore, neurons in :class:`spikingjelly.clock_driven.neuron` have state variable ``v``.
We can print the membrane potential of the newly created LIF neurons layer:

.. code-block:: python

    print(lif.v)
    # 0.0

We can find that ``lif.v`` is now ``0.0`` because we haven't given it any input yet.
We give several different inputs and observe the ``shape`` of ``lif.v``. We can find that it is consistent with the
numel of inputs:

.. code-block:: python

    x = torch.rand(size=[2, 3])
    lif(x)
    print('x.shape', x.shape, 'lif.v.shape', lif.v.shape)
    # x.shape torch.Size([2, 3]) lif.v.shape torch.Size([2, 3])
    lif.reset()

    x = torch.rand(size=[4, 5, 6])
    lif(x)
    print('x.shape', x.shape, 'lif.v.shape', lif.v.shape)
    # x.shape torch.Size([4, 5, 6]) lif.v.shape torch.Size([4, 5, 6])
    lif.reset()

What is the relationship between :math:`V_{t}` and input :math:`X_{t}`? In the spiking neuron,
it not only depends on the input :math:`X_{t}` at time-step ``t``,
but also on its membrane potential :math:`V_{t-1}` at the last time-step ``t-1``.

We often use the sub-threshold (when the membrane potential does not exceed the threshold potential ``V_{threshold}``) neuronal dynamics equation :math:`\frac{\mathrm{d}V(t)}{\mathrm{d}t} = f(V(t), X(t))` to describe the continuous-time
spiking neuron. For example. For LIF neurons, the equation is:

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

where :math:`\tau_{m}` is the membrane time constant and :math:`V_{reset}` is the reset potential. For such a differential equation, :math:`X(t)` is not a constant and it is difficult to obtain a explicit analytical solution.

The neurons in :class:`spikingjelly.clock_driven.neuron` use discrete difference equations to approximate continuous differential equations.
From the perspective of the discrete equation, the charging equation of the LIF neuron is:

.. math::
    \tau_{m} (V_{t} - V_{t-1}) = -(V_{t-1} - V_{reset}) + X_{t}

The expression of :math:`V_{t}` can be obtained as

.. math::
    V_{t} = f(V_{t-1}, X_{t}) = V_{t-1} + \frac{1}{\tau_{m}}(-(V_{t - 1} - V_{reset}) + X_{t})

The corresponding code can be found in :class:`spikingjelly.clock_driven.neuron.LIFNode.neuronal_charge`:

.. code-block:: python

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v += (x - self.v) / self.tau
            else:
                self.v += (x - (self.v - self.v_reset)) / self.tau

Different neurons have different charging equations. However, when the membrane potential exceeds the threshold potential,
the release of spike and the reset of the membrane potential are the same for all kinds of neurons. Therefore,
they all inherit from :class:`spikingjelly.clock_driven.neuron.BaseNode` and share the same discharge and reset equations. The codes of neuronal fire can be found at :class:`spikingjelly.clock_driven.neuron.BaseNode.neuronal_fire`:

.. code-block:: python

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

``surrogate_function()`` is a heaviside step function during forward propagation. When input is greater than or equal
to 0, it will return 1, otherwise it will return 0. We regard this kind of ``tensor`` whose elements are only 0 or 1 as spikes.

The release of spikes consumes the previously accumulated electric charge of the neuron, so there will be an
instantaneous decrease in the membrane potential, which is the neuronal reset. In SNNs, there are
two ways to realize neuronal reset:

#. Hard method: After releasing a spike, the membrane potential is directly set to the reset potential :math:`V = V_{reset}`

#. Soft method: After releasing a spike, the membrane potential subtracts the threshold voltage :math:`V = V - V_{threshold}`

It can be found that for neurons using the soft method, there is no need to reset the voltage :math:`V_{reset}`.
For the neurons in :class:`spikingjelly.clock_driven.neuron`, when ``v_reset`` is set to the a float value (e.g., the default value is ``1.0``), the neuron uses the hard reset; if ``v_reset`` is set to ``None``, the soft reset will be used.
We can find the corresponding codes in :class:`spikingjelly.clock_driven.neuron.BaseNode.neuronal_fire.neuronal_reset`:

.. code-block:: python

    def neuronal_reset(self):
        # ...
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1 - spike) * self.v + spike * self.v_reset


Three Equations to Describe Discrete Spiking Neurons
--------------------------------------------------------------
We can use the three discrete equations: neuronal charge, neuronal fire, and neuronal reset to describe all kinds of discrete spiking neurons. The neuronal charge and fire equations are:

.. math::
    H_{t} & = f(V_{t-1}, X_{t}) \\
    S_{t} & = g(H_{t} - V_{threshold}) = \Theta(H_{t} - V_{threshold})

where :math:`\Theta(x)` is the ``surrogate_function()`` in the parameters, which is a heaviside step function:

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

The hard reset is:

.. math::
    V_{t} = H_{t} \cdot (1 - S_{t}) + V_{reset} \cdot S_{t}

The soft reset is:

.. math::
    V_{t} = H_{t} - V_{threshold} \cdot S_{t}

where :math:`V_{t}` is the membrane potential of the neuron, :math:`X_{t}` is the external input, such as voltage increment.
To avoid confusion, we use :math:`H_{t}` to represent the membrane potential after neuronal charge but before neuronal fire,
:math:`V_{t}` is the membrane potential after the neuronal fire, :math:`f(V(t-1), X(t))` is the neuronal charge function.
The difference between neurons is the neuronal charge.

Clock-driven Simulation
---------------------------

:class:`spikingjelly.clock_driven` uses a clock-driven approach to simulate SNN.

Next, we will stimulate the neuron and check its membrane potential and output spikes.

Now let us give constant input to the LIF neurons layer and plot the membrane potential and output spikes:

.. code-block:: python

    lif.reset()
    x = torch.as_tensor([2.])
    T = 150
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(lif(x))
        v_list.append(lif.v)

    visualizing.plot_one_neuron_v_s(np.asarray(v_list), np.asarray(s_list), v_threshold=lif.v_threshold, v_reset=lif.v_reset,
                                    dpi=200)
    plt.show()

The input is with ``shape=[1]``, and this LIF neurons layer has only 1 neuron. Its membrane potential and output spikes change with time-step as follows:

.. image:: ../_static/tutorials/clock_driven/0_neuron/0.*
    :width: 100%

We reset the neurons layer and give an input with ``shape=[32]`` to see the membrane potential and output spikes of these 32 neurons:

.. code-block:: python

    lif.reset()
    x = torch.rand(size=[32]) * 4
    T = 50
    s_list = []
    v_list = []
    for t in range(T):
        s_list.append(lif(x).unsqueeze(0))
        v_list.append(lif.v.unsqueeze(0))

    s_list = torch.cat(s_list)
    v_list = torch.cat(v_list)

    visualizing.plot_2d_heatmap(array=np.asarray(v_list), title='Membrane Potentials', xlabel='Simulating Step',
                                ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
    visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                               ylabel='Neuron Index', dpi=200)
    plt.show()

The results are as follows:

.. image:: ../_static/tutorials/clock_driven/0_neuron/1.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/0_neuron/2.*
    :width: 100%