Clock driven: Neurons
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

This tutorial focuses on ``spikingjelly.clock_driven.neuron``, which introduces spiking neurons, and clock-driven
simulation methods.

Spiking neuron model
-----------------------------------------------
In ``spikingjelly``, we define a neuron which can only output spikes, i.e. 0 or 1, as a "spiking neuron".
Networks that use spiking neurons are called Spiking Neural Networks.
``spikingjelly.clock_driven.neuron`` defines various common spiking neuron models.
We take ``spikingjelly.clock_driven.neuron.LIFNode`` as an example to introduce spiking neurons.

First, we need to import the relevant modules:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.clock_driven import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt

And then create a new LIF neuron layer:

.. code-block:: python

    lif = neuron.LIFNode()

The LIF neuron layer has some parameters, which are explained in detail in the API documentation:

    - **tau** -- membrane potential time constant, which is shared by all neurons in this layer

    - **v_threshold** -- the threshold voltage of the neuron

    - **v_reset** -- the reset voltage of the neuron. If it is not ``None``, when the neuron releases a spike, the voltage will be reset to ``v_reset``; if it is set to ``None``, the voltage will be subtracted from ``v_threshold``

    - **surrogate_function** -- the surrogate function used to calculate the gradient of the spike function during back propagation

    - **monitor_state** -- whether to set up a monitor to save the voltages and spikes of the neurons. If it is ``True``, ``self.monitor`` is a dictionary, the keys include ``h``, ``v`` and ``s``, which record the voltage after charging, the voltage after releasing a spike, and the released spike respectively.

The corresponding value is a linked list. In order to save memory, the value stored in the list is the value of the original variable converted into a ``numpy`` array.
Also note that the ``self.reset()`` function will clear these linked lists.

The ``surrogate_function`` behaves exactly the same as the step function during forward propagation,
and we will introduce its working principle for back propagation later.

You may be curious about the number of neurons in this layer. For most neuron layers in ``spikingjelly.clock_driven.neuron``,
the number of neurons is automatically determined according to the ``shape`` of the received input after initialization or re-initialization by calling the ``reset()`` function.

Very similar to neurons in RNN, spiking neurons are also stateful, i.e., they have memory.
The state variable of a spiking neuron is generally its membrane potential :math:`V_{t}`.
Therefore, neurons in ``spikingjelly.clock_driven.neuron`` have state variable ``v``.
You can print out the membrane potential of the newly created LIF neuron layer:

.. code-block:: python

    print(lif.v)
    # 0.0

You can find that ``v`` is now ``0.0`` because we haven't given it any input yet.
We apply several different inputs and observe the ``shape`` of the voltage of the neuron,
which is consistent with the number of neurons:

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

So what is the relationship between :math:`V_{t}` and input :math:`X_{t}`? In a spiking neuron,
it not only depends on the input :math:`X_{t}` at the current moment,
but also on its membrane potential :math:`V_{t-1}` at the end of the previous moment.

Usually we use the sub-threshold (when the membrane potential does not exceed the threshold voltage ``V_{threshold}``)
charging differential equation :math:`\frac{\mathrm{d}V(t)}{\mathrm{d}t} = f(V(t), X(t))` to describe the continuous-time
spiking neuron charging process. For example, for LIF neurons, the charging equation is:

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

Where :math:`\tau_{m}` is the membrane potential time constant and :math:`V_{reset}` is the reset voltage. For such differential equations,
since :math:`X(t)` is not a constant, it is difficult to obtain a explicit analytical solution.

The neurons in ``spikingjelly.clock_driven.neuron`` use discrete difference equations to approximate continuous differential equations.
From the perspective of the difference equation, the charging equation of the LIF neuron is:

.. math::
    \tau_{m} (V_{t} - V_{t-1}) = -(V_{t-1} - V_{reset}) + X_{t}

Therefore, the expression of :math:`V_{t}` can be obtained as

.. math::
    V_{t} = f(V_{t-1}, X_{t}) = V_{t-1} + \frac{1}{\tau_{m}}(-(V_{t - 1} - V_{reset}) + X_{t})

The corresponding code can be found in ``neuronal_charge()`` of ``LIFNode``:

.. code-block:: python

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) / self.tau
        else:
            self.v += (dv - (self.v - self.v_reset)) / self.tau

Different neurons have different charging equations. However, when the membrane potential exceeds the threshold voltage,
the release of a spike and the reset of the membrane potential after releasing a spike are the same. Therefore,
they all inherit from ``BaseNode`` and share the same discharge and reset equations. The code for releasing a spike can
be found in ``neuronal_fire()`` of ``BaseNode``:

.. code-block:: python

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

``surrogate_function()`` is a step function during forward propagation, as long as the input is greater than or equal
to 0, it will return 1, otherwise it will return 0. We regard this kind of ``tensor`` whose elements are only 0 or 1 as spikes.

The release of a spike consumes the previously accumulated electric charge of the neuron, so there will be an
instantaneous decrease in the membrane potential, which is the reset of the membrane potential. In SNN, there are
two ways to realize membrane potential reset:

#. Hard method: After releasing a spike, the membrane potential is directly set to the reset voltage::math:`V = V_{reset}`

#. Soft method: After releasing a spike, the membrane potential subtracts the threshold voltage::math:`V = V - V_{threshold}`

It can be found that for neurons using the Soft method, there is no need to reset the voltage :math:`V_{reset}`.
For the neuron in ``spikingjelly.clock_driven.neuron``, when ``v_reset`` is set to the default value ``1.0``, the neuron uses the Hard mode;
if it is set to ``None``, the Soft mode will be used.
You can find the corresponding code in ``neuronal_reset()`` of ``BaseNode``:

.. code-block:: python

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1 - spike) * self.v + spike * self.v_reset


Three equations describing discrete spiking neurons
--------------------------------------------------------------

So far, we can use the three discrete equations of charging, discharging, and resetting to describe any discrete spiking neurons.
The charging and discharging equations are:

.. math::
    H_{t} & = f(V_{t-1}, X_{t}) \\
    S_{t} & = g(H_{t} - V_{threshold}) = \Theta(H_{t} - V_{threshold})

where :math:`\Theta(x)` is the ``surrogate_function()`` in the parameter list, which is a step function:

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

The hard reset equation is:

.. math::
    V_{t} = H_{t} \cdot (1 - S_{t}) + V_{reset} \cdot S_{t}

The soft reset equation is:

.. math::
    V_{t} = H_{t} - V_{threshold} \cdot S_{t}

where :math:`V_{t}` is the membrane potential of the neuron, :math:`X_{t}` is the external input, such as voltage increment.
To avoid confusion, we use :math:`H_{t}` to represent the membrane potential before the neuron releases a spike,
:math:`V_{t}` is the membrane potential after the neuron releases a spike, :math:`f(V(t-1), X(t))` is the update equation of the neuronal state.
The difference between different neurons is the update equation.

Clock-driven simulation
---------------------------

``spikingjelly.clock_driven`` uses a clock-driven approach to gradually simulate SNN.

Next, we will gradually stimulate the neuron and check its membrane potential and output spikes.
In order to record the data, we need to open the ``monitor`` of the neuron layer:

.. code-block:: python

    lif.set_monitor(True)

After turning on the monitor, the neuron layer will automatically record the charged membrane potential
``self.monitor['h']``, the output spikes ``self.monitor['s']``,
and the membrane potential after discharging ``self.monitor['v']`` in the dictionary ``self.monitor`` during simulation.

Now let us exert continuous inputs to the LIF neuron layer and plot the membrane potential and output spikes:

.. code-block:: python

    x = torch.Tensor([2.0])
    T = 150
    for t in range(T):
        lif(x)
    visualizing.plot_one_neuron_v_s(lif.monitor['v'], lif.monitor['s'], v_threshold=lif.v_threshold, v_reset=lif.v_reset, dpi=200)
    plt.show()

We exert an input with ``shape=[1]``, so this LIF neuron layer has only 1 neuron. Its membrane potential and output spikes change with time as follows:

.. image:: ../_static/tutorials/clock_driven/0_neuron/0.*
    :width: 100%

In the following, we reset the neuron layer and exert an input with ``shape=[32]`` to view the membrane potential and output spikes of these 32 neurons:

.. code-block:: python

    lif.reset()
    x = torch.rand(size=[32]) * 4
    T = 50
    for t in range(T):
        lif(x)

    visualizing.plot_2d_heatmap(array=np.asarray(lif.monitor['v']).T, title='Membrane Potentials', xlabel='Simulating Step',
                                        ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
    visualizing.plot_1d_spikes(spikes=np.asarray(lif.monitor['s']).T, title='Membrane Potentials', xlabel='Simulating Step',
                                        ylabel='Neuron Index', dpi=200)
    plt.show()

The results are as follows:

.. image:: ../_static/tutorials/clock_driven/0_neuron/1.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/0_neuron/2.*
    :width: 100%