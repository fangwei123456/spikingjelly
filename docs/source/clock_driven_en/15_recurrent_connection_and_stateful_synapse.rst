Recurrent Connections and Stateful Synapses
======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

Recurrent Connections
-----------------------
The recurrent connections connect a module's outputs to its inputs. For example, [#Effective]_ uses a SRNN(recurrent
networks of spiking neurons), which is shown in the following figure:

.. image:: ../_static/tutorials/clock_driven/15_recurrent_connection_and_stateful_synapse/SRNN_example.*
    :width: 100%

It is easy to use SpikingJelly to implement the recurrent module. Considering a simple case that we add a connection to make
the neuron's outputs :math:`s[t]` at time-step :math:`t` can add with external inputs :math:`x[t+1]` at time-step :math:`t+1`.
It can be implemented by :class:`spikingjelly.clock_driven.layer.ElementWiseRecurrentContainer`. ``ElementWiseRecurrentContainer``
is a container that add a recurrent connection to the contained ``sub_module``. The connection is a user-defined element-wise
function :math:`z=f(x, y)`. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]` and :math:`y[t]` (Note that
:math:`y[t]` is also the outputs of this module), and the inputs of this module as :math:`x[t]`, then

.. math::

    i[t] = f(x[t], y[t-1])

where :math:`f` is the user-defined element-wise function. We set :math:`y[-1] = 0`.

Let us use ``ElementWiseRecurrentContainer`` to contain a IF neuron, and set the element-wise function as `add`:

.. math::

    i[t] = x[t] + y[t-1].

We use soft reset, and give the inputs as :math:`x[t]=[1.5, 0, ..., 0]`:

.. code-block:: python

    T = 8
    def element_wise_add(x, y):
        return x + y
    net = ElementWiseRecurrentContainer(neuron.IFNode(v_reset=None), element_wise_add)
    print(net)
    x = torch.zeros([T])
    x[0] = 1.5
    for t in range(T):
        print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

    functional.reset_net(net)

The outputs are:

.. code-block:: bash

    ElementWiseRecurrentContainer(
      element-wise function=<function element_wise_add at 0x000001FE0F7968B0>
      (sub_module): IFNode(
        v_threshold=1.0, v_reset=None, detach_reset=False
        (surrogate_function): Sigmoid(alpha=1.0, spiking=True)
      )
    )
    0 x[t]=1.5, s[t]=1.0
    1 x[t]=0.0, s[t]=1.0
    2 x[t]=0.0, s[t]=1.0
    3 x[t]=0.0, s[t]=1.0
    4 x[t]=0.0, s[t]=1.0
    5 x[t]=0.0, s[t]=1.0
    6 x[t]=0.0, s[t]=1.0
    7 x[t]=0.0, s[t]=1.0

We can find that due to the recurrent connection, even if :math:`x[t]=0` when :math:`t \\geu 1`, the neuron can still fire
because its output spike is fed back as input.

We can use :class:`spikingjelly.clock_driven.layer.LinearRecurrentContainer` to implement a more complex recurrent connections.

Stateful Synapses
-----------------------

There are many papers using stateful synapses, e.g., [#Unsupervised]_ [#Exploiting]_. We can put :class:`spikingjelly.clock_driven.layer.SynapseFilter` after a stateless synapse to get the stateful synapse:

.. code-block:: python

    stateful_conv = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
        SynapseFilter(tau=100, learnable=True)
    )


.. [#Effective] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

.. [#Unsupervised] Diehl P U, Cook M. Unsupervised learning of digit recognition using spike-timing-dependent plasticity[J]. Frontiers in computational neuroscience, 2015, 9: 99.

.. [#Exploiting] Fang H, Shrestha A, Zhao Z, et al. Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network[J].
